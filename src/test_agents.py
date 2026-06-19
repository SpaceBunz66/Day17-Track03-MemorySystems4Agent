from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    """Build an isolated config for tests."""

    base_config = load_config(Path(__file__).resolve().parent.parent)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return replace(
        base_config,
        state_dir=state_dir,
        compact_threshold_tokens=120,
        compact_keep_messages=2,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, edited, and upserted."""

    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")

    assert store.read_text("dungct").startswith("# User.md")
    path = store.write_text("dungct", "# User.md\n\n## Facts\n- name: DũngCT\n")
    assert path.name == "User.md"
    assert "DũngCT" in store.read_text("dungct")

    assert store.edit_text("dungct", "DũngCT", "DũngCT Lab") is True
    assert store.edit_text("dungct", "missing", "noop") is False
    assert store.facts("dungct")["name"] == "DũngCT Lab"

    store.upsert_fact("dungct", "location", "Đà Nẵng")
    store.upsert_fact("dungct", "location", "Huế")
    assert store.facts("dungct")["location"] == "Huế"
    assert store.file_size("dungct") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    user_id = "compact-user"
    thread_id = "long-thread"

    for index in range(8):
        agent.reply(
            user_id,
            thread_id,
            f"Lượt {index}: mình đang viết một đoạn rất dài về memory compaction, "
            "prompt token cost, recall và trade-off để ép compact xảy ra nhanh trong test.",
        )

    context = agent.compact_memory.context(thread_id)
    assert agent.compaction_count(thread_id) > 0
    assert context["summary"]
    assert len(context["messages"]) <= config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    baseline.reply("dungct", "session-a", "Chào bạn, mình tên là DũngCT và đang ở Huế.")
    advanced.reply("dungct", "session-a", "Chào bạn, mình tên là DũngCT và đang ở Huế.")

    baseline_answer = baseline.reply("dungct", "session-b", "Mình tên gì và hiện đang ở đâu?")["answer"]
    advanced_answer = advanced.reply("dungct", "session-b", "Mình tên gì và hiện đang ở đâu?")["answer"]

    assert "DũngCT" not in baseline_answer
    assert "Huế" not in baseline_answer
    assert "DũngCT" in advanced_answer
    assert "Huế" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    long_message = (
        "Mình tên là DũngCT, đang làm MLOps engineer. "
        "Đây là đoạn dài về dependency management, compact memory, token cost, "
        "cross-session recall và cách giữ recent context đủ nhỏ để benchmark rõ ràng. "
    )

    for index in range(14):
        message = long_message + f"Lượt số {index}, vẫn muốn trả lời ngắn gọn có ví dụ thực chiến."
        baseline.reply("dungct", "same-long-thread", message)
        advanced.reply("dungct", "same-long-thread", message)

    assert advanced.compaction_count("same-long-thread") > 0
    assert advanced.prompt_token_usage("same-long-thread") < baseline.prompt_token_usage("same-long-thread")
