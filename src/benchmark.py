from __future__ import annotations

import json
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of conversations in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Return a fractional recall score for expected facts in the answer."""

    if not expected:
        return 1.0
    normalized_answer = _normalize(answer)
    matches = sum(1 for item in expected if _normalize(item) in normalized_answer)
    return matches / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Score offline response quality without an LLM judge."""

    recall = recall_points(answer, expected)
    normalized = _normalize(answer)
    uncertainty_penalty = 0.25 if "chua co du" in normalized or "khong co fact" in normalized else 0.0
    concise_bonus = 0.1 if 20 <= len(answer) <= 320 else 0.0
    structure_bonus = 0.1 if any(mark in answer for mark in (";", "-", "\n")) else 0.0
    return max(0.0, min(1.0, recall * 0.8 + concise_bonus + structure_bonus - uncertainty_penalty))


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations."""

    used_threads: set[str] = set()
    users = {conversation["user_id"] for conversation in conversations}
    before_memory = _memory_size(agent, users)
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        user_id = conversation["user_id"]
        thread_id = conversation["id"]
        used_threads.add(thread_id)

        for turn in conversation.get("turns", []):
            agent.reply(user_id, thread_id, turn)

        for index, question in enumerate(conversation.get("recall_questions", []), start=1):
            recall_thread_id = f"{thread_id}-recall-{index}"
            used_threads.add(recall_thread_id)
            result = agent.reply(user_id, recall_thread_id, question["question"])
            answer = result["answer"]
            expected = question.get("expected_contains", [])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    after_memory = _memory_size(agent, users)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread_id) for thread_id in used_threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread_id) for thread_id in used_threads),
        recall_score=round(_average(recall_scores), 3),
        response_quality=round(_average(quality_scores), 3),
        memory_growth_bytes=max(0, after_memory - before_memory),
        compactions=sum(agent.compaction_count(thread_id) for thread_id in used_threads),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a markdown table."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    values = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.3f}",
            f"{row.response_quality:.3f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]

    try:
        from tabulate import tabulate

        return tabulate(values, headers=headers, tablefmt="github")
    except ImportError:
        table = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        table.extend("| " + " | ".join(map(str, row)) + " |" for row in values)
        return "\n".join(table)


def main() -> None:
    """Run standard and long-context stress benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)
    standard_data = load_conversations(config.data_dir / "conversations.json")
    stress_data = load_conversations(config.data_dir / "advanced_long_context.json")

    with tempfile.TemporaryDirectory(prefix="memory_lab_benchmark_") as temp_root:
        temp_state = Path(temp_root)
        standard_config = replace(config, state_dir=temp_state / "standard")
        stress_config = replace(
            config,
            state_dir=temp_state / "stress",
            compact_threshold_tokens=max(350, config.compact_threshold_tokens // 2),
            compact_keep_messages=min(config.compact_keep_messages, 4),
        )
        standard_config.state_dir.mkdir(parents=True, exist_ok=True)
        stress_config.state_dir.mkdir(parents=True, exist_ok=True)

        standard_rows = [
            run_agent_benchmark(
                "Baseline", BaselineAgent(standard_config, force_offline=True), standard_data, standard_config
            ),
            run_agent_benchmark(
                "Advanced", AdvancedAgent(standard_config, force_offline=True), standard_data, standard_config
            ),
        ]
        stress_rows = [
            run_agent_benchmark("Baseline", BaselineAgent(stress_config, force_offline=True), stress_data, stress_config),
            run_agent_benchmark("Advanced", AdvancedAgent(stress_config, force_offline=True), stress_data, stress_config),
        ]

        print("## Standard Benchmark")
        print(format_rows(standard_rows))
        print()
        print("## Long-Context Stress Benchmark")
        print(format_rows(stress_rows))
        print()
        print(
            "Note: Advanced may spend more tokens on short conversations because it loads User.md, "
            "but compact memory should reduce prompt tokens processed on long conversations."
        )


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or "")
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    without_marks = without_marks.replace("Đ", "D").replace("đ", "d")
    return re.sub(r"\s+", " ", without_marks.casefold()).strip()


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _memory_size(agent, user_ids: set[str]) -> int:
    if not hasattr(agent, "memory_file_size"):
        return 0
    return sum(agent.memory_file_size(user_id) for user_id in user_ids)


if __name__ == "__main__":
    main()
