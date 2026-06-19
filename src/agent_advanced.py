from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between live provider mode and deterministic offline mode."""

        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                return self._reply_offline(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path with persistent and compact memory."""

        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)

        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return {
            "agent": "advanced",
            "thread_id": thread_id,
            "answer": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate profile + compact summary + recent messages for one turn."""

        profile_tokens = estimate_tokens(self.profile_store.read_text(user_id))
        context = self.compact_memory.context(thread_id)
        summary_tokens = estimate_tokens(str(context.get("summary", "")))
        message_tokens = sum(
            estimate_tokens(f"{item.get('role', '')}: {item.get('content', '')}")
            for item in context.get("messages", [])
        )
        return profile_tokens + summary_tokens + message_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory."""

        facts = self.profile_store.facts(user_id)
        lower = message.lower()
        recall_intent = any(
            phrase in lower
            for phrase in (
                "mình tên gì",
                "tên mình",
                "hiện tại mình",
                "mình làm nghề gì",
                "nghề hiện tại",
                "style trả lời",
                "kiểu trả lời",
                "đồ uống",
                "món ăn",
                "nuôi con gì",
                "ở đâu",
                "tóm tắt",
                "bạn biết",
                "nhắc lại",
                "đâu mới là",
            )
        )

        if recall_intent:
            selected = self._select_facts_for_question(facts, lower)
            if selected:
                return "Mình nhớ từ User.md: " + "; ".join(
                    f"{label}: {value}" for label, value in selected.items()
                ) + "."
            return "Mình chưa có fact đủ chắc trong User.md cho câu hỏi này."

        updates = extract_profile_updates(message)
        if updates:
            return "Mình đã cập nhật User.md: " + "; ".join(
                f"{key}: {value}" for key, value in updates.items()
            ) + "."

        context = self.compact_memory.context(thread_id)
        if context.get("summary"):
            return "Mình đã giữ ý chính trong compact summary và sẽ trả lời ngắn gọn theo profile."
        return "Mình đã nhận ngữ cảnh mới và sẽ dùng User.md cho các fact ổn định."

    def _maybe_build_langchain_agent(self):
        """Build a live chat model when optional provider packages are present."""

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        context = self.compact_memory.context(thread_id)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an assistant with persistent User.md memory. "
                    "Use the profile facts when relevant and keep answers concise.\n\n"
                    f"{self.profile_store.read_text(user_id)}\n\n"
                    f"Compact summary:\n{context.get('summary', '')}"
                ),
            },
            *context.get("messages", []),
        ]
        response = self.langchain_agent.invoke(prompt)
        answer = str(getattr(response, "content", response))
        self.compact_memory.append(thread_id, "assistant", answer)

        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return {
            "agent": "advanced",
            "thread_id": thread_id,
            "answer": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
            "compactions": self.compaction_count(thread_id),
        }

    @staticmethod
    def _select_facts_for_question(facts: dict[str, str], lower_question: str) -> dict[str, str]:
        wanted: dict[str, tuple[tuple[str, ...], str]] = {
            "name": (("tên", "biết", "tóm tắt"), "Tên"),
            "profession": (("nghề", "làm", "ai là", "tóm tắt", "đâu mới là"), "Nghề nghiệp hiện tại"),
            "location": (("ở đâu", "nơi ở", "đang ở", "hiện tại", "đâu mới là"), "Nơi ở hiện tại"),
            "response_style": (("style", "kiểu trả lời", "trả lời", "3 bullet"), "Style trả lời"),
            "favorite_drink": (("đồ uống", "uống"), "Đồ uống yêu thích"),
            "favorite_food": (("món ăn", "ăn"), "Món ăn yêu thích"),
            "pet": (("nuôi", "corgi", "con gì"), "Thú cưng"),
            "technical_interests": (("quan tâm", "kỹ thuật", "python", "ai", "tóm tắt", "biết"), "Mối quan tâm"),
            "priority": (("trade-off", "token", "recall", "số liệu"), "Ưu tiên"),
        }

        selected: dict[str, str] = {}
        for key, (phrases, label) in wanted.items():
            if key in facts and any(phrase in lower_question for phrase in phrases):
                selected[label] = facts[key]

        if not selected and facts:
            for key, label in (
                ("name", "Tên"),
                ("profession", "Nghề nghiệp hiện tại"),
                ("location", "Nơi ở hiện tại"),
                ("response_style", "Style trả lời"),
            ):
                if key in facts:
                    selected[label] = facts[key]
        return selected
