from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Baseline Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return a response and token accounting for one user turn."""

        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(thread_id, message)
            except Exception:
                return self._reply_offline(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic within-session behavior for offline tests."""

        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({"role": "user", "content": message})
        prompt_tokens = self._estimate_prompt_tokens(state.messages)
        answer = self._offline_answer(thread_id, message)
        state.messages.append({"role": "assistant", "content": answer})

        answer_tokens = estimate_tokens(answer)
        state.token_usage += answer_tokens
        state.prompt_tokens_processed += prompt_tokens
        return {
            "agent": "baseline",
            "thread_id": thread_id,
            "answer": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
        }

    def _maybe_build_langchain_agent(self):
        """Build a simple live chat model when provider dependencies are present."""

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({"role": "user", "content": message})
        prompt_tokens = self._estimate_prompt_tokens(state.messages)
        response = self.langchain_agent.invoke(state.messages)
        answer = str(getattr(response, "content", response))
        state.messages.append({"role": "assistant", "content": answer})

        answer_tokens = estimate_tokens(answer)
        state.token_usage += answer_tokens
        state.prompt_tokens_processed += prompt_tokens
        return {
            "agent": "baseline",
            "thread_id": thread_id,
            "answer": answer,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
        }

    def _offline_answer(self, thread_id: str, message: str) -> str:
        facts = self._facts_in_thread(thread_id)
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
            )
        )

        if recall_intent:
            selected = self._select_facts_for_question(facts, lower)
            if not selected:
                return "Mình chưa có đủ thông tin trong thread này để trả lời chắc chắn."
            return "Trong thread này mình nhớ: " + "; ".join(
                f"{label}: {value}" for label, value in selected.items()
            ) + "."

        updates = extract_profile_updates(message)
        if updates:
            return "Mình đã ghi nhận tạm trong thread này: " + "; ".join(
                f"{key}: {value}" for key, value in updates.items()
            ) + "."
        return "Mình đã nhận tin nhắn này, nhưng baseline chỉ giữ ngữ cảnh trong thread hiện tại."

    def _facts_in_thread(self, thread_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for item in self.sessions.get(thread_id, SessionState()).messages:
            if item.get("role") == "user":
                facts.update(extract_profile_updates(item.get("content", "")))
        return facts

    @staticmethod
    def _select_facts_for_question(facts: dict[str, str], lower_question: str) -> dict[str, str]:
        wanted: dict[str, tuple[str, str]] = {
            "name": ("tên", "Tên"),
            "profession": ("nghề|làm", "Nghề nghiệp"),
            "location": ("ở đâu|nơi ở|đang ở", "Nơi ở"),
            "response_style": ("style|kiểu trả lời|trả lời", "Style trả lời"),
            "favorite_drink": ("đồ uống|uống", "Đồ uống yêu thích"),
            "favorite_food": ("món ăn|ăn", "Món ăn yêu thích"),
            "pet": ("nuôi|corgi|con gì", "Thú cưng"),
            "technical_interests": ("quan tâm|kỹ thuật|python|ai|tóm tắt|biết", "Mối quan tâm"),
        }
        selected: dict[str, str] = {}
        for key, (pattern, label) in wanted.items():
            if key in facts and re_search(pattern, lower_question):
                selected[label] = facts[key]
        if not selected and facts:
            for key in ("name", "profession", "location", "technical_interests", "response_style"):
                if key in facts:
                    selected[key] = facts[key]
        return selected

    @staticmethod
    def _estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
        return sum(estimate_tokens(f"{item.get('role', '')}: {item.get('content', '')}") for item in messages)


def re_search(pattern: str, text: str) -> bool:
    import re

    return re.search(pattern, text, flags=re.IGNORECASE) is not None
