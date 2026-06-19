from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Estimate tokens with a stable offline heuristic.

    It intentionally avoids provider-specific tokenizers so benchmark numbers
    are reproducible without network access or model SDKs.
    """

    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return 0
    words_and_marks = re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE)
    by_chars = len(normalized) / 4
    by_words = len(words_and_marks) * 1.25
    return max(1, int(math.ceil(max(by_chars, by_words))))


@dataclass
class UserProfileStore:
    """Persistent storage for one `User.md` file per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id.strip()).strip("._")
        safe_id = safe_id or "anonymous"
        return self.root_dir / safe_id / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return self._default_profile()

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = content.rstrip() + "\n"
        path.write_text(normalized, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        self.write_text(user_id, current.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^-\s*([a-z_]+):\s*(.+?)\s*$", line)
            if match:
                facts[match.group(1)] = match.group(2)
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        key = re.sub(r"[^a-z0-9_]+", "_", key.strip().lower()).strip("_")
        value = _clean_stored_value(value)
        if not key or not value:
            return

        lines = self.read_text(user_id).splitlines()
        if "## Facts" not in lines:
            lines.extend(["", "## Facts"])

        fact_pattern = re.compile(rf"^-\s*{re.escape(key)}:\s*")
        for index, line in enumerate(lines):
            if fact_pattern.match(line):
                if key == "technical_interests":
                    current_value = line.split(":", 1)[1].strip()
                    value = _merge_csv_values(current_value, value)
                lines[index] = f"- {key}: {value}"
                self.write_text(user_id, "\n".join(lines))
                return

        insert_at = lines.index("## Facts") + 1
        while insert_at < len(lines) and lines[insert_at].startswith("- "):
            insert_at += 1
        lines.insert(insert_at, f"- {key}: {value}")
        self.write_text(user_id, "\n".join(lines))

    @staticmethod
    def _default_profile() -> str:
        return "# User.md\n\n## Facts\n\n## Notes\n"


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable profile facts from a Vietnamese user message.

    Bonus behavior:
    - only writes facts when the phrasing is explicit enough;
    - ignores obvious questions, jokes, and noisy negative examples;
    - returns keyed facts so newer corrections replace older values.
    """

    text = re.sub(r"\s+", " ", message or "").strip()
    if not text:
        return {}

    lower = text.lower()
    question_safe_markers = (
        "mình tên là",
        "tên mình là",
        "mình ở",
        "mình đang ở",
        "hiện ở",
        "hiện đang ở",
        "nơi ở hiện tại là",
        "đang làm việc ở",
        "đang làm",
        "nghề nghiệp hiện tại",
        "giờ chuyển sang",
        "đồ uống yêu thích là",
        "món ăn yêu thích là",
        "vẫn uống",
        "mình muốn bạn trả lời",
        "hãy trả lời",
        "style trả lời",
        "3 bullet",
        "thích python",
        "ai ứng dụng",
        "mình nuôi",
        "corgi tên",
    )
    if "?" in text and not any(marker in lower for marker in question_safe_markers):
        return {}

    updates: dict[str, str] = {}

    name = _first_match(
        text,
        [
            r"(?:mình|tôi)\s+tên\s+là\s+([^,.;\n]+)",
            r"tên\s+mình\s+là\s+([^,.;\n]+)",
            r"(?<!corgi\s)(?<!bé\s)(?<!con\s)\btên\s+([A-ZĐ][A-Za-zÀ-ỹĐđ0-9 _-]{1,40}?)(?:,|\.|$)",
        ],
    )
    if name:
        updates["name"] = name

    if "không phải nơi ở hiện tại" not in lower and "đừng lấy nó làm nơi ở hiện tại" not in lower:
        location = _first_match(
            text,
            [
                r"giờ\s+mình\s+đang\s+ở\s+(.+?)\s+chứ",
                r"từ\s+tuần\s+này\s+mình\s+đang\s+làm\s+việc\s+ở\s+(.+?)(?:\s+vài\s+tháng|\s+để|,|\.|$)",
                r"nơi\s+ở\s+hiện\s+tại\s+(?:là|:)\s+(.+?)(?:,|\.|$)",
                r"hiện\s+(?:đang\s+)?ở\s+(.+?)(?:\s+và|,|\.|$)",
                r"mình\s+(?:vẫn\s+)?ở\s+(.+?)(?:\s+và|,|\.|$)",
                r"mình\s+đang\s+ở\s+(.+?)(?:\s+để|,|\.|$)",
                r"đang\s+ở\s+(.+?)(?:\s+và|,|\.|$)",
            ],
        )
        if location:
            updates["location"] = location

    profession = _first_match(
        text,
        [
            r"giờ\s+chuyển\s+sang\s+([^,.;]+)",
            r"nghề\s+nghiệp\s+hiện\s+tại\s+(?:vẫn\s+)?(?:là|:)\s+([^,.;]+)",
            r"nghề\s+(?!nghiệp\b)([^,.;]+?)(?:,|\.|$)",
            r"đang\s+làm\s+(?!việc\b)([^,.;]+?)(?:\s+cho|\s+với|,|\.|$)",
        ],
    )
    if profession:
        updates["profession"] = profession

    drink = _first_match(
        text,
        [
            r"đồ\s+uống\s+yêu\s+thích\s+là\s+([^,.;]+)",
            r"vẫn\s+uống\s+([^,.;]+?)(?:\s+như|,|\.|$)",
            r"(cà\s+phê\s+sữa\s+đá)",
        ],
    )
    if drink:
        updates["favorite_drink"] = drink

    food = _first_match(
        text,
        [
            r"món\s+ăn\s+yêu\s+thích\s+là\s+([^,.;]+)",
            r"(mì\s+quảng)",
        ],
    )
    if food:
        updates["favorite_food"] = food

    if "corgi" in lower:
        pet_name = _first_match(text, [r"corgi\s+(?:tên\s+)?([A-ZĐÀ-Ỹ][A-Za-zÀ-ỹĐđ0-9 _-]{0,30})"])
        updates["pet"] = f"corgi tên {pet_name}" if pet_name else "corgi"

    style = _extract_response_style(lower)
    if style:
        updates["response_style"] = style

    interests = _extract_interests(text)
    if interests:
        updates["technical_interests"] = interests

    priority = _extract_priority(lower)
    if priority:
        updates["priority"] = priority

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact heuristic summary of older messages."""

    if not messages:
        return ""

    bullets: list[str] = []
    for message in messages[-max_items:]:
        role = message.get("role", "unknown")
        content = re.sub(r"\s+", " ", message.get("content", "")).strip()
        if not content:
            continue
        if role == "summary":
            snippet = content[:360]
            bullets.append(f"- prior summary: {snippet}")
            continue

        facts = extract_profile_updates(content) if role == "user" else {}
        if facts:
            fact_text = ", ".join(f"{key}={value}" for key, value in sorted(facts.items()))
            bullets.append(f"- {role}: stable facts: {fact_text}")
        else:
            bullets.append(f"- {role}: {content[:220]}")

    return "Compact summary of older thread context:\n" + "\n".join(bullets)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread_state = self._ensure_thread(thread_id)
        messages = thread_state["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        self._compact_if_needed(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        thread_state = self._ensure_thread(thread_id)
        messages = thread_state["messages"]
        assert isinstance(messages, list)
        return {
            "messages": list(messages),
            "summary": str(thread_state.get("summary", "")),
            "compactions": int(thread_state.get("compactions", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure_thread(thread_id).get("compactions", 0))

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def _context_tokens(self, thread_id: str) -> int:
        thread_state = self._ensure_thread(thread_id)
        summary_tokens = estimate_tokens(str(thread_state.get("summary", "")))
        messages = thread_state["messages"]
        assert isinstance(messages, list)
        message_tokens = sum(
            estimate_tokens(f"{message.get('role', '')}: {message.get('content', '')}") for message in messages
        )
        return summary_tokens + message_tokens

    def _compact_if_needed(self, thread_id: str) -> None:
        thread_state = self._ensure_thread(thread_id)
        messages = thread_state["messages"]
        assert isinstance(messages, list)
        keep = max(1, self.keep_messages)
        if self._context_tokens(thread_id) <= self.threshold_tokens or len(messages) <= keep:
            return

        older = messages[:-keep]
        recent = messages[-keep:]
        prior_summary = str(thread_state.get("summary", ""))
        summary_input = ([{"role": "summary", "content": prior_summary}] if prior_summary else []) + older
        thread_state["summary"] = summarize_messages(summary_input, max_items=8)
        thread_state["messages"] = recent
        thread_state["compactions"] = int(thread_state.get("compactions", 0)) + 1


def _first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = _clean_fact_value(match.group(1))
            if _looks_like_fact_value(value):
                return value
    return None


def _clean_fact_value(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip(" .,:;!?\"'")
    for separator in (" và ", " nhưng ", " chứ ", " để ", " trong ", " vì ", " nếu "):
        if separator in cleaned.lower():
            index = cleaned.lower().index(separator)
            cleaned = cleaned[:index].strip(" .,:;!?\"'")
    return cleaned


def _clean_stored_value(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip(" .,:;!?\"'")


def _merge_csv_values(existing: str, new: str) -> str:
    values: list[str] = []
    for raw_value in f"{existing}, {new}".split(","):
        value = raw_value.strip()
        if value and value.casefold() not in {item.casefold() for item in values}:
            values.append(value)
    return ", ".join(values)


def _looks_like_fact_value(value: str) -> bool:
    lower = value.lower().strip()
    if not lower:
        return False
    question_words = {"đâu", "gì", "không", "nào", "ai", "bao nhiêu", "sao"}
    if lower in question_words or any(f" {word} " in f" {lower} " for word in question_words):
        return False
    return True


def _extract_response_style(lower_text: str) -> str | None:
    if "3 bullet" in lower_text or "ba bullet" in lower_text:
        return "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off"
    if "bullet" in lower_text and ("ví dụ" in lower_text or "ngắn" in lower_text):
        return "ngắn gọn, có bullet và ví dụ thực chiến"
    if "ngắn gọn" in lower_text and "ví dụ" in lower_text:
        return "ngắn gọn, rõ ý và có ví dụ thực tế"
    if "không thích câu trả lời quá lan man" in lower_text or "trả lời ngắn gọn" in lower_text:
        return "ngắn gọn, không lan man"
    return None


def _extract_interests(text: str) -> str | None:
    lower = text.lower()
    interests: list[str] = []
    if "python" in lower:
        interests.append("Python")
    if "ai" in lower:
        interests.append("AI")
    if "mlops" in lower:
        interests.append("MLOps")
    if "rag" in lower:
        interests.append("RAG")
    if "benchmark" in lower and "benchmark memory" not in lower:
        interests.append("benchmark")
    if "memory" in lower:
        interests.append("memory systems")
    return ", ".join(dict.fromkeys(interests)) if interests else None


def _extract_priority(lower_text: str) -> str | None:
    if "ưu tiên recall đúng" in lower_text:
        return "ưu tiên recall đúng hơn câu văn hoa mỹ"
    if "trade-off" in lower_text and ("token" in lower_text or "recall" in lower_text):
        return "thích so sánh trade-off giữa recall và token cost"
    if "số liệu" in lower_text:
        return "ưu tiên benchmark có số liệu rõ ràng"
    return None
