"""
AUTONOMOUS LEARNING AI
Version 4.0

Python-only local learning system.

Main improvements:
- Much better question understanding
- Definition-aware retrieval
- Topic detection
- Synonym and alias handling
- Phrase matching
- Concept matching
- Question-type matching
- Definition prioritization
- Better answer construction
- Duplicate prevention
- Persistent memory
- Feedback learning
- Local knowledge learning
- Automatic background learning
- Conversation memory
- Search commands
- No external packages required
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import threading
import time

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "4.0"

KNOWLEDGE_DIR = Path("knowledge")
BRAIN_FILE = Path("autonomous_brain.json")
CONFIG_FILE = Path("autonomous_config.json")

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".log",
}

MAX_KNOWLEDGE_ITEMS = 100000
MAX_MEMORY_ITEMS = 10000
MAX_CONVERSATION_ITEMS = 10000
MAX_WORDS = 200000
MAX_ASSOCIATIONS = 200000

DEFAULT_CONFIG = {
    "auto_learning": True,
    "learning_interval_seconds": 60,
    "allow_web": False,
    "min_answer_score": 0.15,
    "max_search_results": 12,
    "remember_conversations": True,
}


# ============================================================
# WORD DATA
# ============================================================

STOP_WORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "at",
    "for", "from", "with", "by", "as", "and", "or",
    "but", "if", "then", "than", "so", "because",
    "this", "that", "these", "those", "it", "its",
    "they", "them", "their", "there", "here",
    "i", "me", "my", "mine", "we", "us", "our",
    "you", "your", "yours", "he", "she", "his", "her",
    "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "would",
    "should", "will", "shall", "may", "might",
    "about", "into", "over", "under", "up", "down",
    "more", "most", "some", "any", "many", "much",
    "very", "really", "just", "also", "too",
    "tell", "give", "show", "explain", "describe",
    "please", "help",
}


QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "can",
    "could",
    "does",
    "do",
    "did",
    "is",
    "are",
    "difference",
    "example",
}


# Common aliases and related forms.
ALIASES = {
    "py": "python",
    "python3": "python",
    "python2": "python",
    "functions": "function",
    "methods": "method",
    "variables": "variable",
    "lists": "list",
    "tuples": "tuple",
    "sets": "set",
    "dictionaries": "dictionary",
    "dict": "dictionary",
    "dicts": "dictionary",
    "loops": "loop",
    "classes": "class",
    "objects": "object",
    "modules": "module",
    "packages": "package",
    "libraries": "library",
    "exceptions": "exception",
    "errors": "error",
    "strings": "string",
    "integers": "integer",
    "floats": "float",
    "booleans": "boolean",
    "numbers": "number",
    "arguments": "argument",
    "parameters": "parameter",
    "programs": "program",
    "programming": "programming",
    "developers": "developer",
    "coding": "code",
    "codes": "code",
    "applications": "application",
    "apps": "application",
    "computer": "computing",
    "computers": "computing",
}


RELATED_TERMS = {
    "python": {
        "language",
        "programming",
        "code",
        "software",
        "program",
        "syntax",
        "developer",
    },

    "function": {
        "parameter",
        "argument",
        "return",
        "def",
        "call",
        "code",
        "function",
    },

    "variable": {
        "value",
        "name",
        "assignment",
        "data",
        "type",
    },

    "list": {
        "collection",
        "sequence",
        "item",
        "element",
        "index",
        "mutable",
    },

    "tuple": {
        "collection",
        "sequence",
        "item",
        "element",
        "index",
        "immutable",
    },

    "dictionary": {
        "mapping",
        "key",
        "value",
        "pair",
        "lookup",
    },

    "loop": {
        "iteration",
        "repeat",
        "for",
        "while",
        "condition",
    },

    "class": {
        "object",
        "instance",
        "method",
        "attribute",
        "inheritance",
    },

    "object": {
        "class",
        "instance",
        "attribute",
        "method",
    },

    "exception": {
        "error",
        "try",
        "except",
        "raise",
        "handling",
    },

    "string": {
        "text",
        "character",
        "sequence",
        "format",
    },
}


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize whitespace and invisible characters."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line)
        lines.append(line.strip())

    return "\n".join(lines).strip()


def normalize_for_search(text: str) -> str:
    """Normalize text specifically for searching."""
    text = text.lower()

    replacements = {
        "’": "'",
        "“": '"',
        "”": '"',
        "`": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-z0-9+#.\-_' ]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str) -> List[str]:
    """Return lowercase searchable tokens."""
    text = normalize_for_search(text)

    return re.findall(
        r"[a-zA-Z0-9_+#.-]+",
        text.lower()
    )


def canonical_word(word: str) -> str:
    """Convert words to their canonical form."""
    word = word.lower().strip()

    if word in ALIASES:
        return ALIASES[word]

    # Basic plural handling.
    if len(word) > 4:
        if word.endswith("ies"):
            return word[:-3] + "y"

        if word.endswith("es") and not word.endswith("ses"):
            return word[:-2]

        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]

    return word


def canonical_words(words: List[str]) -> List[str]:
    return [canonical_word(word) for word in words]


def content_words(text: str) -> List[str]:
    words = canonical_words(tokenize(text))

    return [
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 1
    ]


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def split_sentences(text: str) -> List[str]:
    """Split text into reasonably useful sentences."""
    text = normalize_text(text)

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        text
    )

    result = []

    for part in parts:
        part = part.strip()

        if len(part) >= 8:
            result.append(part)

    return result


def clean_answer(text: str) -> str:
    """Clean a retrieved answer before displaying it."""
    text = normalize_text(text)

    # Remove common source labels.
    text = re.sub(
        r"^(FACT|DEFINITION|CONCEPT|RULE|EXAMPLE|ANSWER|TERM|RELATION|NOTE)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip(" -–—:")

    # Avoid giant answers.
    if len(text) > 900:
        text = text[:900].rsplit(" ", 1)[0] + "..."

    return text


def text_similarity(a: str, b: str) -> float:
    """Simple Jaccard-style similarity."""
    a_words = set(content_words(a))
    b_words = set(content_words(b))

    if not a_words or not b_words:
        return 0.0

    intersection = len(a_words & b_words)
    union = len(a_words | b_words)

    return intersection / max(1, union)


def phrase_contains(text: str, phrase: str) -> bool:
    text = normalize_for_search(text)
    phrase = normalize_for_search(phrase)

    if not phrase:
        return False

    return phrase in text


# ============================================================
# QUESTION UNDERSTANDING
# ============================================================

@dataclass
class ParsedQuestion:
    original: str
    normalized: str
    question_type: str
    subject: str
    subject_words: List[str]
    important_words: List[str]
    phrases: List[str]
    intent_words: List[str]


class QuestionAnalyzer:
    """
    Converts a natural-language question into structured intent.

    This is the part that fixes the major problem where:
        "What is Python?"
    accidentally retrieves:
        "Python supports functional programming techniques."
    """

    DEFINITION_PATTERNS = [
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bdefine\b",
        r"\bdefinition of\b",
        r"\bmeaning of\b",
        r"\bwhat does .* mean\b",
        r"\bwhat's\b",
    ]

    HOW_PATTERNS = [
        r"\bhow does\b",
        r"\bhow do\b",
        r"\bhow can\b",
        r"\bhow to\b",
        r"\bhow is\b",
        r"\bhow are\b",
    ]

    WHY_PATTERNS = [
        r"\bwhy is\b",
        r"\bwhy are\b",
        r"\bwhy does\b",
        r"\bwhy do\b",
        r"\bwhy can\b",
    ]

    COMPARISON_PATTERNS = [
        r"\bdifference between\b",
        r"\bdifference of\b",
        r"\bcompare\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bwhich is different\b",
    ]

    EXAMPLE_PATTERNS = [
        r"\bgive me an example\b",
        r"\bexample of\b",
        r"\bshow me an example\b",
        r"\bexamples of\b",
    ]

    EXPLANATION_PATTERNS = [
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\btell me about\b",
        r"\bhow does .* work\b",
        r"\bwhat does .* do\b",
    ]

    WHEN_PATTERNS = [
        r"\bwhen was\b",
        r"\bwhen did\b",
        r"\bwhen is\b",
        r"\bwhen are\b",
    ]

    WHERE_PATTERNS = [
        r"\bwhere is\b",
        r"\bwhere are\b",
        r"\bwhere does\b",
        r"\bwhere can\b",
    ]

    WHO_PATTERNS = [
        r"\bwho is\b",
        r"\bwho was\b",
        r"\bwho are\b",
    ]

    @classmethod
    def detect_type(cls, question: str) -> str:
        q = question.lower().strip()

        for pattern in cls.COMPARISON_PATTERNS:
            if re.search(pattern, q):
                return "comparison"

        for pattern in cls.EXAMPLE_PATTERNS:
            if re.search(pattern, q):
                return "example"

        for pattern in cls.DEFINITION_PATTERNS:
            if re.search(pattern, q):
                return "definition"

        for pattern in cls.HOW_PATTERNS:
            if re.search(pattern, q):
                return "how"

        for pattern in cls.WHY_PATTERNS:
            if re.search(pattern, q):
                return "why"

        for pattern in cls.EXPLANATION_PATTERNS:
            if re.search(pattern, q):
                return "explanation"

        for pattern in cls.WHEN_PATTERNS:
            if re.search(pattern, q):
                return "when"

        for pattern in cls.WHERE_PATTERNS:
            if re.search(pattern, q):
                return "where"

        for pattern in cls.WHO_PATTERNS:
            if re.search(pattern, q):
                return "who"

        return "general"

    @classmethod
    def extract_subject(cls, question: str) -> str:
        q = question.strip().rstrip("?!. ")

        patterns = [
            r"what is (.+)",
            r"what are (.+)",
            r"define (.+)",
            r"definition of (.+)",
            r"meaning of (.+)",
            r"what does (.+) mean",
            r"explain (.+)",
            r"describe (.+)",
            r"tell me about (.+)",
            r"how does (.+) work",
            r"how do (.+) work",
            r"why is (.+)",
            r"why are (.+)",
            r"what can (.+) be used for",
            r"what does (.+) do",
            r"give me an example of (.+)",
            r"example of (.+)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                q,
                flags=re.IGNORECASE,
            )

            if match:
                subject = match.group(1).strip()

                subject = re.sub(
                    r"\b(in python|with python)\b",
                    "",
                    subject,
                    flags=re.IGNORECASE,
                )

                return subject.strip(" .?!")

        # Fall back to content words.
        words = content_words(q)

        return " ".join(words[:8])

    @classmethod
    def build_phrases(cls, text: str) -> List[str]:
        words = canonical_words(tokenize(text))

        phrases = []

        for size in (4, 3, 2):
            for i in range(len(words) - size + 1):
                chunk = words[i:i + size]

                if all(word not in STOP_WORDS for word in chunk):
                    phrases.append(" ".join(chunk))

        return unique_preserve_order(phrases)

    @classmethod
    def analyze(cls, question: str) -> ParsedQuestion:
        normalized = normalize_for_search(question)
        question_type = cls.detect_type(question)

        subject = cls.extract_subject(question)

        subject_words = unique_preserve_order(
            content_words(subject)
        )

        important = unique_preserve_order(
            content_words(question)
        )

        phrases = cls.build_phrases(question)

        intent_words = [
            word
            for word in tokenize(question)
            if word.lower() in QUESTION_WORDS
        ]

        return ParsedQuestion(
            original=question,
            normalized=normalized,
            question_type=question_type,
            subject=subject,
            subject_words=subject_words,
            important_words=important,
            phrases=phrases,
            intent_words=intent_words,
        )


# ============================================================
# KNOWLEDGE ITEM
# ============================================================

@dataclass
class KnowledgeItem:
    id: int
    kind: str
    text: str
    source: str
    topic: str
    keywords: List[str]
    created: float
    times_used: int = 0
    useful_votes: int = 0
    bad_votes: int = 0

    def feedback_score(self) -> float:
        total = self.useful_votes + self.bad_votes

        if total == 0:
            return 0.0

        return (
            self.useful_votes - self.bad_votes
        ) / total


# ============================================================
# CONFIG MANAGER
# ============================================================

class ConfigManager:

    def __init__(self, path: Path):
        self.path = path
        self.data = dict(DEFAULT_CONFIG)
        self.load()

    def load(self):
        if not self.path.exists():
            self.save()
            return

        try:
            with self.path.open(
                "r",
                encoding="utf-8",
            ) as f:
                loaded = json.load(f)

            if isinstance(loaded, dict):
                self.data.update(loaded)

        except Exception:
            self.data = dict(DEFAULT_CONFIG)

    def save(self):
        try:
            with self.path.open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    self.data,
                    f,
                    indent=2,
                )
        except Exception:
            pass

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        self.data[key] = value
        self.save()


# ============================================================
# BRAIN
# ============================================================

class Brain:

    def __init__(self, path: Path):
        self.path = path

        self.knowledge: List[KnowledgeItem] = []

        self.word_counts: Counter = Counter()
        self.topic_counts: Counter = Counter()
        self.kind_counts: Counter = Counter()

        self.associations: Dict[str, Counter] = defaultdict(Counter)

        self.source_counts: Counter = Counter()

        self.file_hashes: Dict[str, str] = {}

        self.memories: List[Dict[str, Any]] = []
        self.conversations: List[Dict[str, Any]] = []

        self.question_history: List[Dict[str, Any]] = []

        self.next_id = 1

        self.lock = threading.RLock()

        self.load()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    def load(self):
        if not self.path.exists():
            self.save()
            return

        try:
            with self.path.open(
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            self.next_id = int(
                data.get("next_id", 1)
            )

            raw_knowledge = data.get(
                "knowledge",
                [],
            )

            self.knowledge = []

            for item in raw_knowledge:
                try:
                    self.knowledge.append(
                        KnowledgeItem(
                            id=int(item["id"]),
                            kind=str(item.get("kind", "fact")),
                            text=str(item.get("text", "")),
                            source=str(item.get("source", "")),
                            topic=str(item.get("topic", "")),
                            keywords=list(
                                item.get("keywords", [])
                            ),
                            created=float(
                                item.get(
                                    "created",
                                    time.time(),
                                )
                            ),
                            times_used=int(
                                item.get(
                                    "times_used",
                                    0,
                                )
                            ),
                            useful_votes=int(
                                item.get(
                                    "useful_votes",
                                    0,
                                )
                            ),
                            bad_votes=int(
                                item.get(
                                    "bad_votes",
                                    0,
                                )
                            ),
                        )
                    )
                except Exception:
                    continue

            self.memories = list(
                data.get("memories", [])
            )

            self.conversations = list(
                data.get("conversations", [])
            )

            self.question_history = list(
                data.get("question_history", [])
            )

            self.file_hashes = dict(
                data.get("file_hashes", {})
            )

            self.rebuild_indexes()

        except Exception as exc:
            print(
                f"Warning: could not load brain: {exc}"
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save(self):
        with self.lock:
            data = {
                "version": VERSION,
                "next_id": self.next_id,
                "knowledge": [
                    asdict(item)
                    for item in self.knowledge[
                        -MAX_KNOWLEDGE_ITEMS:
                    ]
                ],
                "memories": self.memories[
                    -MAX_MEMORY_ITEMS:
                ],
                "conversations": self.conversations[
                    -MAX_CONVERSATION_ITEMS:
                ],
                "question_history": self.question_history[
                    -MAX_CONVERSATION_ITEMS:
                ],
                "file_hashes": self.file_hashes,
            }

            temp_path = self.path.with_suffix(
                ".tmp"
            )

            try:
                with temp_path.open(
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        data,
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                temp_path.replace(self.path)

            except Exception:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass

    # --------------------------------------------------------
    # INDEXES
    # --------------------------------------------------------

    def rebuild_indexes(self):
        self.word_counts = Counter()
        self.topic_counts = Counter()
        self.kind_counts = Counter()
        self.associations = defaultdict(Counter)
        self.source_counts = Counter()

        for item in self.knowledge:

            words = canonical_words(
                tokenize(item.text)
            )

            for word in words:
                if word not in STOP_WORDS:
                    self.word_counts[word] += 1

            if item.topic:
                self.topic_counts[
                    canonical_word(item.topic)
                ] += 1

            self.kind_counts[item.kind] += 1

            self.source_counts[item.source] += 1

            unique_words = unique_preserve_order(
                [
                    word
                    for word in words
                    if word not in STOP_WORDS
                ]
            )

            # Limit association growth.
            unique_words = unique_words[:100]

            for a in unique_words:
                for b in unique_words:
                    if a != b:
                        self.associations[a][b] += 1

        # Trim associations.
        for word in list(self.associations.keys()):
            self.associations[word] = Counter(
                dict(
                    self.associations[word].most_common(50)
                )
            )

    # --------------------------------------------------------
    # KNOWLEDGE
    # --------------------------------------------------------

    def add_knowledge(
        self,
        kind: str,
        text: str,
        source: str = "",
        topic: str = "",
        keywords: Optional[List[str]] = None,
    ) -> Optional[KnowledgeItem]:

        text = clean_answer(text)

        if len(text) < 5:
            return None

        normalized = normalize_for_search(text)

        with self.lock:

            # Duplicate check.
            for existing in self.knowledge[-5000:]:
                if (
                    normalize_for_search(
                        existing.text
                    ) == normalized
                ):
                    return None

            item = KnowledgeItem(
                id=self.next_id,
                kind=kind.lower().strip(),
                text=text,
                source=source,
                topic=topic,
                keywords=unique_preserve_order(
                    canonical_words(
                        keywords or content_words(text)
                    )
                )[:100],
                created=time.time(),
            )

            self.next_id += 1

            self.knowledge.append(item)

            if len(self.knowledge) > MAX_KNOWLEDGE_ITEMS:
                self.knowledge = self.knowledge[
                    -MAX_KNOWLEDGE_ITEMS:
                ]

            self._index_item(item)

            return item

    def _index_item(self, item: KnowledgeItem):
        words = canonical_words(
            tokenize(item.text)
        )

        useful_words = [
            word
            for word in words
            if word not in STOP_WORDS
        ]

        for word in useful_words:
            self.word_counts[word] += 1

        if item.topic:
            self.topic_counts[
                canonical_word(item.topic)
            ] += 1

        self.kind_counts[item.kind] += 1
        self.source_counts[item.source] += 1

        unique_words = unique_preserve_order(
            useful_words
        )[:100]

        for a in unique_words:
            for b in unique_words:
                if a != b:
                    self.associations[a][b] += 1

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    def add_memory(
        self,
        text: str,
        category: str = "general",
    ):
        text = clean_answer(text)

        if len(text) < 5:
            return

        with self.lock:
            self.memories.append({
                "text": text,
                "category": category,
                "created": time.time(),
            })

            self.memories = self.memories[
                -MAX_MEMORY_ITEMS:
            ]

    def add_conversation(
        self,
        question: str,
        answer: str,
    ):
        with self.lock:
            self.conversations.append({
                "question": question,
                "answer": answer,
                "created": time.time(),
            })

            self.conversations = self.conversations[
                -MAX_CONVERSATION_ITEMS:
            ]

    def record_question(
        self,
        question: str,
        question_type: str,
        subject: str,
    ):
        self.question_history.append({
            "question": question,
            "type": question_type,
            "subject": subject,
            "created": time.time(),
        })

        self.question_history = self.question_history[
            -MAX_CONVERSATION_ITEMS:
        ]

    # --------------------------------------------------------
    # FEEDBACK
    # --------------------------------------------------------

    def vote(
        self,
        item_id: int,
        useful: bool,
    ) -> bool:

        for item in self.knowledge:

            if item.id == item_id:

                if useful:
                    item.useful_votes += 1
                else:
                    item.bad_votes += 1

                self.save()

                return True

        return False

    # --------------------------------------------------------
    # FILE HASHES
    # --------------------------------------------------------

    def get_file_hash(
        self,
        path: str,
    ) -> Optional[str]:

        return self.file_hashes.get(path)

    def set_file_hash(
        self,
        path: str,
        file_hash: str,
    ):
        self.file_hashes[path] = file_hash

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def stats(self) -> Dict[str, Any]:

        return {
            "knowledge": len(self.knowledge),
            "words": len(self.word_counts),
            "topics": len(self.topic_counts),
            "memories": len(self.memories),
            "conversations": len(self.conversations),
            "questions": len(self.question_history),
            "sources": len(self.source_counts),
        }


# ============================================================
# KNOWLEDGE EXTRACTOR
# ============================================================

class KnowledgeExtractor:
    """
    Turns raw files into structured knowledge.

    It recognizes labels such as:

    FACT:
    DEFINITION:
    CONCEPT:
    RULE:
    EXAMPLE:
    QUESTION:
    ANSWER:
    TERM:
    RELATION:
    PROJECT:
    NOTE:
    """

    LABELS = {
        "FACT": "fact",
        "DEFINITION": "definition",
        "CONCEPT": "concept",
        "RULE": "rule",
        "EXAMPLE": "example",
        "QUESTION": "question",
        "ANSWER": "answer",
        "TERM": "term",
        "RELATION": "relation",
        "PROJECT": "project",
        "NOTE": "note",
    }

    DEFINITION_WORDS = {
        "is",
        "means",
        "refers",
        "defined",
        "definition",
        "used",
    }

    @classmethod
    def infer_topic(
        cls,
        text: str,
        current_topic: str = "",
    ) -> str:

        if current_topic:
            return current_topic

        words = content_words(text)

        if not words:
            return ""

        # The first meaningful word is often the subject
        # in educational knowledge files.
        return words[0]

    @classmethod
    def extract(
        cls,
        text: str,
        source: str,
    ) -> List[Dict[str, Any]]:

        text = normalize_text(text)

        if not text:
            return []

        results = []

        current_topic = ""

        lines = text.split("\n")

        for line in lines:

            stripped = line.strip()

            if not stripped:
                continue

            # Markdown heading.
            heading = re.match(
                r"^#{1,6}\s+(.+)$",
                stripped,
            )

            if heading:
                current_topic = clean_answer(
                    heading.group(1)
                )

                continue

            # Structured label.
            match = re.match(
                r"^\s*(FACT|DEFINITION|CONCEPT|RULE|EXAMPLE|QUESTION|ANSWER|TERM|RELATION|PROJECT|NOTE)\s*:\s*(.+)$",
                stripped,
                flags=re.IGNORECASE,
            )

            if match:

                label = match.group(1).upper()
                content = match.group(2).strip()

                kind = cls.LABELS.get(
                    label,
                    "fact",
                )

                topic = cls.infer_topic(
                    content,
                    current_topic,
                )

                results.append({
                    "kind": kind,
                    "text": content,
                    "topic": topic,
                    "keywords": content_words(
                        content
                    ),
                })

                continue

            # Term = definition format.
            term_match = re.match(
                r"^([A-Za-z][A-Za-z0-9_+#.-]{1,50})\s*[:=]\s*(.{10,})$",
                stripped,
            )

            if term_match:

                term = term_match.group(1)
                definition = term_match.group(2)

                results.append({
                    "kind": "definition",
                    "text": (
                        f"{term} is {definition}"
                    ),
                    "topic": term,
                    "keywords": content_words(
                        f"{term} {definition}"
                    ),
                })

                continue

            # Normal sentence.
            sentences = split_sentences(
                stripped
            )

            for sentence in sentences:

                words = content_words(
                    sentence
                )

                if len(words) < 3:
                    continue

                lowered = sentence.lower()

                kind = "fact"

                if (
                    " is " in lowered
                    or " are " in lowered
                    or " means " in lowered
                    or " refers to " in lowered
                ):
                    kind = "definition"

                results.append({
                    "kind": kind,
                    "text": sentence,
                    "topic": cls.infer_topic(
                        sentence,
                        current_topic,
                    ),
                    "keywords": words,
                })

        return results


# ============================================================
# FILE LEARNER
# ============================================================

class FileLearner:

    def __init__(self, brain: Brain):
        self.brain = brain

    @staticmethod
    def hash_file(path: Path) -> str:

        hasher = hashlib.sha256()

        try:
            with path.open(
                "rb"
            ) as f:

                while True:
                    chunk = f.read(65536)

                    if not chunk:
                        break

                    hasher.update(chunk)

        except Exception:
            return ""

        return hasher.hexdigest()

    def read_file(
        self,
        path: Path,
    ) -> str:

        extension = path.suffix.lower()

        try:

            if extension == ".json":

                with path.open(
                    "r",
                    encoding="utf-8",
                ) as f:

                    data = json.load(f)

                return json.dumps(
                    data,
                    indent=2,
                    ensure_ascii=False,
                )

            if extension == ".csv":

                rows = []

                with path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as f:

                    reader = csv.reader(f)

                    for row in reader:
                        rows.append(
                            " | ".join(row)
                        )

                return "\n".join(rows)

            with path.open(
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as f:

                return f.read()

        except Exception:
            return ""

    def learn_file(
        self,
        path: Path,
        force: bool = False,
    ) -> int:

        if not path.exists():
            return 0

        if not path.is_file():
            return 0

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return 0

        path_string = str(path)

        file_hash = self.hash_file(path)

        if (
            not force
            and file_hash
            and self.brain.get_file_hash(
                path_string
            ) == file_hash
        ):
            return 0

        content = self.read_file(path)

        if not content:
            return 0

        extracted = KnowledgeExtractor.extract(
            content,
            path_string,
        )

        added = 0

        for item in extracted:

            created = self.brain.add_knowledge(
                kind=item["kind"],
                text=item["text"],
                source=path_string,
                topic=item["topic"],
                keywords=item["keywords"],
            )

            if created:
                added += 1

        if file_hash:
            self.brain.set_file_hash(
                path_string,
                file_hash,
            )

        return added

    def learn_folder(
        self,
        folder: Path,
        force: bool = False,
    ) -> int:

        if not folder.exists():
            return 0

        total = 0

        for path in folder.rglob("*"):

            if path.is_file():

                total += self.learn_file(
                    path,
                    force=force,
                )

        if total:
            self.brain.save()

        return total


# ============================================================
# SEARCH RESULT
# ============================================================

@dataclass
class SearchResult:
    item: KnowledgeItem
    score: float
    reasons: List[str]


# ============================================================
# SEARCH ENGINE
# ============================================================

class SearchEngine:
    """
    High-quality local retrieval.

    Important rule:

    A definition question strongly prefers:
        definition
        term
        concept

    over:
        unrelated fact
        example
        project

    This directly addresses the original failure.
    """

    KIND_WEIGHTS = {
        "definition": 1.00,
        "term": 0.92,
        "concept": 0.85,
        "fact": 0.68,
        "rule": 0.62,
        "relation": 0.58,
        "example": 0.48,
        "answer": 0.55,
        "note": 0.40,
        "project": 0.30,
        "question": 0.15,
    }

    TYPE_KIND_PREFERENCES = {

        "definition": {
            "definition": 1.0,
            "term": 0.95,
            "concept": 0.85,
            "fact": 0.55,
            "relation": 0.45,
            "example": 0.20,
            "project": 0.05,
        },

        "explanation": {
            "concept": 1.0,
            "definition": 0.90,
            "fact": 0.82,
            "relation": 0.78,
            "rule": 0.75,
            "example": 0.70,
        },

        "how": {
            "concept": 0.95,
            "rule": 0.95,
            "example": 0.90,
            "fact": 0.70,
            "definition": 0.60,
        },

        "why": {
            "concept": 0.95,
            "fact": 0.90,
            "relation": 0.90,
            "rule": 0.82,
            "definition": 0.55,
        },

        "example": {
            "example": 1.0,
            "concept": 0.60,
            "fact": 0.45,
            "definition": 0.35,
        },

        "comparison": {
            "relation": 1.0,
            "concept": 0.90,
            "fact": 0.85,
            "definition": 0.75,
        },

        "general": {
            "concept": 0.75,
            "definition": 0.75,
            "fact": 0.70,
            "relation": 0.65,
            "example": 0.55,
        },
    }

    def __init__(self, brain: Brain):
        self.brain = brain

    # --------------------------------------------------------
    # EXPANSION
    # --------------------------------------------------------

    def expand_terms(
        self,
        words: List[str],
    ) -> List[str]:

        expanded = list(words)

        for word in words:

            if word in RELATED_TERMS:

                expanded.extend(
                    RELATED_TERMS[word]
                )

            associated = self.brain.associations.get(
                word,
                {},
            )

            for related, count in associated.most_common(5):

                if count >= 2:
                    expanded.append(related)

        return unique_preserve_order(
            expanded
        )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    def score_item(
        self,
        item: KnowledgeItem,
        parsed: ParsedQuestion,
    ) -> SearchResult:

        score = 0.0
        reasons = []

        item_words = set(
            canonical_words(
                tokenize(item.text)
            )
        )

        item_keywords = set(
            canonical_words(
                item.keywords
            )
        )

        subject_words = set(
            parsed.subject_words
        )

        important_words = set(
            parsed.important_words
        )

        # ----------------------------------------------------
        # Subject matching
        # ----------------------------------------------------

        if subject_words:

            subject_matches = (
                subject_words
                & (item_words | item_keywords)
            )

            if subject_matches:

                ratio = (
                    len(subject_matches)
                    / len(subject_words)
                )

                score += 4.0 * ratio

                reasons.append(
                    f"subject {len(subject_matches)}/{len(subject_words)}"
                )

        # ----------------------------------------------------
        # Exact subject phrase
        # ----------------------------------------------------

        if parsed.subject:

            if phrase_contains(
                item.text,
                parsed.subject,
            ):

                score += 3.5

                reasons.append(
                    "exact subject phrase"
                )

        # ----------------------------------------------------
        # Important-word overlap
        # ----------------------------------------------------

        if important_words:

            matches = (
                important_words
                & (item_words | item_keywords)
            )

            ratio = (
                len(matches)
                / max(1, len(important_words))
            )

            score += 2.0 * ratio

            if matches:
                reasons.append(
                    f"word overlap {len(matches)}"
                )

        # ----------------------------------------------------
        # Phrase matching
        # ----------------------------------------------------

        for phrase in parsed.phrases:

            if len(phrase.split()) < 2:
                continue

            if phrase_contains(
                item.text,
                phrase,
            ):

                score += 1.5

                reasons.append(
                    "phrase match"
                )

                break

        # ----------------------------------------------------
        # Expanded semantic terms
        # ----------------------------------------------------

        expanded = set(
            self.expand_terms(
                parsed.subject_words
                + parsed.important_words
            )
        )

        semantic_matches = (
            expanded
            & (item_words | item_keywords)
        )

        if semantic_matches:

            score += min(
                2.0,
                len(semantic_matches) * 0.25,
            )

            reasons.append(
                "related concepts"
            )

        # ----------------------------------------------------
        # Topic match
        # ----------------------------------------------------

        if item.topic:

            topic = canonical_word(
                item.topic
            )

            if topic in subject_words:

                score += 3.0

                reasons.append(
                    "topic match"
                )

        # ----------------------------------------------------
        # Question type
        # ----------------------------------------------------

        preferences = self.TYPE_KIND_PREFERENCES.get(
            parsed.question_type,
            self.TYPE_KIND_PREFERENCES["general"],
        )

        kind_bonus = preferences.get(
            item.kind,
            0.25,
        )

        score += 3.0 * kind_bonus

        # ----------------------------------------------------
        # SPECIAL DEFINITION LOGIC
        # ----------------------------------------------------

        if parsed.question_type == "definition":

            if item.kind == "definition":

                score += 6.0

                reasons.append(
                    "definition requested"
                )

            elif item.kind == "term":

                score += 5.0

                reasons.append(
                    "term definition"
                )

            elif item.kind == "concept":

                score += 3.5

                reasons.append(
                    "concept definition"
                )

            elif item.kind == "example":

                score -= 2.0

            elif item.kind == "project":

                score -= 2.5

        # ----------------------------------------------------
        # Example logic
        # ----------------------------------------------------

        if parsed.question_type == "example":

            if item.kind == "example":
                score += 5.0

        # ----------------------------------------------------
        # How logic
        # ----------------------------------------------------

        if parsed.question_type == "how":

            if item.kind in {
                "example",
                "rule",
                "concept",
            }:
                score += 2.0

        # ----------------------------------------------------
        # Comparison logic
        # ----------------------------------------------------

        if parsed.question_type == "comparison":

            if item.kind == "relation":
                score += 5.0

        # ----------------------------------------------------
        # Feedback
        # ----------------------------------------------------

        feedback = item.feedback_score()

        score += feedback * 1.5

        # ----------------------------------------------------
        # Usage quality
        # ----------------------------------------------------

        if item.times_used > 0:

            usage_bonus = min(
                0.5,
                math.log1p(
                    item.times_used
                ) / 10,
            )

            score += usage_bonus

        # ----------------------------------------------------
        # General quality
        # ----------------------------------------------------

        word_count = len(
            content_words(item.text)
        )

        if 5 <= word_count <= 80:
            score += 0.35

        # Penalize extremely tiny fragments.
        if word_count < 4:
            score -= 1.0

        return SearchResult(
            item=item,
            score=score,
            reasons=reasons,
        )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    def search(
        self,
        question: str,
        limit: int = 12,
    ) -> List[SearchResult]:

        parsed = QuestionAnalyzer.analyze(
            question
        )

        results = []

        for item in self.brain.knowledge:

            result = self.score_item(
                item,
                parsed,
            )

            if result.score > 0:
                results.append(result)

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        # Deduplicate near-identical answers.
        final = []

        for result in results:

            duplicate = False

            for existing in final:

                if text_similarity(
                    result.item.text,
                    existing.item.text,
                ) > 0.85:

                    duplicate = True
                    break

            if not duplicate:
                final.append(result)

            if len(final) >= limit:
                break

        return final


# ============================================================
# ANSWER BUILDER
# ============================================================

class AnswerBuilder:

    def __init__(
        self,
        brain: Brain,
        search_engine: SearchEngine,
    ):

        self.brain = brain
        self.search_engine = search_engine

    # --------------------------------------------------------
    # FIND SUBJECT
    # --------------------------------------------------------

    def find_subject_item(
        self,
        results: List[SearchResult],
        subject: str,
    ) -> Optional[SearchResult]:

        subject_words = set(
            content_words(subject)
        )

        # First: exact topic.
        for result in results:

            topic = canonical_word(
                result.item.topic
            )

            if (
                topic
                and topic in subject_words
            ):
                return result

        # Second: definition/term.
        for result in results:

            if result.item.kind in {
                "definition",
                "term",
                "concept",
            }:

                item_words = set(
                    content_words(
                        result.item.text
                    )
                )

                if subject_words & item_words:
                    return result

        return None

    # --------------------------------------------------------
    # DEFINITION
    # --------------------------------------------------------

    def build_definition(
        self,
        parsed: ParsedQuestion,
        results: List[SearchResult],
    ) -> Tuple[str, Optional[int], float]:

        subject = parsed.subject.strip()

        best = self.find_subject_item(
            results,
            subject,
        )

        if best:

            text = clean_answer(
                best.item.text
            )

            # If the source contains a good definition,
            # return it directly.
            if text:

                best.item.times_used += 1

                confidence = self.confidence(
                    best.score,
                    results,
                )

                return (
                    text,
                    best.item.id,
                    confidence,
                )

        # Fall back to highest-ranked result.
        if results:

            result = results[0]

            result.item.times_used += 1

            return (
                clean_answer(
                    result.item.text
                ),
                result.item.id,
                self.confidence(
                    result.score,
                    results,
                ),
            )

        return (
            "I don't know that yet.",
            None,
            0.0,
        )

    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    def build_general(
        self,
        parsed: ParsedQuestion,
        results: List[SearchResult],
    ) -> Tuple[str, Optional[int], float]:

        if not results:

            return (
                "I don't know that yet.",
                None,
                0.0,
            )

        best = results[0]

        best.item.times_used += 1

        answer = clean_answer(
            best.item.text
        )

        return (
            answer,
            best.item.id,
            self.confidence(
                best.score,
                results,
            ),
        )

    # --------------------------------------------------------
    # EXAMPLE
    # --------------------------------------------------------

    def build_example(
        self,
        parsed: ParsedQuestion,
        results: List[SearchResult],
    ) -> Tuple[str, Optional[int], float]:

        example_results = [
            result
            for result in results
            if result.item.kind == "example"
        ]

        selected = (
            example_results
            or results
        )

        if not selected:

            return (
                "I don't know an example for that yet.",
                None,
                0.0,
            )

        result = selected[0]

        result.item.times_used += 1

        return (
            clean_answer(
                result.item.text
            ),
            result.item.id,
            self.confidence(
                result.score,
                results,
            ),
        )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    def build_comparison(
        self,
        parsed: ParsedQuestion,
        results: List[SearchResult],
    ) -> Tuple[str, Optional[int], float]:

        if not results:

            return (
                "I don't have enough learned information to compare those yet.",
                None,
                0.0,
            )

        selected = [
            result
            for result in results
            if result.item.kind
            in {
                "relation",
                "concept",
                "fact",
                "definition",
            }
        ]

        if not selected:
            selected = results

        pieces = []

        ids = []

        for result in selected[:3]:

            text = clean_answer(
                result.item.text
            )

            if text not in pieces:
                pieces.append(text)

            ids.append(
                result.item.id
            )

        answer = " ".join(pieces)

        for result in selected[:3]:
            result.item.times_used += 1

        return (
            answer,
            ids[0] if ids else None,
            self.confidence(
                selected[0].score,
                results,
            ),
        )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    def build_explanation(
        self,
        parsed: ParsedQuestion,
        results: List[SearchResult],
    ) -> Tuple[str, Optional[int], float]:

        if not results:

            return (
                "I don't know enough about that yet.",
                None,
                0.0,
            )

        selected = []

        seen = set()

        for result in results:

            text = clean_answer(
                result.item.text
            )

            if text in seen:
                continue

            seen.add(text)
            selected.append(result)

            if len(selected) >= 3:
                break

        answer_parts = [
            clean_answer(
                result.item.text
            )
            for result in selected
        ]

        for result in selected:
            result.item.times_used += 1

        answer = " ".join(answer_parts)

        return (
            answer,
            selected[0].item.id,
            self.confidence(
                selected[0].score,
                results,
            ),
        )

    # --------------------------------------------------------
    # MAIN BUILD
    # --------------------------------------------------------

    def build(
        self,
        question: str,
    ) -> Tuple[
        str,
        Optional[int],
        float,
        ParsedQuestion,
        List[SearchResult],
    ]:

        parsed = QuestionAnalyzer.analyze(
            question
        )

        results = self.search_engine.search(
            question,
            limit=12,
        )

        if parsed.question_type == "definition":

            answer, item_id, confidence = (
                self.build_definition(
                    parsed,
                    results,
                )
            )

        elif parsed.question_type == "example":

            answer, item_id, confidence = (
                self.build_example(
                    parsed,
                    results,
                )
            )

        elif parsed.question_type == "comparison":

            answer, item_id, confidence = (
                self.build_comparison(
                    parsed,
                    results,
                )
            )

        elif parsed.question_type in {
            "explanation",
            "how",
            "why",
        }:

            answer, item_id, confidence = (
                self.build_explanation(
                    parsed,
                    results,
                )
            )

        else:

            answer, item_id, confidence = (
                self.build_general(
                    parsed,
                    results,
                )
            )

        return (
            answer,
            item_id,
            confidence,
            parsed,
            results,
        )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    @staticmethod
    def confidence(
        score: float,
        results: List[SearchResult],
    ) -> float:

        if not results:
            return 0.0

        # Convert retrieval score to a readable
        # confidence-like number.
        value = 1.0 - math.exp(
            -max(0.0, score) / 8.0
        )

        if len(results) >= 2:

            gap = (
                results[0].score
                - results[1].score
            )

            value += min(
                0.12,
                max(0.0, gap) / 20,
            )

        return max(
            0.0,
            min(0.99, value),
        )


# ============================================================
# MEMORY SEARCH
# ============================================================

class MemorySearch:

    def __init__(self, brain: Brain):
        self.brain = brain

    def search(
        self,
        question: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:

        query_words = set(
            content_words(question)
        )

        scored = []

        for memory in self.brain.memories:

            text = memory.get(
                "text",
                "",
            )

            words = set(
                content_words(text)
            )

            overlap = len(
                query_words & words
            )

            if overlap:

                score = (
                    overlap
                    / max(
                        1,
                        len(query_words),
                    )
                )

                scored.append(
                    (
                        score,
                        memory,
                    )
                )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            item
            for _, item in scored[:limit]
        ]


# ============================================================
# SELF LEARNING ENGINE
# ============================================================

class SelfLearningEngine:

    def __init__(
        self,
        brain: Brain,
        learner: FileLearner,
    ):

        self.brain = brain
        self.learner = learner

    def learn_local(self) -> int:

        KNOWLEDGE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        return self.learner.learn_folder(
            KNOWLEDGE_DIR,
            force=False,
        )

    def learn_from_memories(self) -> int:

        added = 0

        for memory in self.brain.memories[-100:]:

            text = memory.get(
                "text",
                "",
            )

            if len(text) < 20:
                continue

            item = self.brain.add_knowledge(
                kind="note",
                text=text,
                source="conversation_memory",
                topic="memory",
                keywords=content_words(text),
            )

            if item:
                added += 1

        return added

    def cycle(self) -> int:

        total = 0

        total += self.learn_local()

        total += self.learn_from_memories()

        if total:
            self.brain.save()

        return total


# ============================================================
# STATISTICS
# ============================================================

class Statistics:

    @staticmethod
    def display(brain: Brain):

        stats = brain.stats()

        print()
        print("=" * 60)
        print("AI STATISTICS")
        print("=" * 60)

        print(
            f"Knowledge items: {stats['knowledge']}"
        )

        print(
            f"Known words:     {stats['words']}"
        )

        print(
            f"Topics:          {stats['topics']}"
        )

        print(
            f"Memories:        {stats['memories']}"
        )

        print(
            f"Conversations:   {stats['conversations']}"
        )

        print(
            f"Questions:       {stats['questions']}"
        )

        print(
            f"Sources:         {stats['sources']}"
        )

        print("=" * 60)
        print()


# ============================================================
# MAIN AI
# ============================================================

class AutonomousAI:

    def __init__(self):

        KNOWLEDGE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.config = ConfigManager(
            CONFIG_FILE
        )

        self.brain = Brain(
            BRAIN_FILE
        )

        self.learner = FileLearner(
            self.brain
        )

        self.search_engine = SearchEngine(
            self.brain
        )

        self.answer_builder = AnswerBuilder(
            self.brain,
            self.search_engine,
        )

        self.memory_search = MemorySearch(
            self.brain
        )

        self.self_learning = SelfLearningEngine(
            self.brain,
            self.learner,
        )

        self.running = True

        self.background_thread = None

    # --------------------------------------------------------
    # STARTUP
    # --------------------------------------------------------

    def startup(self):

        print()
        print("=" * 64)
        print("AUTONOMOUS LEARNING AI")
        print(f"Version {VERSION}")
        print("=" * 64)

        print(
            f"Knowledge items loaded: "
            f"{len(self.brain.knowledge)}"
        )

        print(
            f"Known words: "
            f"{len(self.brain.word_counts)}"
        )

        if self.config.get(
            "auto_learning",
            True,
        ):

            self.start_background_learning()

        # Immediately scan for new knowledge.
        learned = self.self_learning.learn_local()

        if learned:

            print(
                f"Learned {learned} new knowledge items."
            )

        print()
        print(
            "Type /help for commands."
        )

        print()

    # --------------------------------------------------------
    # BACKGROUND LEARNING
    # --------------------------------------------------------

    def start_background_learning(self):

        if (
            self.background_thread
            and self.background_thread.is_alive()
        ):
            return

        self.background_thread = threading.Thread(
            target=self.background_learning_loop,
            daemon=True,
        )

        self.background_thread.start()

    def background_learning_loop(self):

        while self.running:

            interval = self.config.get(
                "learning_interval_seconds",
                60,
            )

            try:
                interval = max(
                    10,
                    int(interval),
                )
            except Exception:
                interval = 60

            for _ in range(interval):

                if not self.running:
                    return

                time.sleep(1)

            try:
                if self.config.get(
                    "auto_learning",
                    True,
                ):
                    self.self_learning.cycle()

            except Exception:
                pass

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    def answer(
        self,
        question: str,
    ) -> bool:

        question = question.strip()

        if not question:
            return True

        (
            answer,
            item_id,
            confidence,
            parsed,
            results,
        ) = self.answer_builder.build(
            question
        )

        self.brain.record_question(
            question,
            parsed.question_type,
            parsed.subject,
        )

        if self.config.get(
            "remember_conversations",
            True,
        ):

            self.brain.add_conversation(
                question,
                answer,
            )

            self.brain.add_memory(
                f"Question: {question} Answer: {answer}",
                category="conversation",
            )

        self.brain.save()

        # ----------------------------------------------------
        # Confidence label
        # ----------------------------------------------------

        if confidence >= 0.72:
            label = "high"

        elif confidence >= 0.42:
            label = "medium"

        else:
            label = "low"

        print()

        print(
            f"AI [{label} {confidence:.2f}]:"
        )

        print(
            answer
        )

        # Show interpretation when useful.
        if parsed.question_type != "general":

            print(
                f"\n[understood: "
                f"{parsed.question_type}"
                f" | subject: "
                f"{parsed.subject or 'unknown'}]"
            )

        print()

        feedback = input(
            "Was that useful? [y/n/enter]: "
        ).strip().lower()

        if (
            feedback == "y"
            and item_id is not None
        ):

            self.brain.vote(
                item_id,
                True,
            )

            print(
                "Learned from your positive feedback."
            )

        elif (
            feedback == "n"
            and item_id is not None
        ):

            self.brain.vote(
                item_id,
                False,
            )

            print(
                "Recorded negative feedback."
            )

        return True

    # --------------------------------------------------------
    # HELP
    # --------------------------------------------------------

    def help(self):

        print()
        print("=" * 64)
        print("COMMANDS")
        print("=" * 64)

        commands = [
            (
                "/help",
                "Show this help menu",
            ),
            (
                "/stats",
                "Show learning statistics",
            ),
            (
                "/learn",
                "Learn the knowledge folder",
            ),
            (
                "/learn <path>",
                "Learn a specific file or folder",
            ),
            (
                "/forcelearn",
                "Force relearning of knowledge files",
            ),
            (
                "/selflearn",
                "Run a complete learning cycle",
            ),
            (
                "/search <text>",
                "Search learned knowledge",
            ),
            (
                "/topics",
                "Show learned topics",
            ),
            (
                "/words",
                "Show common learned words",
            ),
            (
                "/facts",
                "Show recent knowledge",
            ),
            (
                "/memory",
                "Show conversation memories",
            ),
            (
                "/config",
                "Show configuration",
            ),
            (
                "/autolearn on",
                "Enable automatic learning",
            ),
            (
                "/autolearn off",
                "Disable automatic learning",
            ),
            (
                "/clear",
                "Clear the terminal",
            ),
            (
                "/quit",
                "Exit",
            ),
        ]

        for command, description in commands:

            print(
                f"{command:<22} {description}"
            )

        print("=" * 64)
        print()

    # --------------------------------------------------------
    # COMMAND: SEARCH
    # --------------------------------------------------------

    def command_search(
        self,
        query: str,
    ):

        if not query:

            print(
                "Usage: /search <text>"
            )

            return

        results = self.search_engine.search(
            query,
            limit=10,
        )

        print()

        if not results:

            print(
                "No matching knowledge found."
            )

            return

        print(
            f"Search results for: {query}"
        )

        print("-" * 64)

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"{index}. "
                f"[{result.item.kind}] "
                f"{result.item.text}"
            )

            print(
                f"   score={result.score:.2f} "
                f"topic={result.item.topic}"
            )

        print()

    # --------------------------------------------------------
    # COMMAND: TOPICS
    # --------------------------------------------------------

    def command_topics(self):

        print()

        if not self.brain.topic_counts:

            print(
                "No topics learned yet."
            )

            return

        print("TOP LEARNED TOPICS")
        print("-" * 40)

        for topic, count in (
            self.brain.topic_counts
            .most_common(30)
        ):

            print(
                f"{topic:<25} {count}"
            )

        print()

    # --------------------------------------------------------
    # COMMAND: WORDS
    # --------------------------------------------------------

    def command_words(self):

        print()

        if not self.brain.word_counts:

            print(
                "No words learned yet."
            )

            return

        print("MOST COMMON LEARNED WORDS")
        print("-" * 40)

        for word, count in (
            self.brain.word_counts
            .most_common(50)
        ):

            print(
                f"{word:<25} {count}"
            )

        print()

    # --------------------------------------------------------
    # COMMAND: FACTS
    # --------------------------------------------------------

    def command_facts(self):

        print()

        if not self.brain.knowledge:

            print(
                "No knowledge learned yet."
            )

            return

        print("RECENT KNOWLEDGE")
        print("-" * 64)

        for item in self.brain.knowledge[-30:]:

            print(
                f"[{item.kind}] "
                f"{item.text}"
            )

        print()

    # --------------------------------------------------------
    # COMMAND: MEMORY
    # --------------------------------------------------------

    def command_memory(self):

        print()

        if not self.brain.memories:

            print(
                "No memories yet."
            )

            return

        print("RECENT MEMORIES")
        print("-" * 64)

        for memory in (
            self.brain.memories[-20:]
        ):

            print(
                f"[{memory.get('category', 'general')}] "
                f"{memory.get('text', '')}"
            )

        print()

    # --------------------------------------------------------
    # COMMAND: CONFIG
    # --------------------------------------------------------

    def command_config(self):

        print()
        print("CONFIGURATION")
        print("-" * 40)

        for key, value in (
            self.config.data.items()
        ):

            print(
                f"{key:<35} {value}"
            )

        print()

    # --------------------------------------------------------
    # COMMAND: LEARN
    # --------------------------------------------------------

    def command_learn(
        self,
        argument: str,
    ):

        argument = argument.strip()

        if not argument:

            count = self.self_learning.learn_local()

            print(
                f"Learned {count} new knowledge items."
            )

            return

        path = Path(argument)

        if not path.exists():

            print(
                f"Path does not exist: {path}"
            )

            return

        if path.is_file():

            count = self.learner.learn_file(
                path,
                force=False,
            )

        else:

            count = self.learner.learn_folder(
                path,
                force=False,
            )

        self.brain.save()

        print(
            f"Learned {count} new knowledge items."
        )

    # --------------------------------------------------------
    # FORCE LEARN
    # --------------------------------------------------------

    def command_forcelearn(self):

        count = self.learner.learn_folder(
            KNOWLEDGE_DIR,
            force=True,
        )

        self.brain.save()

        print(
            f"Force learned {count} knowledge items."
        )

    # --------------------------------------------------------
    # SELF LEARN
    # --------------------------------------------------------

    def command_selflearn(self):

        print(
            "Running self-learning cycle..."
        )

        count = self.self_learning.cycle()

        print(
            f"Self-learning added {count} items."
        )

    # --------------------------------------------------------
    # AUTOLEARN
    # --------------------------------------------------------

    def command_autolearn(
        self,
        value: str,
    ):

        value = value.lower().strip()

        if value == "on":

            self.config.set(
                "auto_learning",
                True,
            )

            self.start_background_learning()

            print(
                "Automatic learning enabled."
            )

        elif value == "off":

            self.config.set(
                "auto_learning",
                False,
            )

            print(
                "Automatic learning disabled."
            )

        else:

            print(
                "Usage: /autolearn on"
            )

            print(
                "       /autolearn off"
            )

    # --------------------------------------------------------
    # COMMAND HANDLER
    # --------------------------------------------------------

    def command(
        self,
        text: str,
    ) -> bool:

        parts = text.split(
            maxsplit=1
        )

        command = parts[0].lower()

        argument = (
            parts[1]
            if len(parts) > 1
            else ""
        )

        if command in {
            "/quit",
            "/exit",
            "/q",
        }:

            self.running = False
            self.brain.save()

            print(
                "Goodbye."
            )

            return False

        if command in {
            "/help",
            "/h",
        }:

            self.help()
            return True

        if command == "/stats":

            Statistics.display(
                self.brain
            )

            return True

        if command == "/learn":

            self.command_learn(
                argument
            )

            return True

        if command == "/forcelearn":

            self.command_forcelearn()

            return True

        if command == "/selflearn":

            self.command_selflearn()

            return True

        if command == "/search":

            self.command_search(
                argument
            )

            return True

        if command == "/topics":

            self.command_topics()

            return True

        if command == "/words":

            self.command_words()

            return True

        if command == "/facts":

            self.command_facts()

            return True

        if command == "/memory":

            self.command_memory()

            return True

        if command == "/config":

            self.command_config()

            return True

        if command == "/autolearn":

            self.command_autolearn(
                argument
            )

            return True

        if command == "/clear":

            os.system("clear")

            return True

        print(
            "Unknown command. Type /help."
        )

        return True

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    def run(self):

        self.startup()

        while self.running:

            try:

                user_input = input(
                    "You: "
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError,
            ):

                print()

                self.running = False
                self.brain.save()

                break

            if not user_input:
                continue

            if user_input.startswith("/"):

                if not self.command(
                    user_input
                ):
                    break

                continue

            self.answer(
                user_input
            )


# ============================================================
# SAFE STARTUP
# ============================================================

def main():

    try:

        ai = AutonomousAI()

        ai.run()

    except KeyboardInterrupt:

        print(
            "\nStopped."
        )

    except Exception as exc:

        print()
        print(
            "The AI encountered an error:"
        )

        print(
            repr(exc)
        )

        print()
        print(
            "Your brain file was not intentionally deleted."
        )


if __name__ == "__main__":
    main()