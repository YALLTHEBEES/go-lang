"""
AUTONOMOUS LEARNING AI
======================

A single-file, Python-standard-library learning system.

What it does:
- Learns from conversations.
- Learns from local .txt/.md/.csv/.json files.
- Can learn from URLs using Python's standard urllib library.
- Can crawl additional links from a page when explicitly enabled.
- Builds a vocabulary, n-gram language model, concepts, associations,
  facts, examples, and question/answer memories.
- Tracks confidence and source counts.
- Revisits knowledge and consolidates related information.
- Uses feedback to reinforce or weaken responses.
- Saves everything to a JSON brain file.
- Runs background "self-learning" cycles from its configured knowledge
  sources while the program is open.
- Does NOT call ChatGPT, OpenAI, or another external AI API.
- Uses only Python's standard library.

Important:
This is a learning engine, not a human-level neural network.
"Learning on its own" here means that after you give it trusted local
files or URLs, it can continuously extract patterns, vocabulary,
relationships, and knowledge from those sources without you manually
teaching every individual response.

Python version: 3.10+
"""

# ============================================================
# IMPORTS
# ============================================================

import csv
import json
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

BRAIN_FILE = "autonomous_brain.json"
CONFIG_FILE = "autonomous_config.json"
KNOWLEDGE_DIR = "knowledge"

VERSION = "5.0"

MAX_INPUTS = 20000
MAX_RESPONSES = 30
MAX_FACTS = 30000
MAX_WORDS = 50000
MAX_CONCEPTS = 20000
MAX_ASSOCIATIONS = 50000
MAX_SOURCE_TEXT = 2_000_000

MIN_MATCH_SCORE = 0.34
GOOD_MATCH_SCORE = 0.58
VERY_GOOD_MATCH_SCORE = 0.78

MAX_CANDIDATES = 15
MAX_SENTENCE_LENGTH = 600

AUTO_LEARN_INTERVAL = 60
AUTO_SAVE_INTERVAL = 30

REQUEST_TIMEOUT = 12
MAX_CRAWL_PAGES = 25
MAX_CRAWL_DEPTH = 1

USER_AGENT = "AutonomousLearningAI/5.0"

STOP_WORDS = {
    "a", "an", "the", "is", "are", "am", "to", "of", "in", "on",
    "at", "for", "and", "or", "but", "i", "you", "he", "she", "it",
    "we", "they", "me", "my", "your", "our", "their", "this", "that",
    "these", "those", "with", "do", "does", "did", "be", "been",
    "being", "was", "were", "as", "from", "by", "about", "what",
    "how", "why", "when", "where", "who", "which", "can", "could",
    "would", "should", "will", "shall", "may", "might", "must",
    "have", "has", "had", "having", "not", "no", "yes", "if",
    "then", "than", "so", "very", "just", "into", "over", "under",
    "again", "also", "there", "here", "all", "any", "some", "more",
    "most", "other", "such", "only", "too", "up", "down", "out",
    "off", "once", "because", "while", "during", "before", "after",
    "each", "both", "few", "many", "much", "own", "same", "new"
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z0-9_']+")
URL_RE = re.compile(r"https?://[^\s<>\"]+")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def now() -> float:
    return time.time()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def clean_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_text(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(clean_text(text))


def useful_words(text: str) -> List[str]:
    return [
        word for word in tokenize(text)
        if word not in STOP_WORDS and len(word) > 1
    ]


def word_set(text: str) -> Set[str]:
    return set(useful_words(text))


def word_counter(text: str) -> Counter:
    return Counter(useful_words(text))


def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    pieces = SENTENCE_SPLIT.split(text)
    result = []
    for piece in pieces:
        piece = piece.strip()
        if piece:
            result.append(piece[:MAX_SENTENCE_LENGTH])
    return result


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value[:100] or "source"


def timestamp_string(value: Optional[float] = None) -> str:
    value = value or now()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))


def weighted_average(old: float, new: float, weight: float) -> float:
    weight = clamp(weight)
    return old * (1.0 - weight) + new * weight


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ============================================================
# SIMILARITY
# ============================================================

def character_similarity(a: str, b: str) -> float:
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


def word_overlap(a: str, b: str) -> float:
    wa = word_set(a)
    wb = word_set(b)

    if not wa or not wb:
        return 0.0

    return len(wa & wb) / len(wa | wb)


def containment_similarity(a: str, b: str) -> float:
    wa = word_set(a)
    wb = word_set(b)

    if not wa or not wb:
        return 0.0

    smaller = min(len(wa), len(wb))
    if smaller == 0:
        return 0.0

    return len(wa & wb) / smaller


def cosine_similarity(a: str, b: str) -> float:
    ca = word_counter(a)
    cb = word_counter(b)

    if not ca or not cb:
        return 0.0

    vocabulary = set(ca) | set(cb)

    dot = 0.0
    ma = 0.0
    mb = 0.0

    for word in vocabulary:
        x = ca.get(word, 0)
        y = cb.get(word, 0)
        dot += x * y
        ma += x * x
        mb += y * y

    if ma == 0 or mb == 0:
        return 0.0

    return dot / (math.sqrt(ma) * math.sqrt(mb))


def ngram_similarity(a: str, b: str, n: int = 2) -> float:
    def grams(text: str) -> Set[str]:
        words = tokenize(text)
        if len(words) < n:
            return set(words)
        return {
            " ".join(words[i:i+n])
            for i in range(len(words) - n + 1)
        }

    ga = grams(a)
    gb = grams(b)

    if not ga or not gb:
        return 0.0

    return len(ga & gb) / len(ga | gb)


def combined_similarity(a: str, b: str) -> float:
    scores = [
        character_similarity(a, b),
        word_overlap(a, b),
        cosine_similarity(a, b),
        containment_similarity(a, b),
        ngram_similarity(a, b),
    ]

    score = (
        scores[0] * 0.18 +
        scores[1] * 0.28 +
        scores[2] * 0.24 +
        scores[3] * 0.18 +
        scores[4] * 0.12
    )

    return clamp(score)


# ============================================================
# HTML EXTRACTION
# ============================================================

class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self.links: List[str] = []
        self.skip_depth = 0
        self.skip_tags = {
            "script", "style", "noscript", "svg", "canvas",
            "nav", "footer", "header", "form"
        }

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        tag = tag.lower()

        if tag in self.skip_tags:
            self.skip_depth += 1

        if tag == "a":
            for key, value in attrs:
                if key.lower() == "href" and value:
                    self.links.append(value)

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self.skip_tags and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str):
        if self.skip_depth == 0:
            data = re.sub(r"\s+", " ", data).strip()
            if data:
                self.parts.append(data)

    def text(self) -> str:
        return "\n".join(self.parts)


# ============================================================
# SOURCE RECORD
# ============================================================

@dataclass
class SourceRecord:
    source_id: str
    source_type: str
    location: str
    title: str
    added: float
    last_read: float
    times_read: int
    words_learned: int
    active: bool = True


# ============================================================
# BRAIN
# ============================================================

class Brain:
    """
    Persistent storage and learning core.
    """

    def __init__(self, path: str = BRAIN_FILE):
        self.path = path
        self.lock = threading.RLock()
        self.data = self.default_data()
        self.load()

    def default_data(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "created": now(),
            "updated": now(),

            "inputs": {},
            "words": {},
            "ngrams": {
                "1": {},
                "2": {},
                "3": {}
            },
            "facts": {},
            "concepts": {},
            "associations": {},
            "sources": {},
            "sentences": {},
            "documents": {},

            "stats": {
                "messages": 0,
                "learn_events": 0,
                "facts_learned": 0,
                "sentences_learned": 0,
                "words_learned": 0,
                "sources_read": 0,
                "responses": 0,
                "correct": 0,
                "incorrect": 0,
                "self_cycles": 0,
            },

            "settings": {
                "auto_learn": True,
                "auto_save": True,
                "allow_web": False,
                "allow_crawl": False,
                "crawl_depth": MAX_CRAWL_DEPTH,
                "crawl_pages": MAX_CRAWL_PAGES,
                "learn_interval": AUTO_LEARN_INTERVAL,
                "save_interval": AUTO_SAVE_INTERVAL,
            },

            "queue": [],
            "recent_topics": [],
        }

    def merge_defaults(self, loaded: Dict[str, Any]) -> None:
        defaults = self.default_data()

        for key, value in defaults.items():
            if key not in loaded:
                loaded[key] = value

        for section in (
            "stats",
            "settings",
            "ngrams",
        ):
            if section not in loaded:
                loaded[section] = defaults[section]

        for key, value in defaults["stats"].items():
            loaded["stats"].setdefault(key, value)

        for key, value in defaults["settings"].items():
            loaded["settings"].setdefault(key, value)

        for key in ("1", "2", "3"):
            loaded["ngrams"].setdefault(key, {})

    def load(self) -> None:
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as file:
                loaded = json.load(file)

            if isinstance(loaded, dict):
                self.merge_defaults(loaded)
                self.data = loaded

        except (OSError, json.JSONDecodeError) as error:
            print(f"[brain] Could not load brain: {error}")

    def save(self) -> None:
        with self.lock:
            self.data["updated"] = now()
            temp = self.path + ".tmp"

            try:
                with open(temp, "w", encoding="utf-8") as file:
                    json.dump(
                        self.data,
                        file,
                        indent=2,
                        ensure_ascii=False
                    )

                os.replace(temp, self.path)

            except OSError as error:
                print(f"[brain] Save failed: {error}")

    # --------------------------------------------------------
    # Generic counters
    # --------------------------------------------------------

    def increment(self, stat: str, amount: int = 1) -> None:
        self.data["stats"][stat] = (
            self.data["stats"].get(stat, 0) + amount
        )

    # --------------------------------------------------------
    # Word learning
    # --------------------------------------------------------

    def learn_words(self, text: str, source_id: str = "") -> List[str]:
        words = tokenize(text)
        learned = []

        with self.lock:
            for word in words:
                if len(word) < 2:
                    continue

                record = self.data["words"].setdefault(
                    word,
                    {
                        "count": 0,
                        "first_seen": now(),
                        "last_seen": now(),
                        "sources": {},
                    }
                )

                record["count"] += 1
                record["last_seen"] = now()

                if source_id:
                    record["sources"][source_id] = (
                        record["sources"].get(source_id, 0) + 1
                    )

                learned.append(word)

            self.increment("words_learned", len(set(learned)))

            self.prune_words()

        return learned

    def prune_words(self) -> None:
        if len(self.data["words"]) <= MAX_WORDS:
            return

        items = sorted(
            self.data["words"].items(),
            key=lambda item: item[1].get("count", 0),
            reverse=True
        )

        self.data["words"] = dict(items[:MAX_WORDS])

    # --------------------------------------------------------
    # N-gram learning
    # --------------------------------------------------------

    def learn_ngrams(self, text: str) -> None:
        words = tokenize(text)

        if not words:
            return

        with self.lock:
            for word in words:
                bucket = self.data["ngrams"]["1"]
                bucket[word] = bucket.get(word, 0) + 1

            for n in (2, 3):
                if len(words) < n:
                    continue

                bucket = self.data["ngrams"][str(n)]

                for index in range(len(words) - n + 1):
                    gram = " ".join(words[index:index+n])
                    bucket[gram] = bucket.get(gram, 0) + 1

    # --------------------------------------------------------
    # Input/response memory
    # --------------------------------------------------------

    def remember_input(
        self,
        user_input: str,
        response: str,
        source_id: str = "conversation"
    ) -> None:

        key = normalize_text(user_input)
        response = response.strip()

        if not key or not response:
            return

        with self.lock:
            record = self.data["inputs"].setdefault(
                key,
                {
                    "responses": {},
                    "seen": 0,
                    "created": now(),
                    "last_seen": now(),
                    "confidence": 0.5,
                    "sources": {},
                }
            )

            record["seen"] += 1
            record["last_seen"] = now()

            response_record = record["responses"].setdefault(
                response,
                {
                    "count": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "score": 0.5,
                }
            )

            response_record["count"] += 1

            if source_id:
                record["sources"][source_id] = (
                    record["sources"].get(source_id, 0) + 1
                )

            self.increment("learn_events")
            self.prune_inputs()

    def prune_inputs(self) -> None:
        if len(self.data["inputs"]) <= MAX_INPUTS:
            return

        items = sorted(
            self.data["inputs"].items(),
            key=lambda item: (
                item[1].get("seen", 0),
                item[1].get("last_seen", 0)
            ),
            reverse=True
        )

        self.data["inputs"] = dict(items[:MAX_INPUTS])

    # --------------------------------------------------------
    # Sentences
    # --------------------------------------------------------

    def learn_sentence(
        self,
        sentence: str,
        source_id: str = ""
    ) -> None:

        sentence = re.sub(r"\s+", " ", sentence).strip()

        if len(sentence) < 8:
            return

        key = normalize_text(sentence)

        with self.lock:
            record = self.data["sentences"].setdefault(
                key,
                {
                    "text": sentence,
                    "count": 0,
                    "sources": {},
                    "first_seen": now(),
                    "last_seen": now(),
                    "confidence": 0.4,
                }
            )

            record["count"] += 1
            record["last_seen"] = now()

            if source_id:
                record["sources"][source_id] = (
                    record["sources"].get(source_id, 0) + 1
                )

            self.increment("sentences_learned")

            if len(self.data["sentences"]) > MAX_FACTS:
                self.prune_sentences()

    def prune_sentences(self) -> None:
        items = sorted(
            self.data["sentences"].items(),
            key=lambda item: (
                item[1].get("count", 0),
                len(item[1].get("sources", {}))
            ),
            reverse=True
        )

        self.data["sentences"] = dict(items[:MAX_FACTS])

    # --------------------------------------------------------
    # Facts
    # --------------------------------------------------------

    def add_fact(
        self,
        subject: str,
        relation: str,
        value: str,
        source_id: str,
        confidence: float = 0.5
    ) -> str:

        subject = normalize_text(subject)
        relation = normalize_text(relation)
        value = value.strip()

        if not subject or not relation or not value:
            return ""

        fact_id = f"{subject}|{relation}|{normalize_text(value)}"

        with self.lock:
            record = self.data["facts"].setdefault(
                fact_id,
                {
                    "subject": subject,
                    "relation": relation,
                    "value": value,
                    "confidence": clamp(confidence),
                    "times_seen": 0,
                    "sources": {},
                    "created": now(),
                    "last_seen": now(),
                }
            )

            record["times_seen"] += 1
            record["last_seen"] = now()

            old_conf = record.get("confidence", 0.5)

            source_count = len(record.get("sources", {}))

            record["confidence"] = weighted_average(
                old_conf,
                clamp(confidence),
                0.10
            )

            if source_id:
                record["sources"][source_id] = (
                    record["sources"].get(source_id, 0) + 1
                )

            # Independent sources increase confidence.
            if len(record["sources"]) > source_count:
                record["confidence"] = clamp(
                    record["confidence"] + 0.04
                )

            self.increment("facts_learned")

            self.prune_facts()

        return fact_id

    def prune_facts(self) -> None:
        if len(self.data["facts"]) <= MAX_FACTS:
            return

        items = sorted(
            self.data["facts"].items(),
            key=lambda item: (
                item[1].get("confidence", 0),
                item[1].get("times_seen", 0),
                len(item[1].get("sources", {}))
            ),
            reverse=True
        )

        self.data["facts"] = dict(items[:MAX_FACTS])

    # --------------------------------------------------------
    # Concepts
    # --------------------------------------------------------

    def add_concept(
        self,
        concept: str,
        words: Iterable[str],
        source_id: str = ""
    ) -> None:

        concept = normalize_text(concept)

        if not concept:
            return

        with self.lock:
            record = self.data["concepts"].setdefault(
                concept,
                {
                    "words": {},
                    "count": 0,
                    "sources": {},
                    "confidence": 0.3,
                    "created": now(),
                    "last_seen": now(),
                }
            )

            record["count"] += 1
            record["last_seen"] = now()

            for word in words:
                word = normalize_text(word)
                if word:
                    record["words"][word] = (
                        record["words"].get(word, 0) + 1
                    )

            if source_id:
                record["sources"][source_id] = (
                    record["sources"].get(source_id, 0) + 1
                )

            record["confidence"] = clamp(
                record["confidence"] + 0.01
            )

            self.prune_concepts()

    def prune_concepts(self) -> None:
        if len(self.data["concepts"]) <= MAX_CONCEPTS:
            return

        items = sorted(
            self.data["concepts"].items(),
            key=lambda item: item[1].get("count", 0),
            reverse=True
        )

        self.data["concepts"] = dict(items[:MAX_CONCEPTS])

    # --------------------------------------------------------
    # Associations
    # --------------------------------------------------------

    def associate(
        self,
        left: str,
        right: str,
        strength: float = 1.0,
        source_id: str = ""
    ) -> None:

        left = normalize_text(left)
        right = normalize_text(right)

        if not left or not right or left == right:
            return

        if len(left) > 80 or len(right) > 80:
            return

        key = f"{left}||{right}"

        with self.lock:
            record = self.data["associations"].setdefault(
                key,
                {
                    "left": left,
                    "right": right,
                    "strength": 0.0,
                    "count": 0,
                    "sources": {},
                    "last_seen": now(),
                }
            )

            record["strength"] += strength
            record["count"] += 1
            record["last_seen"] = now()

            if source_id:
                record["sources"][source_id] = (
                    record["sources"].get(source_id, 0) + 1
                )

            self.prune_associations()

    def prune_associations(self) -> None:
        if len(self.data["associations"]) <= MAX_ASSOCIATIONS:
            return

        items = sorted(
            self.data["associations"].items(),
            key=lambda item: item[1].get("strength", 0),
            reverse=True
        )

        self.data["associations"] = dict(items[:MAX_ASSOCIATIONS])

    # --------------------------------------------------------
    # Source records
    # --------------------------------------------------------

    def add_source(
        self,
        source_id: str,
        source_type: str,
        location: str,
        title: str = ""
    ) -> None:

        with self.lock:
            self.data["sources"].setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "source_type": source_type,
                    "location": location,
                    "title": title or location,
                    "added": now(),
                    "last_read": 0,
                    "times_read": 0,
                    "words_learned": 0,
                    "active": True,
                }
            )

    def mark_source_read(
        self,
        source_id: str,
        words_learned: int = 0
    ) -> None:

        with self.lock:
            if source_id not in self.data["sources"]:
                return

            source = self.data["sources"][source_id]

            source["last_read"] = now()
            source["times_read"] += 1
            source["words_learned"] += words_learned

            self.increment("sources_read")

    # --------------------------------------------------------
    # Feedback
    # --------------------------------------------------------

    def reinforce(
        self,
        user_input: str,
        response: str,
        correct: bool
    ) -> None:

        key = normalize_text(user_input)

        with self.lock:
            record = self.data["inputs"].get(key)

            if not record:
                return

            response_record = record["responses"].get(response)

            if not response_record:
                return

            if correct:
                response_record["correct"] += 1
                self.increment("correct")

                response_record["score"] = clamp(
                    response_record["score"] + 0.10
                )

                record["confidence"] = clamp(
                    record["confidence"] + 0.04
                )

            else:
                response_record["incorrect"] += 1
                self.increment("incorrect")

                response_record["score"] = clamp(
                    response_record["score"] - 0.15
                )

                record["confidence"] = clamp(
                    record["confidence"] - 0.06
                )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    def search_inputs(
        self,
        query: str,
        limit: int = MAX_CANDIDATES
    ) -> List[Dict[str, Any]]:

        results = []

        for stored, memory in self.data["inputs"].items():
            score = combined_similarity(query, stored)

            if score >= MIN_MATCH_SCORE:
                results.append({
                    "input": stored,
                    "score": score,
                    "memory": memory,
                })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:limit]

    def search_facts(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:

        qwords = word_set(query)
        results = []

        for fact in self.data["facts"].values():
            text = (
                fact["subject"] + " " +
                fact["relation"] + " " +
                fact["value"]
            )

            score = word_overlap(query, text)

            if qwords & word_set(text):
                score += fact.get("confidence", 0) * 0.25

            if score > 0:
                results.append({
                    "score": score,
                    "fact": fact,
                })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:limit]

    def search_sentences(
        self,
        query: str,
        limit: int = 8
    ) -> List[Dict[str, Any]]:

        results = []

        for record in self.data["sentences"].values():
            score = combined_similarity(
                query,
                record["text"]
            )

            if score >= 0.30:
                results.append({
                    "score": score,
                    "record": record,
                })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:limit]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        stats = dict(self.data["stats"])

        stats["inputs"] = len(self.data["inputs"])
        stats["words"] = len(self.data["words"])
        stats["facts"] = len(self.data["facts"])
        stats["concepts"] = len(self.data["concepts"])
        stats["associations"] = len(
            self.data["associations"]
        )
        stats["sentences"] = len(
            self.data["sentences"]
        )
        stats["sources"] = len(self.data["sources"])

        total_feedback = (
            stats.get("correct", 0) +
            stats.get("incorrect", 0)
        )

        stats["accuracy"] = (
            stats.get("correct", 0) /
            total_feedback * 100
            if total_feedback else 0
        )

        return stats


# ============================================================
# KNOWLEDGE EXTRACTOR
# ============================================================

class KnowledgeExtractor:
    """
    Turns documents into structured learning events.
    """

    SUBJECT_PATTERNS = [
        re.compile(
            r"^(.{2,80}?)\s+is\s+(.{2,400})[.!?]$",
            re.I
        ),
        re.compile(
            r"^(.{2,80}?)\s+are\s+(.{2,400})[.!?]$",
            re.I
        ),
        re.compile(
            r"^(.{2,80}?)\s+was\s+(.{2,400})[.!?]$",
            re.I
        ),
        re.compile(
            r"^(.{2,80}?)\s+has\s+(.{2,400})[.!?]$",
            re.I
        ),
        re.compile(
            r"^(.{2,80}?)\s+means\s+(.{2,400})[.!?]$",
            re.I
        ),
        re.compile(
            r"^(.{2,80}?)\s+refers to\s+(.{2,400})[.!?]$",
            re.I
        ),
    ]

    RELATIONS = {
        "is": "definition",
        "are": "definition",
        "was": "description",
        "has": "property",
        "means": "meaning",
        "refers to": "meaning",
    }

    def __init__(self, brain: Brain):
        self.brain = brain

    def extract(
        self,
        text: str,
        source_id: str
    ) -> Dict[str, int]:

        text = text[:MAX_SOURCE_TEXT]

        sentences = sentence_split(text)

        counts = {
            "sentences": 0,
            "facts": 0,
            "words": 0,
            "associations": 0,
            "concepts": 0,
        }

        for sentence in sentences:
            if len(sentence) < 8:
                continue

            self.brain.learn_sentence(
                sentence,
                source_id
            )

            self.brain.learn_words(
                sentence,
                source_id
            )

            self.brain.learn_ngrams(sentence)

            counts["sentences"] += 1

            facts = self.extract_facts(
                sentence,
                source_id
            )

            counts["facts"] += facts

            associations = self.extract_associations(
                sentence,
                source_id
            )

            counts["associations"] += associations

            concepts = self.extract_concepts(
                sentence,
                source_id
            )

            counts["concepts"] += concepts

        counts["words"] = len(tokenize(text))

        return counts

    def extract_facts(
        self,
        sentence: str,
        source_id: str
    ) -> int:

        sentence = sentence.strip()

        for pattern in self.SUBJECT_PATTERNS:
            match = pattern.match(sentence)

            if not match:
                continue

            subject = match.group(1).strip()
            value = match.group(2).strip()

            relation_word = "is"

            lower = sentence.lower()

            if " are " in lower:
                relation_word = "are"
            elif " was " in lower:
                relation_word = "was"
            elif " has " in lower:
                relation_word = "has"
            elif " means " in lower:
                relation_word = "means"
            elif " refers to " in lower:
                relation_word = "refers to"

            relation = self.RELATIONS.get(
                relation_word,
                "description"
            )

            if len(subject.split()) <= 12:
                self.brain.add_fact(
                    subject,
                    relation,
                    value,
                    source_id,
                    confidence=0.50
                )

                return 1

        return 0

    def extract_associations(
        self,
        sentence: str,
        source_id: str
    ) -> int:

        words = useful_words(sentence)

        if len(words) < 2:
            return 0

        unique = list(dict.fromkeys(words))

        # Nearby words become associations.
        count = 0

        for index, left in enumerate(unique):
            for right in unique[index + 1:index + 5]:
                self.brain.associate(
                    left,
                    right,
                    strength=1.0,
                    source_id=source_id
                )
                count += 1

        return count

    def extract_concepts(
        self,
        sentence: str,
        source_id: str
    ) -> int:

        words = useful_words(sentence)

        if len(words) < 3:
            return 0

        # High-frequency useful words act as emerging concepts.
        counts = word_counter(sentence)

        candidates = [
            word for word, count in counts.items()
            if len(word) >= 4 and count >= 1
        ]

        if not candidates:
            return 0

        created = 0

        for concept in candidates[:5]:
            related = [
                word for word in words
                if word != concept
            ][:12]

            self.brain.add_concept(
                concept,
                related,
                source_id
            )

            created += 1

        return created


# ============================================================
# FILE LEARNER
# ============================================================

class FileLearner:
    """
    Learns from local documents.
    """

    SUPPORTED = {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".log",
    }

    def __init__(
        self,
        brain: Brain,
        extractor: KnowledgeExtractor
    ):
        self.brain = brain
        self.extractor = extractor

    def read_file(self, path: Path) -> str:
        try:
            return path.read_text(
                encoding="utf-8",
                errors="ignore"
            )[:MAX_SOURCE_TEXT]
        except OSError:
            return ""

    def csv_to_text(self, path: Path) -> str:
        rows = []

        try:
            with path.open(
                "r",
                encoding="utf-8",
                errors="ignore",
                newline=""
            ) as file:

                reader = csv.reader(file)

                for row in reader:
                    rows.append(" ".join(row))

        except OSError:
            return ""

        return "\n".join(rows)

    def json_to_text(self, path: Path) -> str:
        try:
            with path.open(
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as file:

                data = json.load(file)

            return self.flatten_json(data)

        except (OSError, json.JSONDecodeError):
            return ""

    def flatten_json(
        self,
        value: Any,
        prefix: str = ""
    ) -> str:

        output = []

        if isinstance(value, dict):

            for key, child in value.items():

                text = self.flatten_json(
                    child,
                    str(key)
                )

                if text:
                    output.append(
                        f"{key}: {text}"
                    )

        elif isinstance(value, list):

            for child in value:
                output.append(
                    self.flatten_json(child)
                )

        else:

            output.append(str(value))

        return " ".join(
            item for item in output
            if item
        )

    def learn_file(self, path: Path) -> Dict[str, Any]:

        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED:
            return {
                "ok": False,
                "reason": "unsupported"
            }

        source_id = "file:" + str(
            path.resolve()
        )

        self.brain.add_source(
            source_id,
            "file",
            str(path.resolve()),
            path.name
        )

        if suffix == ".csv":
            text = self.csv_to_text(path)
        elif suffix == ".json":
            text = self.json_to_text(path)
        else:
            text = self.read_file(path)

        if not text:
            return {
                "ok": False,
                "reason": "empty_or_unreadable"
            }

        result = self.extractor.extract(
            text,
            source_id
        )

        self.brain.mark_source_read(
            source_id,
            result["words"]
        )

        self.brain.data["documents"][source_id] = {
            "path": str(path.resolve()),
            "title": path.name,
            "words": result["words"],
            "sentences": result["sentences"],
            "last_learned": now(),
        }

        return {
            "ok": True,
            "source": source_id,
            **result
        }

    def learn_folder(
        self,
        folder: str
    ) -> Dict[str, int]:

        path = Path(folder)

        totals = Counter()

        if not path.exists():
            return {
                "files": 0,
                "words": 0,
                "facts": 0,
                "sentences": 0
            }

        for file_path in path.rglob("*"):

            if not file_path.is_file():
                continue

            result = self.learn_file(
                file_path
            )

            if result.get("ok"):
                totals["files"] += 1
                totals["words"] += result.get(
                    "words", 0
                )
                totals["facts"] += result.get(
                    "facts", 0
                )
                totals["sentences"] += result.get(
                    "sentences", 0
                )

        self.brain.save()

        return dict(totals)


# ============================================================
# WEB LEARNER
# ============================================================

class WebLearner:
    """
    Reads web pages using urllib.

    Web learning is disabled by default and must be explicitly
    enabled in the application settings.
    """

    def __init__(
        self,
        brain: Brain,
        extractor: KnowledgeExtractor
    ):
        self.brain = brain
        self.extractor = extractor

    def fetch(
        self,
        url: str
    ) -> Tuple[str, List[str]]:

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT
            }
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=REQUEST_TIMEOUT
            ) as response:

                content_type = response.headers.get(
                    "Content-Type",
                    ""
                )

                raw = response.read(
                    MAX_SOURCE_TEXT
                )

                charset = response.headers.get_content_charset(
                    "utf-8"
                )

                text = raw.decode(
                    charset or "utf-8",
                    errors="ignore"
                )

                if "html" in content_type.lower():
                    parser = TextExtractor()
                    parser.feed(text)

                    return (
                        parser.text(),
                        parser.links
                    )

                return text, []

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError
        ) as error:

            print(
                f"[web] Could not read {url}: {error}"
            )

            return "", []

    def learn_url(
        self,
        url: str,
        crawl: bool = False,
        depth: int = 0,
        visited: Optional[Set[str]] = None,
        page_counter: Optional[List[int]] = None
    ) -> int:

        if visited is None:
            visited = set()

        if page_counter is None:
            page_counter = [0]

        if url in visited:
            return 0

        if page_counter[0] >= MAX_CRAWL_PAGES:
            return 0

        visited.add(url)
        page_counter[0] += 1

        text, links = self.fetch(url)

        if not text:
            return 0

        source_id = "url:" + url

        self.brain.add_source(
            source_id,
            "url",
            url,
            url
        )

        result = self.extractor.extract(
            text,
            source_id
        )

        self.brain.mark_source_read(
            source_id,
            result["words"]
        )

        learned_pages = 1

        if (
            crawl and
            self.brain.data["settings"].get(
                "allow_crawl",
                False
            ) and
            depth < self.brain.data["settings"].get(
                "crawl_depth",
                MAX_CRAWL_DEPTH
            )
        ):

            base = url

            for link in links:

                if learned_pages >= MAX_CRAWL_PAGES:
                    break

                absolute = urllib.parse.urljoin(
                    base,
                    link
                )

                parsed = urllib.parse.urlparse(
                    absolute
                )

                if parsed.scheme not in {
                    "http",
                    "https"
                }:
                    continue

                # Keep crawling on the same host.
                if parsed.netloc != urllib.parse.urlparse(
                    url
                ).netloc:
                    continue

                learned_pages += self.learn_url(
                    absolute,
                    crawl=True,
                    depth=depth + 1,
                    visited=visited,
                    page_counter=page_counter
                )

        self.brain.save()

        return learned_pages


# ============================================================
# LANGUAGE MODEL
# ============================================================

class LocalLanguageModel:
    """
    Small local n-gram model.

    It does not understand language like a large neural model.
    It learns statistical word transitions from the text it receives.
    """

    def __init__(self, brain: Brain):
        self.brain = brain

    def next_word_candidates(
        self,
        previous_words: List[str],
        limit: int = 10
    ) -> List[Tuple[str, float]]:

        previous_words = [
            clean_text(word)
            for word in previous_words
            if word
        ]

        candidates = Counter()

        if len(previous_words) >= 2:

            prefix = " ".join(
                previous_words[-2:]
            )

            bucket = self.brain.data[
                "ngrams"
            ]["3"]

            for gram, count in bucket.items():

                parts = gram.split()

                if len(parts) == 3 and (
                    " ".join(parts[:2]) == prefix
                ):

                    candidates[parts[2]] += count

        if not candidates and previous_words:

            prefix = previous_words[-1]

            bucket = self.brain.data[
                "ngrams"
            ]["2"]

            for gram, count in bucket.items():

                parts = gram.split()

                if len(parts) == 2 and parts[0] == prefix:
                    candidates[parts[1]] += count

        return candidates.most_common(limit)

    def generate(
        self,
        seed: str,
        length: int = 20
    ) -> str:

        words = tokenize(seed)

        if not words:
            return ""

        for _ in range(length):

            candidates = self.next_word_candidates(
                words
            )

            if not candidates:
                break

            total = sum(
                count for _, count in candidates
            )

            if total <= 0:
                break

            # Weighted random choice.
            roll = random.uniform(0, total)
            current = 0.0
            selected = candidates[0][0]

            for word, count in candidates:
                current += count

                if roll <= current:
                    selected = word
                    break

            words.append(selected)

        return " ".join(words)


# ============================================================
# RESPONSE ENGINE
# ============================================================

class ResponseEngine:
    """
    Combines learned conversations, facts, sentences, concepts,
    and the local n-gram model.
    """

    def __init__(self, brain: Brain):
        self.brain = brain
        self.language_model = LocalLanguageModel(
            brain
        )

    def confidence_label(
        self,
        score: float
    ) -> str:

        if score >= VERY_GOOD_MATCH_SCORE:
            return "high"

        if score >= GOOD_MATCH_SCORE:
            return "medium"

        return "low"

    def select_memory_response(
        self,
        memory: Dict[str, Any]
    ) -> Optional[str]:

        responses = memory.get("responses", {})

        if not responses:
            return None

        weighted = []

        for text, record in responses.items():

            score = record.get("score", 0.5)

            count = record.get("count", 1)

            correct = record.get("correct", 0)

            incorrect = record.get("incorrect", 0)

            weight = (
                max(score, 0.05) *
                (1 + math.log1p(count)) *
                (1 + correct) /
                (1 + incorrect)
            )

            repetitions = min(
                20,
                max(1, int(weight * 5))
            )

            weighted.extend(
                [text] * repetitions
            )

        if not weighted:
            return None

        return random.choice(weighted)

    def fact_answer(
        self,
        query: str
    ) -> Optional[Tuple[str, float]]:

        matches = self.brain.search_facts(
            query
        )

        if not matches:
            return None

        best = matches[0]

        fact = best["fact"]

        confidence = clamp(
            best["score"] *
            (0.6 + fact.get(
                "confidence",
                0.5
            ) * 0.4)
        )

        answer = (
            f"{fact['subject']} "
            f"{fact['relation']}: "
            f"{fact['value']}"
        )

        return answer, confidence

    def sentence_answer(
        self,
        query: str
    ) -> Optional[Tuple[str, float]]:

        matches = self.brain.search_sentences(
            query
        )

        if not matches:
            return None

        best = matches[0]

        return (
            best["record"]["text"],
            clamp(best["score"] * 0.85)
        )

    def concept_answer(
        self,
        query: str
    ) -> Optional[Tuple[str, float]]:

        qwords = word_set(query)

        candidates = []

        for concept, record in (
            self.brain.data["concepts"].items()
        ):

            concept_words = set(
                record.get("words", {}).keys()
            )

            overlap = len(
                qwords & (
                    concept_words | {concept}
                )
            )

            if overlap:

                score = (
                    overlap /
                    max(1, len(qwords))
                )

                score *= (
                    0.5 +
                    record.get(
                        "confidence",
                        0.3
                    ) * 0.5
                )

                candidates.append(
                    (score, concept, record)
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        score, concept, record = candidates[0]

        related = sorted(
            record.get("words", {}).items(),
            key=lambda item: item[1],
            reverse=True
        )[:8]

        related_words = ", ".join(
            word for word, _ in related
        )

        answer = (
            f"I associate {concept} with "
            f"{related_words}."
        )

        return answer, clamp(score)

    def answer(
        self,
        query: str
    ) -> Tuple[Optional[str], float, str]:

        matches = self.brain.search_inputs(
            query
        )

        if matches:

            best = matches[0]

            response = self.select_memory_response(
                best["memory"]
            )

            if response:

                score = best["score"]

                # Feedback-adjusted confidence.
                score = clamp(
                    score * 0.75 +
                    best["memory"].get(
                        "confidence",
                        0.5
                    ) * 0.25
                )

                return (
                    response,
                    score,
                    "conversation"
                )

        fact = self.fact_answer(query)

        if fact:
            return (
                fact[0],
                fact[1],
                "fact"
            )

        sentence = self.sentence_answer(query)

        if sentence:
            return (
                sentence[0],
                sentence[1],
                "knowledge"
            )

        concept = self.concept_answer(query)

        if concept:
            return (
                concept[0],
                concept[1],
                "concept"
            )

        return None, 0.0, "unknown"


# ============================================================
# SELF LEARNING ENGINE
# ============================================================

class SelfLearningEngine:
    """
    The part that allows the program to keep learning without
    requiring a new manual teaching event every time.

    It watches the knowledge directory and configured source queue.
    """

    def __init__(
        self,
        brain: Brain,
        file_learner: FileLearner,
        web_learner: WebLearner
    ):
        self.brain = brain
        self.file_learner = file_learner
        self.web_learner = web_learner

        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.last_cycle = 0.0

    def start(self) -> None:

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True
        )

        self.thread.start()

    def stop(self) -> None:

        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def _loop(self) -> None:

        while self.running:

            try:

                if self.brain.data["settings"].get(
                    "auto_learn",
                    True
                ):

                    interval = self.brain.data[
                        "settings"
                    ].get(
                        "learn_interval",
                        AUTO_LEARN_INTERVAL
                    )

                    if now() - self.last_cycle >= interval:

                        self.run_cycle()

            except Exception as error:

                print(
                    f"[self-learning] {error}"
                )

            time.sleep(2)

    def run_cycle(self) -> Dict[str, Any]:

        self.last_cycle = now()

        results = {
            "files": 0,
            "words": 0,
            "facts": 0,
            "sentences": 0,
            "web_pages": 0,
        }

        folder_result = (
            self.file_learner.learn_folder(
                KNOWLEDGE_DIR
            )
        )

        results["files"] = folder_result.get(
            "files",
            0
        )

        results["words"] = folder_result.get(
            "words",
            0
        )

        results["facts"] = folder_result.get(
            "facts",
            0
        )

        results["sentences"] = folder_result.get(
            "sentences",
            0
        )

        if self.brain.data["settings"].get(
            "allow_web",
            False
        ):

            queue = self.brain.data.get(
                "queue",
                []
            )

            remaining = []

            for item in queue:

                if not item.get("active", True):
                    continue

                url = item.get("url", "")

                if not url:
                    continue

                pages = self.web_learner.learn_url(
                    url,
                    crawl=item.get(
                        "crawl",
                        False
                    )
                )

                results["web_pages"] += pages

            self.brain.data["queue"] = remaining

        self.consolidate()

        self.brain.increment(
            "self_cycles"
        )

        self.brain.save()

        return results

    def consolidate(self) -> None:
        """
        Strengthens knowledge that repeatedly appears across
        sentences and sources.
        """

        words = self.brain.data["words"]

        if not words:
            return

        common_words = [
            word for word, record in sorted(
                words.items(),
                key=lambda item: item[1].get(
                    "count",
                    0
                ),
                reverse=True
            )[:500]
        ]

        # Rebuild a small number of strong associations.
        for index, left in enumerate(common_words):

            left_record = words[left]

            source_set = set(
                left_record.get(
                    "sources",
                    {}
                ).keys()
            )

            if not source_set:
                continue

            for right in common_words[
                index + 1:index + 12
            ]:

                right_record = words[right]

                right_sources = set(
                    right_record.get(
                        "sources",
                        {}
                    ).keys()
                )

                shared = source_set & right_sources

                if shared:

                    strength = (
                        math.log1p(
                            len(shared)
                        ) *
                        0.2
                    )

                    self.brain.associate(
                        left,
                        right,
                        strength=strength,
                        source_id=(
                            "consolidation"
                        )
                    )


# ============================================================
# COMMAND SYSTEM
# ============================================================

class CommandSystem:

    def __init__(
        self,
        app: "AutonomousAI"
    ):
        self.app = app

    def execute(
        self,
        command: str
    ) -> Optional[str]:

        command = command.strip()

        lower = command.lower()

        if lower == "/help":
            return self.help()

        if lower == "/stats":
            return self.stats()

        if lower == "/memory":
            return self.memory()

        if lower == "/facts":
            return self.facts()

        if lower == "/sources":
            return self.sources()

        if lower == "/words":
            return self.words()

        if lower == "/topics":
            return self.topics()

        if lower == "/associations":
            return self.associations()

        if lower == "/brain":
            return self.brain_info()

        if lower == "/selflearn":
            return self.selflearn()

        if lower == "/save":
            self.app.brain.save()
            return "Brain saved."

        if lower == "/auto on":
            self.app.brain.data[
                "settings"
            ]["auto_learn"] = True
            return "Automatic learning is ON."

        if lower == "/auto off":
            self.app.brain.data[
                "settings"
            ]["auto_learn"] = False
            return "Automatic learning is OFF."

        if lower == "/web on":
            self.app.brain.data[
                "settings"
            ]["allow_web"] = True
            return "Web learning is ON."

        if lower == "/web off":
            self.app.brain.data[
                "settings"
            ]["allow_web"] = False
            return "Web learning is OFF."

        if lower == "/crawl on":
            self.app.brain.data[
                "settings"
            ]["allow_crawl"] = True
            return "Crawling is ON."

        if lower == "/crawl off":
            self.app.brain.data[
                "settings"
            ]["allow_crawl"] = False
            return "Crawling is OFF."

        if lower.startswith("/learn "):
            target = command[7:].strip()
            return self.learn_target(target)

        if lower.startswith("/learnurl "):
            url = command[10:].strip()
            return self.learn_url(url)

        if lower.startswith("/addsource "):
            url = command[11:].strip()
            return self.add_source(url)

        if lower.startswith("/forget "):
            text = command[8:].strip()
            return self.forget(text)

        if lower == "/clear":
            return self.clear()

        if lower.startswith("/search "):
            query = command[8:].strip()
            return self.search(query)

        if lower.startswith("/generate "):
            seed = command[10:].strip()
            return self.generate(seed)

        if lower == "/export":
            return self.export()

        return None

    def help(self) -> str:
        return """
============================================================
COMMANDS
============================================================

/help
    Show this help.

/stats
    Show learning statistics.

/memory
    Show learned conversation patterns.

/facts
    Show learned facts.

/sources
    Show files and web sources.

/words
    Show frequently learned words.

/topics
    Show emerging concepts.

/associations
    Show strong word relationships.

/brain
    Show brain information.

/learn <path>
    Learn one file or an entire folder.

/learnurl <url>
    Learn one web page. Web learning must be ON.

/addsource <url>
    Add a URL to the automatic learning queue.

/selflearn
    Run a learning cycle immediately.

/auto on
/auto off
    Enable/disable automatic background learning.

/web on
/web off
    Enable/disable web learning.

/crawl on
/crawl off
    Enable/disable same-site crawling.

/search <text>
    Search learned knowledge.

/generate <text>
    Generate a small local n-gram continuation.

/save
    Save the brain.

/forget <text>
    Forget a learned conversation input.

/clear
    Delete learned data after confirmation.

/export
    Create a backup brain file.

Type quit to exit.
============================================================
"""

    def stats(self) -> str:
        stats = self.app.brain.statistics()

        return f"""
============================================================
BRAIN STATISTICS
============================================================

Messages:       {stats['messages']}
Learning events:{stats['learn_events']}
Self cycles:    {stats['self_cycles']}

Inputs:         {stats['inputs']}
Words:          {stats['words']}
Facts:          {stats['facts']}
Concepts:       {stats['concepts']}
Sentences:      {stats['sentences']}
Associations:   {stats['associations']}
Sources:        {stats['sources']}

Correct:        {stats['correct']}
Incorrect:      {stats['incorrect']}
Feedback rate:  {stats['accuracy']:.1f}%

Last update:
{timestamp_string(self.app.brain.data['updated'])}
============================================================
"""

    def memory(self) -> str:
        memories = self.app.brain.data["inputs"]

        if not memories:
            return "No conversation memories yet."

        lines = [
            "LEARNED CONVERSATIONS",
            ""
        ]

        for question, record in list(
            memories.items()
        )[-40:]:

            lines.append(
                f"INPUT: {question}"
            )

            for response, info in record[
                "responses"
            ].items():

                lines.append(
                    f"  -> {response}"
                    f"  [score={info.get('score', 0.5):.2f}]"
                )

            lines.append("")

        return "\n".join(lines)

    def facts(self) -> str:
        facts = self.app.brain.data["facts"]

        if not facts:
            return "No structured facts yet."

        lines = [
            "LEARNED FACTS",
            ""
        ]

        sorted_facts = sorted(
            facts.values(),
            key=lambda x: (
                x.get("confidence", 0),
                x.get("times_seen", 0)
            ),
            reverse=True
        )

        for fact in sorted_facts[:40]:

            lines.append(
                f"- {fact['subject']} "
                f"[{fact['relation']}] "
                f"{fact['value']}"
            )

            lines.append(
                f"  confidence={fact.get('confidence', 0):.2f} "
                f"sources={len(fact.get('sources', {}))}"
            )

        return "\n".join(lines)

    def sources(self) -> str:
        sources = self.app.brain.data["sources"]

        if not sources:
            return "No sources have been learned yet."

        lines = [
            "SOURCES",
            ""
        ]

        for source in list(
            sources.values()
        )[-50:]:

            state = (
                "active"
                if source.get("active", True)
                else "inactive"
            )

            lines.append(
                f"- {source.get('source_type')}: "
                f"{source.get('location')}"
            )

            lines.append(
                f"  reads={source.get('times_read', 0)} "
                f"words={source.get('words_learned', 0)} "
                f"{state}"
            )

        return "\n".join(lines)

    def words(self) -> str:
        words = self.app.brain.data["words"]

        if not words:
            return "No words learned yet."

        items = sorted(
            words.items(),
            key=lambda item: item[1].get(
                "count",
                0
            ),
            reverse=True
        )[:50]

        lines = [
            "MOST FREQUENT WORDS",
            ""
        ]

        for word, record in items:

            lines.append(
                f"{word:<20} "
                f"{record.get('count', 0):>7} "
                f"sources={len(record.get('sources', {}))}"
            )

        return "\n".join(lines)

    def topics(self) -> str:
        concepts = self.app.brain.data["concepts"]

        if not concepts:
            return "No concepts learned yet."

        items = sorted(
            concepts.items(),
            key=lambda item: (
                item[1].get("count", 0),
                item[1].get("confidence", 0)
            ),
            reverse=True
        )[:40]

        lines = [
            "EMERGING CONCEPTS",
            ""
        ]

        for concept, record in items:

            related = sorted(
                record.get(
                    "words",
                    {}
                ).items(),
                key=lambda item: item[1],
                reverse=True
            )[:6]

            related_text = ", ".join(
                word for word, _ in related
            )

            lines.append(
                f"- {concept}: {related_text}"
            )

        return "\n".join(lines)

    def associations(self) -> str:
        associations = (
            self.app.brain.data[
                "associations"
            ]
        )

        if not associations:
            return "No associations yet."

        items = sorted(
            associations.items(),
            key=lambda item: item[1].get(
                "strength",
                0
            ),
            reverse=True
        )[:50]

        lines = [
            "STRONG ASSOCIATIONS",
            ""
        ]

        for _, record in items:

            lines.append(
                f"- {record['left']} <-> "
                f"{record['right']} "
                f"strength={record['strength']:.2f}"
            )

        return "\n".join(lines)

    def brain_info(self) -> str:
        data = self.app.brain.data

        return (
            f"Brain version: {data.get('version')}\n"
            f"Created: {timestamp_string(data.get('created'))}\n"
            f"Updated: {timestamp_string(data.get('updated'))}\n"
            f"File: {self.app.brain.path}"
        )

    def selflearn(self) -> str:
        result = self.app.self_learning.run_cycle()

        return (
            "Self-learning cycle complete.\n"
            f"Files: {result.get('files', 0)}\n"
            f"Web pages: {result.get('web_pages', 0)}\n"
            f"Words: {result.get('words', 0)}\n"
            f"Facts: {result.get('facts', 0)}\n"
            f"Sentences: {result.get('sentences', 0)}"
        )

    def learn_target(self, target: str) -> str:
        path = Path(target)

        if path.is_dir():

            result = (
                self.app.file_learner.learn_folder(
                    target
                )
            )

            return (
                f"Learned folder.\n"
                f"Files: {result.get('files', 0)}\n"
                f"Words: {result.get('words', 0)}\n"
                f"Facts: {result.get('facts', 0)}\n"
                f"Sentences: {result.get('sentences', 0)}"
            )

        if path.is_file():

            result = (
                self.app.file_learner.learn_file(
                    path
                )
            )

            if not result.get("ok"):
                return (
                    "I could not learn that file: "
                    + result.get(
                        "reason",
                        "unknown"
                    )
                )

            return (
                f"Learned {path.name}.\n"
                f"Words: {result.get('words', 0)}\n"
                f"Facts: {result.get('facts', 0)}\n"
                f"Sentences: {result.get('sentences', 0)}"
            )

        return "That file or folder does not exist."

    def learn_url(self, url: str) -> str:
        if not self.app.brain.data[
            "settings"
        ].get("allow_web", False):
            return (
                "Web learning is OFF. "
                "Use /web on first."
            )

        pages = self.app.web_learner.learn_url(
            url,
            crawl=False
        )

        return f"Learned {pages} web page(s)."

    def add_source(self, url: str) -> str:
        if not url.startswith(
            ("http://", "https://")
        ):
            return "That does not look like a web URL."

        queue = self.app.brain.data[
            "queue"
        ]

        queue.append({
            "url": url,
            "crawl": False,
            "active": True,
            "added": now(),
        })

        self.app.brain.save()

        return (
            "Added the URL to the learning queue. "
            "The background learner can process it "
            "when web learning is enabled."
        )

    def forget(self, text: str) -> str:
        key = normalize_text(text)

        if key in self.app.brain.data["inputs"]:

            del self.app.brain.data[
                "inputs"
            ][key]

            self.app.brain.save()

            return "Forgot that conversation memory."

        return "I couldn't find that memory."

    def clear(self) -> str:
        answer = input(
            "Type DELETE EVERYTHING to confirm: "
        )

        if answer != "DELETE EVERYTHING":
            return "Nothing was deleted."

        self.app.brain.data[
            "inputs"
        ] = {}

        self.app.brain.data[
            "words"
        ] = {}

        self.app.brain.data[
            "ngrams"
        ] = {
            "1": {},
            "2": {},
            "3": {},
        }

        self.app.brain.data[
            "facts"
        ] = {}

        self.app.brain.data[
            "concepts"
        ] = {}

        self.app.brain.data[
            "associations"
        ] = {}

        self.app.brain.data[
            "sentences"
        ] = {}

        self.app.brain.data[
            "documents"
        ] = {}

        self.app.brain.data[
            "stats"
        ] = self.app.brain.default_data()[
            "stats"
        ]

        self.app.brain.save()

        return "All learned knowledge was deleted."

    def search(self, query: str) -> str:
        facts = self.app.brain.search_facts(
            query,
            limit=5
        )

        sentences = self.app.brain.search_sentences(
            query,
            limit=5
        )

        inputs = self.app.brain.search_inputs(
            query,
            limit=5
        )

        lines = [
            "SEARCH RESULTS",
            ""
        ]

        if inputs:
            lines.append("CONVERSATIONS")

            for item in inputs:
                lines.append(
                    f"{item['score']:.2f} "
                    f"{item['input']}"
                )

        if facts:
            lines.append("")
            lines.append("FACTS")

            for item in facts:
                fact = item["fact"]

                lines.append(
                    f"{item['score']:.2f} "
                    f"{fact['subject']} "
                    f"{fact['value']}"
                )

        if sentences:
            lines.append("")
            lines.append("SENTENCES")

            for item in sentences:
                lines.append(
                    f"{item['score']:.2f} "
                    f"{item['record']['text']}"
                )

        if len(lines) == 2:
            return "No matching knowledge found."

        return "\n".join(lines)

    def generate(self, seed: str) -> str:
        text = self.app.response_engine.language_model.generate(
            seed,
            length=30
        )

        if not text:
            return "I haven't learned enough word patterns yet."

        return text

    def export(self) -> str:
        filename = (
            "autonomous_brain_backup_"
            + time.strftime("%Y%m%d_%H%M%S")
            + ".json"
        )

        try:
            with open(
                filename,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    self.app.brain.data,
                    file,
                    indent=2,
                    ensure_ascii=False
                )

            return f"Backup created: {filename}"

        except OSError as error:
            return f"Backup failed: {error}"


# ============================================================
# AUTONOMOUS AI APPLICATION
# ============================================================

class AutonomousAI:

    def __init__(self):

        self.brain = Brain()

        self.extractor = KnowledgeExtractor(
            self.brain
        )

        self.file_learner = FileLearner(
            self.brain,
            self.extractor
        )

        self.web_learner = WebLearner(
            self.brain,
            self.extractor
        )

        self.response_engine = ResponseEngine(
            self.brain
        )

        self.self_learning = SelfLearningEngine(
            self.brain,
            self.file_learner,
            self.web_learner
        )

        self.commands = CommandSystem(
            self
        )

        self.running = True

        self.last_save = now()

    # --------------------------------------------------------
    # Startup
    # --------------------------------------------------------

    def startup(self) -> None:

        Path(KNOWLEDGE_DIR).mkdir(
            parents=True,
            exist_ok=True
        )

        print()
        print("=" * 70)
        print("                 AUTONOMOUS LEARNING AI")
        print("=" * 70)
        print()
        print("Python-only local learning engine")
        print()
        print(
            "I learn from conversations, files, "
            "and optionally web sources."
        )
        print(
            "Put documents in the 'knowledge' folder "
            "and I can learn them automatically."
        )
        print()
        print(
            "Type /help for commands."
        )
        print(
            "Type quit to save and exit."
        )
        print()
        print("=" * 70)
        print()

        stats = self.brain.statistics()

        print(
            f"Loaded brain: "
            f"{stats['words']} words, "
            f"{stats['facts']} facts, "
            f"{stats['sources']} sources."
        )

        if self.brain.data["settings"].get(
            "auto_learn",
            True
        ):
            self.self_learning.start()

    # --------------------------------------------------------
    # Learn conversation
    # --------------------------------------------------------

    def learn_conversation(
        self,
        user_input: str,
        response: str
    ) -> None:

        self.brain.remember_input(
            user_input,
            response,
            "conversation"
        )

        self.extractor.extract(
            user_input,
            "conversation"
        )

        self.extractor.extract(
            response,
            "conversation"
        )

    # --------------------------------------------------------
    # Handle unknown input
    # --------------------------------------------------------

    def handle_unknown(
        self,
        user_input: str
    ) -> None:

        print()
        print(
            "I don't have a reliable answer for that yet."
        )

        print(
            "You can teach me a response, "
            "or type /learn <file/folder> "
            "to give me knowledge."
        )

        answer = input(
            "Teach a response? [y/n]: "
        ).strip().lower()

        if answer == "y":

            response = input(
                "What should I say? "
            ).strip()

            if response:

                self.learn_conversation(
                    user_input,
                    response
                )

                self.brain.save()

                print(
                    "Learned and saved."
                )

    # --------------------------------------------------------
    # Feedback
    # --------------------------------------------------------

    def feedback(
        self,
        user_input: str,
        response: str
    ) -> None:

        answer = input(
            "Was that useful? [y/n/enter]: "
        ).strip().lower()

        if answer == "y":

            self.brain.reinforce(
                user_input,
                response,
                True
            )

            print(
                "Reinforced that response."
            )

        elif answer == "n":

            self.brain.reinforce(
                user_input,
                response,
                False
            )

            print()

            correction = input(
                "What would be better? "
            ).strip()

            if correction:

                self.learn_conversation(
                    user_input,
                    correction
                )

                self.brain.reinforce(
                    user_input,
                    correction,
                    True
                )

                print(
                    "Learned the correction."
                )

    # --------------------------------------------------------
    # Process normal message
    # --------------------------------------------------------

    def process(self, user_input: str) -> None:

        stripped = user_input.strip()

        if not stripped:
            return

        if normalize_text(stripped) in {
            "quit",
            "exit",
            "stop"
        }:

            self.running = False
            return

        if stripped.startswith("/"):

            result = self.commands.execute(
                stripped
            )

            if result is not None:

                print()
                print(result)
                print()

                return

        self.brain.increment(
            "messages"
        )

        # Learn from the message itself.
        self.brain.learn_words(
            stripped,
            "conversation"
        )

        self.brain.learn_ngrams(
            stripped
        )

        response, score, mode = (
            self.response_engine.answer(
                stripped
            )
        )

        if response is None:

            self.handle_unknown(
                stripped
            )

            return

        label = (
            self.response_engine.confidence_label(
                score
            )
        )

        print()
        print(
            f"AI [{mode} | {label} {score:.2f}]:"
        )
        print(response)
        print()

        self.brain.increment(
            "responses"
        )

        self.feedback(
            stripped,
            response
        )

        self.brain.save_if_needed()

    # --------------------------------------------------------
    # Save scheduling
    # --------------------------------------------------------

    def save_if_needed(self) -> None:

        if not self.brain.data[
            "settings"
        ].get(
            "auto_save",
            True
        ):
            return

        interval = self.brain.data[
            "settings"
        ].get(
            "save_interval",
            AUTO_SAVE_INTERVAL
        )

        if now() - self.last_save >= interval:

            self.brain.save()

            self.last_save = now()

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    def run(self) -> None:

        self.startup()

        try:

            while self.running:

                try:

                    user_input = input(
                        "You: "
                    )

                    self.process(
                        user_input
                    )

                except KeyboardInterrupt:

                    print()
                    print(
                        "Saving..."
                    )
                    break

                except EOFError:

                    print()
                    break

                except Exception as error:

                    print()
                    print(
                        f"[error] {error}"
                    )
                    print()

                self.save_if_needed()

        finally:

            self.running = False

            self.self_learning.stop()

            self.brain.save()

            print(
                "Brain saved. Goodbye."
            )


# ============================================================
# BRAIN MIGRATION
# ============================================================

def migrate_old_brain() -> None:
    """
    If the older program's brain.json exists, copy basic
    conversation knowledge into the new format.

    This allows the program to continue using knowledge from
    the earlier version.
    """

    old_file = "brain.json"

    if not os.path.exists(old_file):
        return

    if os.path.exists(BRAIN_FILE):
        return

    try:

        with open(
            old_file,
            "r",
            encoding="utf-8"
        ) as file:

            old = json.load(file)

    except (
        OSError,
        json.JSONDecodeError
    ):
        return

    brain = Brain()

    old_inputs = old.get(
        "inputs",
        {}
    )

    for question, record in old_inputs.items():

        responses = record.get(
            "responses",
            []
        )

        for response in responses:

            brain.remember_input(
                question,
                response,
                "migrated:brain.json"
            )

    old_words = old.get(
        "words",
        {}
    )

    for word, record in old_words.items():

        brain.data["words"][word] = {
            "count": record.get(
                "count",
                1
            ),
            "first_seen": record.get(
                "first_seen",
                now()
            ),
            "last_seen": record.get(
                "last_seen",
                now()
            ),
            "sources": {
                "migrated:brain.json": record.get(
                    "count",
                    1
                )
            }
        }

    brain.save()

    print(
        "Migrated knowledge from old brain.json."
    )


# ============================================================
# BUILT-IN STARTER KNOWLEDGE
# ============================================================

def create_starter_knowledge() -> None:
    """
    Creates a small optional starter file.

    This is deliberately not a huge hard-coded knowledge base.
    The system is intended to learn from sources you provide.
    """

    folder = Path(KNOWLEDGE_DIR)

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    starter = folder / "starter.txt"

    if starter.exists():
        return

    text = """
Python is a programming language.
A computer is an electronic machine that processes information.
Programming is the process of writing instructions for computers.
Machine learning is a method where software learns patterns from data.
Data is information that can be stored and processed.
A file is a collection of information stored on a computer.
A program is a set of instructions executed by a computer.
An algorithm is a sequence of steps used to solve a problem.
A variable is a named place used to store a value in a program.
A function is a reusable block of program instructions.
"""

    try:
        starter.write_text(
            text.strip(),
            encoding="utf-8"
        )

    except OSError:
        pass


# ============================================================
# CONFIGURATION HELPERS
# ============================================================

def load_external_config(
    brain: Brain
) -> None:

    if not os.path.exists(CONFIG_FILE):
        return

    try:

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            config = json.load(file)

        if isinstance(config, dict):

            for key, value in config.items():

                if key in brain.data[
                    "settings"
                ]:
                    brain.data[
                        "settings"
                    ][key] = value

    except (
        OSError,
        json.JSONDecodeError
    ):
        pass


def create_config_file(
    brain: Brain
) -> None:

    if os.path.exists(CONFIG_FILE):
        return

    try:

        with open(
            CONFIG_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                brain.data["settings"],
                file,
                indent=2
            )

    except OSError:
        pass


# ============================================================
# ADVANCED SELF-LEARNING UTILITIES
# ============================================================

class KnowledgeGraph:
    """
    Lightweight graph view over learned concepts.
    """

    def __init__(self, brain: Brain):
        self.brain = brain

    def neighbors(
        self,
        concept: str,
        limit: int = 15
    ) -> List[Tuple[str, float]]:

        concept = normalize_text(concept)

        found = []

        for record in (
            self.brain.data[
                "associations"
            ].values()
        ):

            if record["left"] == concept:

                found.append(
                    (
                        record["right"],
                        record["strength"]
                    )
                )

            elif record["right"] == concept:

                found.append(
                    (
                        record["left"],
                        record["strength"]
                    )
                )

        found.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return found[:limit]

    def expand(
        self,
        words: Iterable[str],
        depth: int = 2
    ) -> Set[str]:

        frontier = set(
            normalize_text(word)
            for word in words
            if word
        )

        visited = set(frontier)

        for _ in range(depth):

            next_frontier = set()

            for word in frontier:

                for neighbor, _ in self.neighbors(
                    word,
                    limit=10
                ):

                    if neighbor not in visited:

                        visited.add(
                            neighbor
                        )

                        next_frontier.add(
                            neighbor
                        )

            frontier = next_frontier

            if not frontier:
                break

        return visited


class MemoryDecay:
    """
    Allows stale, low-value conversation memories to gradually
    lose priority without immediately deleting them.
    """

    def __init__(self, brain: Brain):
        self.brain = brain

    def apply(self) -> None:

        current = now()

        for record in (
            self.brain.data[
                "inputs"
            ].values()
        ):

            last_seen = record.get(
                "last_seen",
                current
            )

            age_days = (
                current - last_seen
            ) / 86400

            decay = min(
                0.20,
                age_days * 0.002
            )

            record["confidence"] = clamp(
                record.get(
                    "confidence",
                    0.5
                ) - decay
            )


class SourceAgreement:
    """
    Measures how strongly a fact is supported by multiple
    independent sources.
    """

    def __init__(self, brain: Brain):
        self.brain = brain

    def score_fact(
        self,
        fact: Dict[str, Any]
    ) -> float:

        confidence = fact.get(
            "confidence",
            0.5
        )

        sources = len(
            fact.get(
                "sources",
                {}
            )
        )

        agreement_bonus = min(
            0.30,
            max(0, sources - 1) * 0.05
        )

        return clamp(
            confidence +
            agreement_bonus
        )


class AutoCurator:
    """
    Periodically cleans and reorganizes learned knowledge.
    """

    def __init__(
        self,
        brain: Brain
    ):
        self.brain = brain
        self.decay = MemoryDecay(
            brain
        )
        self.agreement = SourceAgreement(
            brain
        )

    def run(self) -> None:

        self.decay.apply()

        for fact in (
            self.brain.data[
                "facts"
            ].values()
        ):

            fact["confidence"] = (
                self.agreement.score_fact(
                    fact
                )
            )

        self.remove_duplicate_sentences()

    def remove_duplicate_sentences(self) -> None:

        sentences = list(
            self.brain.data[
                "sentences"
            ].items()
        )

        if len(sentences) < 2:
            return

        # Only compare a limited window to avoid O(n^2) behavior
        # over huge brains.
        sample = sentences[-1000:]

        removed = set()

        for i in range(len(sample)):

            key_a, a = sample[i]

            if key_a in removed:
                continue

            for j in range(
                i + 1,
                min(i + 80, len(sample))
            ):

                key_b, b = sample[j]

                if key_b in removed:
                    continue

                score = combined_similarity(
                    a["text"],
                    b["text"]
                )

                if score >= 0.94:

                    # Keep the one with more sources.
                    sources_a = len(
                        a.get("sources", {})
                    )

                    sources_b = len(
                        b.get("sources", {})
                    )

                    if sources_a >= sources_b:

                        a["count"] += b.get(
                            "count",
                            0
                        )

                        removed.add(
                            key_b
                        )

                    else:

                        b["count"] += a.get(
                            "count",
                            0
                        )

                        removed.add(
                            key_a
                        )

        for key in removed:
            self.brain.data[
                "sentences"
            ].pop(
                key,
                None
            )


# ============================================================
# PATCH SELF-LEARNING ENGINE WITH CURATION
# ============================================================

def attach_advanced_learning(
    app: AutonomousAI
) -> None:

    app.graph = KnowledgeGraph(
        app.brain
    )

    app.curator = AutoCurator(
        app.brain
    )

    original_cycle = (
        app.self_learning.run_cycle
    )

    def enhanced_cycle():

        result = original_cycle()

        app.curator.run()

        app.brain.save()

        return result

    app.self_learning.run_cycle = enhanced_cycle


# ============================================================
# STARTUP
# ============================================================

def main() -> None:

    migrate_old_brain()

    create_starter_knowledge()

    brain = Brain()

    load_external_config(
        brain
    )

    create_config_file(
        brain
    )

    brain.save()

    app = AutonomousAI()

    # Replace the newly created brain with the configured one
    # only when configuration was loaded before app creation.
    app.brain = brain

    app.extractor = KnowledgeExtractor(
        app.brain
    )

    app.file_learner = FileLearner(
        app.brain,
        app.extractor
    )

    app.web_learner = WebLearner(
        app.brain,
        app.extractor
    )

    app.response_engine = ResponseEngine(
        app.brain
    )

    app.self_learning = SelfLearningEngine(
        app.brain,
        app.file_learner,
        app.web_learner
    )

    app.commands = CommandSystem(
        app
    )

    attach_advanced_learning(
        app
    )

    app.run()


# ============================================================
# END
# ============================================================

if __name__ == "__main__":
    main()
