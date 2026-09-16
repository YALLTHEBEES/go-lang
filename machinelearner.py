import json
import os
import re
import random
import math
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# ============================================================
# CONFIGURATION
# ============================================================

BRAIN_FILE = "brain.json"
VERSION = "3.0"

MIN_SIMILARITY = 0.25
GOOD_SIMILARITY = 0.60

MAX_MEMORY_KEYS = 5000
MAX_RESPONSES_PER_INPUT = 15

# Removed interrogatives (what, how, why, etc.) to keep query intent context
STOP_WORDS = {
    "a", "an", "the", "is", "are", "am", "to", "of",
    "in", "on", "at", "for", "and", "or", "but", "i",
    "you", "he", "she", "it", "we", "they", "me", "my",
    "your", "this", "that", "with", "do", "does", "did",
    "be", "been", "being", "was", "were", "as", "from",
    "by", "about"
}

# ============================================================
# TEXT PROCESSING & NLP PIPELINE
# ============================================================

def clean_text(text):
    """Normalizes text by removing special characters and lowering case."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize(text):
    """Splits normalized text into a list of words."""
    return clean_text(text).split()

def useful_words(text):
    """Filters out common stop words to isolate meaningful terms."""
    return [word for word in tokenize(text) if word not in STOP_WORDS]

def word_set(text):
    """Returns unique meaningful words from text."""
    return set(useful_words(text))

def word_count(text):
    """Generates a frequency distribution map of meaningful words."""
    return Counter(useful_words(text))

# ============================================================
# PERFORMANCE SIMILARITY METRICS
# ============================================================

def Jaccard_similarity(set_a, set_b):
    """Calculates intersection over union for token groups rapidly."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / len(union)

def cosine_similarity(count_a, count_b):
    """Computes vector distance using precalculated word counts."""
    if not count_a or not count_b:
        return 0.0
    all_words = set(count_a) | set(count_b)
    dot_product = sum(count_a.get(w, 0) * count_b.get(w, 0) for w in all_words)
    mag_a = sum(v * v for v in count_a.values())
    mag_b = sum(v * v for v in count_b.values())
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product / (math.sqrt(mag_a) * math.sqrt(mag_b))

def fast_similarity(query, target, query_words, target_words, query_counts, target_counts):
    """Calculates quick composite matching score without string re-cleaning."""
    word_score = Jaccard_similarity(query_words, target_words)
    cosine_score = cosine_similarity(query_counts, target_counts)
    
    # Cheap character match fallback only if token metrics overlap slightly
    char_score = 0.0
    if word_score > 0.1:
        char_score = SequenceMatcher(None, query, target).ratio()
        
    return (char_score * 0.20) + (word_score * 0.40) + (cosine_score * 0.40)

# ============================================================
# OPTIMIZED LOCAL ENGINE CLASS
# ============================================================

class FastLocalBrain:
    def __init__(self):
        self.data = {
            "version": VERSION,
            "inputs": {},
            "stats": {"learned": 0, "conversations": 0}
        }
        # Inverted index: maps single words to lists of input_keys that contain them
        self.inverted_index = defaultdict(set)
        # Runtime cache for parsed text structures to avoid processing inside hot loops
        self.parsed_cache = {}
        self.load()

    def load(self):
        """Loads knowledge graph and compiles index maps."""
        if os.path.exists(BRAIN_FILE):
            try:
                with open(BRAIN_FILE, "r", encoding="utf-8") as file:
                    self.data.update(json.load(file))
                # Rebuild inverted index and cache parameters
                for key in self.data["inputs"].keys():
                    words = word_set(key)
                    self.parsed_cache[key] = (words, word_count(key))
                    for word in words:
                        self.inverted_index[word].add(key)
            except Exception as e:
                print(f"[Engine] Core boot error, generating fresh schema: {e}")

    def save(self):
        """Saves knowledge down to local storage disk securely."""
        try:
            with open(BRAIN_FILE, "w", encoding="utf-8") as file:
                json.dump(self.data, file, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[Engine] Storage pipeline failed: {e}")

    def learn_interaction(self, user_input, system_response):
        """Learns bi-directionally from both sides of the dialogue loop."""
        in_key = clean_text(user_input)
        out_key = clean_text(system_response)
        
        if in_key and system_response:
            self._store_node(in_key, system_response)
        if out_key and user_input:
            self._store_node(out_key, user_input)
            
        self._prune_memory()
        self.save()

    def _store_node(self, lookup_key, reply_content):
        """Internal method to map response variations under an exact match string."""
        if lookup_key not in self.data["inputs"]:
            self.data["inputs"][lookup_key] = {"responses": [], "weights": {}, "seen": 0}
            self.data["stats"]["learned"] += 1
            
            # Index structural features dynamically
            words = word_set(lookup_key)
            self.parsed_cache[lookup_key] = (words, word_count(lookup_key))
            for word in words:
                self.inverted_index[word].add(lookup_key)
                
        node = self.data["inputs"][lookup_key]
        node["seen"] += 1
        
        if reply_content not in node["responses"]:
            node["responses"].append(reply_content)
            node["weights"][reply_content] = 1
        else:
            node["weights"][reply_content] = node["weights"].get(reply_content, 1) + 1
            
        if len(node["responses"]) > MAX_RESPONSES_PER_INPUT:
            discarded = node["responses"].pop(0)
            node["weights"].pop(discarded, None)

    def _prune_memory(self):
        """Prevents standard file memory leakage by purging oldest inputs."""
        if len(self.data["inputs"]) > MAX_MEMORY_KEYS:
            # Simple chronological/usage cull by stripping the first few keys
            overflow = len(self.data["inputs"]) - MAX_MEMORY_KEYS
            keys_to_remove = list(self.data["inputs"].keys())[:overflow]
            for k in keys_to_remove:
                self.data["inputs"].pop(k, None)
                self.parsed_cache.pop(k, None)
                for word in list(self.inverted_index.keys()):
                    self.inverted_index[word].discard(k)

    def search_brain(self, query):
        """Fast sub-linear query execution matching through inverted word maps."""
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return None, 0.0
            
        # Complete exact matching shortcut bypasses calculations entirely
        if cleaned_query in self.data["inputs"]:
            return self._extract_weighted_reply(self.data["inputs"][cleaned_query]), 1.0
            
        query_words = word_set(cleaned_query)
        query_counts = word_count(cleaned_query)
        
        # Pull only relevant candidate inputs containing shared keyword components
        candidates = set()
        for word in query_words:
            if word in self.inverted_index:
                candidates.update(self.inverted_index[word])
                
        if not candidates:
            return None, 0.0
            
        best_score = 0.0
        best_reply = None
        
        for candidate in candidates:
            c_words, c_counts = self.parsed_cache[candidate]
            score = fast_similarity(cleaned_query, candidate, query_words, c_words, query_counts, c_counts)
            
            if score > best_score and score >= MIN_SIMILARITY:
                best_score = score
                best_reply = self._extract_weighted_reply(self.data["inputs"][candidate])
                
        return best_reply, best_score

    def _extract_weighted_reply(self, node):
        """Selects a response from a node using selection frequency weighting."""
        resps = node["responses"]
        if not resps:
            return None
        weights = [node["weights"].get(r, 1) for r in resps]
        return random.choices(resps, weights=weights, k=1)[0]

# ============================================================
# INTERACTIVE TERMINAL LOOP
# ============================================================

def run_chat_system():
    engine = FastLocalBrain()
    print("=" * 60)
    print(f" FAST ENGINE ONLINE (v{VERSION}) - FULLY LOCAL")
    print(" Conversing with this engine trains its responses recursively.")
    print(" Type 'exit' or 'quit' to save data state and shut down.")
    print("=" * 60)
    
    last_system_output = None
    
    while True:
        try:
            user_raw = input("\nYou: ").strip()
            if not user_raw:
                continue
                
            if user_raw.lower() in ["exit", "quit"]:
                print("[Engine] Writing memory configurations safely... Goodbye.")
                engine.save()
                break
                
            # Perform accelerated index matching evaluation
            matched_response, match_score = engine.search_brain(user_raw)
            
            # Recursive learning block
            if last_system_output:
                engine.learn_interaction(last_system_output, user_raw)
                
            if matched_response and match_score >= GOOD_SIMILARITY:
                print(f"Bot: {matched_response} (Score: {match_score:.2f})")
                last_system_output = matched_response
            else:
                # Default safety fallback state if structural metrics fall below threshold
                fallbacks = [
                    "Interesting point! Could you elaborate on that?",
                    "I am tracking your query, tell me more about it.",
                    "Fascinating. What leads you to mention that?",
                    "I see! How does that connect to what we're doing?",
                    "That's outside my current index records. Teach me what to say!"
                ]
                chosen_fallback = random.choice(fallbacks)
                print(f"Bot: {chosen_fallback} [Unindexed Query]")
                
                # Treat unindexed responses as an interaction learning source point
                engine.learn_interaction(user_raw, chosen_fallback)
                last_system_output = chosen_fallback
                
        except KeyboardInterrupt:
            print("\n[Engine] Process interrupted. Saving structural state indexes safely...")
            engine.save()
            break

if __name__ == "__main__":
    run_chat_system()
