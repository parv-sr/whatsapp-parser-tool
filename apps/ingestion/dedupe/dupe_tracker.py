class DupeTracker:
    def __init__(self):
        self.total_candidates = 0
        self.in_chat = 0
        self.in_batch = 0
        self.in_db = 0
        self.inserted = 0
        self.by_hash = {}

    def add_candidate(self):
        self.total_candidates += 1

    def add_in_chat(self, text: str):
        self.in_chat += 1
        short = hash(text) % 100000
        self.by_hash[short] = self.by_hash.get(short, 0) + 1

    def add_in_batch(self, listing_key: str):
        self.in_batch += 1
        short = hash(listing_key) % 100000
        self.by_hash[short] = self.by_hash.get(short, 0) + 1

    def add_in_db(self, listing_key: str):
        self.in_db += 1
        short = hash(listing_key) % 100000
        self.by_hash[short] = self.by_hash.get(short, 0) + 1

    def add_inserted(self, count: int = 1):
        self.inserted += count

    def as_dict(self):
        return {
            "total_candidates": self.total_candidates,
            "in_chat": self.in_chat,
            "in_batch": self.in_batch,
            "in_db": self.in_db,
            "inserted": self.inserted,
            "by_hash": dict(self.by_hash),
        }

    @classmethod
    def from_dict(cls, payload):
        tracker = cls()
        if not payload:
            return tracker
        tracker.total_candidates = int(payload.get("total_candidates", 0) or 0)
        tracker.in_chat = int(payload.get("in_chat", 0) or 0)
        tracker.in_batch = int(payload.get("in_batch", 0) or 0)
        tracker.in_db = int(payload.get("in_db", 0) or 0)
        tracker.inserted = int(payload.get("inserted", 0) or 0)
        tracker.by_hash = dict(payload.get("by_hash", {}) or {})
        return tracker

    def merge(self, other):
        self.total_candidates += other.total_candidates
        self.in_chat += other.in_chat
        self.in_batch += other.in_batch
        self.in_db += other.in_db
        self.inserted += other.inserted
        for key, value in other.by_hash.items():
            self.by_hash[key] = self.by_hash.get(key, 0) + value

    def summary(self):
        lines = [
            "\n===================",
            "DUPE SUMMARY",
            "===================",
            f"• Candidates seen:     {self.total_candidates}",
            f"• In-chat duplicates: {self.in_chat}",
            f"• In-batch duplicates: {self.in_batch}",
            f"• In-DB duplicates:  {self.in_db}",
            f"• Inserted listings:  {self.inserted}",
            "",
            "Top repeated fingerprints:",
        ]

        top = sorted(self.by_hash.items(), key=lambda x: -x[1])[:5]
        for k, v in top:
            lines.append(f"    - hash {k}: {v} occurrences")

        lines.append("===================\n")
        return "\n".join(lines)