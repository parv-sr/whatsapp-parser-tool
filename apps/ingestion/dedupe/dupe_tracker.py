class DupeTracker:
    def __init__(self):
        self.in_chat = 0
        self.in_db = 0
        self.by_hash = {}

    def add_in_chat(self, text: str):
        self.in_chat += 1
        short = hash(text) % 100000
        self.by_hash[short] = self.by_hash.get(short, 0) + 1

    def add_in_db(self, listing_key: str):
        self.in_db += 1
        short = hash(listing_key) % 100000
        self.by_hash[short] = self.by_hash.get(short, 0) + 1

    def summary(self):
        lines = [
            "\n===================",
            "DUPE SUMMARY",
            "===================",
            f"• In-chat duplicates: {self.in_chat}",
            f"• In-DB duplicates:  {self.in_db}",
            "",
            "Top repeated fingerprints:",
        ]

        top = sorted(self.by_hash.items(), key=lambda x: -x[1])[:5]
        for k, v in top:
            lines.append(f"    - hash {k}: {v} occurrences")

        lines.append("===================\n")
        return "\n".join(lines)
        



  