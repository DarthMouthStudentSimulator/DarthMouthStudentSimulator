import json

class EpisodicMemory:
    def __init__(self, uid):
        self.uid_id = uid
        self.filename = f"{self.uid_id}_emotion_status_history.jsonl"
        self.memories = []
        self.load_memories()
    
    def load_memories(self): 
        try:
            with open(self.filename, 'r') as f:
                for line in f:
                    self.memories.append(json.loads(line))
        except FileNotFoundError:
            self.memories = []
    def retrieve_recent(self, week: int, day: int, max_entries: int = 3):
        def is_before(memory_week, memory_day):
            return (memory_week < week) or (memory_week == week and memory_day < day)

        filtered = [
            m for m in self.memories
            if is_before(m.get("week", -1), m.get("day", -1))
        ]
        return filtered[-max_entries:]