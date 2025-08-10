import json
from collections import deque
from .utils import get_timestamp, ensure_directory_exists

class ShortTermMemory:
    def __init__(self, file_path, max_capacity=10):
        self.max_capacity = max_capacity
        self.file_path = file_path
        ensure_directory_exists(self.file_path)
        self.memory = deque()  # Remove maxlen to prevent auto-eviction
        self.load()

    def add_qa_pair(self, qa_pair):
        # Ensure timestamp exists, add if not
        if 'timestamp' not in qa_pair or not qa_pair['timestamp']:
            qa_pair["timestamp"] = get_timestamp()
        
        # Add without auto-eviction - spillover will handle capacity in background
        self.memory.append(qa_pair)
        print(f"ShortTermMemory: Added QA. User: {qa_pair.get('user_input','')[:30]}...")
        self.save()

    def get_all(self):
        return list(self.memory)

    def is_full(self):
        return len(self.memory) >= self.max_capacity # Use >= to be safe

    def pop_oldest(self):
        if self.memory:
            msg = self.memory.popleft()
            print("ShortTermMemory: Evicted oldest QA pair.")
            self.save()
            return msg
        return None

    def save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(list(self.memory), f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error saving ShortTermMemory to {self.file_path}: {e}")

    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure items are loaded correctly, especially if file was empty or malformed
                if isinstance(data, list):
                    self.memory = deque(data)  # Load without maxlen
                else:
                    self.memory = deque()
            print(f"ShortTermMemory: Loaded from {self.file_path}.")
        except FileNotFoundError:
            self.memory = deque()
            print(f"ShortTermMemory: No history file found at {self.file_path}. Initializing new memory.")
        except json.JSONDecodeError:
            self.memory = deque()
            print(f"ShortTermMemory: Error decoding JSON from {self.file_path}. Initializing new memory.")
        except Exception as e:
            self.memory = deque()
            print(f"ShortTermMemory: An unexpected error occurred during load from {self.file_path}: {e}. Initializing new memory.") 