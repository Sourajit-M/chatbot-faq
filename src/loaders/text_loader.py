import json
from pathlib import Path
from typing import List, Dict


class TextFAQLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> List[Dict]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} not found")

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        for idx, item in enumerate(data):
            content = f"Question: {item['question']}\nAnswer: {item['answer']}"
            documents.append(
                {
                    "content": content,
                    "metadata": {
                        "source": "faq",
                        "id": idx
                    }
                }
            )

        return documents