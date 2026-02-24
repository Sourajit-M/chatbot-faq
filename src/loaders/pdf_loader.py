from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader


class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> List[Dict]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"{self.file_path} not found")

        reader = PdfReader(self.file_path)
        documents = []

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()

            if not text or not text.strip():
                continue

            documents.append(
                {
                    "content": text.strip(),
                    "metadata": {
                        "source": "pdf",
                        "file_name": self.file_path.name,
                        "page": page_number + 1
                    }
                }
            )

        return documents