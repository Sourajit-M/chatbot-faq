from typing import List, Dict


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 700,
    overlap: int = 150
) -> List[Dict]:
    chunked_docs = []

    for doc in documents:
        text = doc["content"]
        metadata = doc["metadata"]

        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk_text = text[start:end]

            chunked_docs.append(
                {
                    "content": chunk_text,
                    "metadata": metadata
                }
            )

            start += chunk_size - overlap

    return chunked_docs