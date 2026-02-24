from typing import List, Dict
from src.llm.gemini_client import GeminiClient


class RAGPipeline:
    def __init__(self):
        self.llm = GeminiClient()

    def build_prompt(self, query: str, documents: List[Dict]) -> str:
        if not documents:
            return (
                "The user asked a question, but no relevant context was found.\n\n"
                "Answer: I don't know based on the provided documents."
            )

        context_blocks = []

        for i, doc in enumerate(documents):
            context_blocks.append(
                f"[Source {i+1}]\n{doc['content']}"
            )

        context_text = "\n\n".join(context_blocks)

        prompt = f"""
You are an AI assistant answering questions using ONLY the provided context.

Rules:
- Answer strictly from the context below
- You may infer information if it is clearly implied across multiple sources
- Do NOT add any external or general knowledge
- If the answer is truly missing, say exactly:
  "I don't know based on the provided documents."
- Be concise, factual, and clear

Context:
{context_text}

Question:
{query}

Answer:
"""
        return prompt.strip()

    def generate_answer(self, query: str, documents: List[Dict]) -> str:
        prompt = self.build_prompt(query, documents)
        return self.llm.generate_response(prompt)