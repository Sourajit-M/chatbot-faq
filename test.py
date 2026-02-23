from src.llm.gemini_client import GeminiClient

client = GeminiClient()
print(client.generate_response("Explain RAG in one sentence"))