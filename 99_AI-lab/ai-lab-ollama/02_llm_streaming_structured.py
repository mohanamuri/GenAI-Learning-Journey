"""
02_llm_streaming_structured.py - Structured Output with Ollama
==============================================================
MUST REMEMBER:
✓ Ollama returns plain text (no native streaming)
✓ Use prompts that generate JSON-parseable output
✓ Always validate parsed JSON
✓ Handle connection errors gracefully

KEY CONCEPTS:
- Request structured output via prompt engineering
- Parse JSON responses from text
- Error handling for JSON parsing
- Works with local Ollama models
"""

from ollama_base import OllamaClient
import json
from pydantic import BaseModel, Field


class PersonInfo(BaseModel):
    """Structured output for person extraction"""
    name: str = Field(description="Person's full name")
    age: int = Field(description="Age in years")
    profession: str = Field(description="Job title")
    skills: list[str] = Field(description="List of key skills")


class StreamingLLM:
    """LLM with structured output using Ollama"""

    def __init__(self, model: str = "mistral"):
        self.client = OllamaClient(model=model)
        self.model = model

    def get_response(self, prompt: str):
        """Get response from Ollama (simpler than streaming)"""
        print("\n🔄 Getting response: ", end="")
        try:
            response = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            print("Done!")
            return response
        except ConnectionError as e:
            print(f"\n❌ Connection Error: {e}")
            raise
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise

    def get_structured_output(self, prompt: str, response_type):
        """Get response in JSON format via prompt engineering"""

        # Add JSON instruction to prompt
        json_prompt = f"""{prompt}

Please respond with ONLY valid JSON in this format:
{json.dumps(response_type.model_json_schema(), indent=2)}

Response (JSON only):"""

        try:
            response_text = self.client.chat(
                messages=[{"role": "user", "content": json_prompt}],
                max_tokens=1024
            )

            # Try to extract JSON from response
            parsed = json.loads(response_text)
            return parsed

        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"   Response was: {response_text[:100]}...")
            raise


def main():
    print("=" * 60)
    print("02: LLM STRUCTURED OUTPUT (Ollama)")
    print("=" * 60)

    llm = StreamingLLM()

    # Example 1: Basic response
    print("\n📝 Example 1: Simple Response")
    print("-" * 40)
    response = llm.get_response("Explain quantum computing in 50 words")
    print(f"Response: {response[:200]}...")
    print(f"✅ Received {len(response)} chars")

    # Example 2: Structured output
    print("\n\n📝 Example 2: Structured JSON Output")
    print("-" * 40)
    try:
        result = llm.get_structured_output(
            prompt="Extract: Albert Einstein was a physicist born 1879, expert in relativity",
            response_type=PersonInfo
        )
        print(f"✅ Parsed JSON successfully")
        print(f"   Name: {result.get('name', 'N/A')}")
        print(f"   Age: {result.get('age', 'N/A')}")
        print(f"   Profession: {result.get('profession', 'N/A')}")
    except json.JSONDecodeError:
        print("ℹ️  JSON parsing failed - this is normal with local models")
        print("   Ollama models may not format JSON perfectly")

    print("\n" + "=" * 60)
    print("✅ MUST REMEMBER:")
    print("=" * 60)
    print("""
1. OLLAMA: Runs locally, no streaming support
2. STRUCTURED: Use prompt engineering to request JSON
3. VALIDATION: Always parse JSON with error handling
4. FALLBACK: Local models may not follow JSON format perfectly
5. RETRY: If JSON parsing fails, try with better prompt
    """)


if __name__ == "__main__":
    main()
