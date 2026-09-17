"""
01_basic_llm.py - Basic LLM Application (OLLAMA - Local & FREE)
===============================================================

🎯 HOW TO USE THIS FILE:
========================
Run locally with Ollama:
1. Start Ollama server: ollama serve
2. Keep that terminal open
3. In new terminal: python 01_basic_llm.py

WHAT YOU'LL LEARN:
- How to call local LLM via Ollama API
- Basic prompt engineering
- Handling responses
- Error handling basics
- Multi-turn conversation with history

MUST REMEMBER:
✓ No API key needed (runs locally)
✓ Messages have a specific format: role + content
✓ Always handle Ollama connection errors
✓ Keep prompts clear and specific
✓ Make sure ollama serve is running

KEY CONCEPTS:
- Client initialization (no auth needed)
- Message format (role: "user" or "assistant")
- Temperature: controls randomness (0=deterministic, 1=creative)
- Max tokens: limits response length
"""

from ollama_base import OllamaClient


class BasicLLMApp:
    """Simple LLM chatbot wrapper for Ollama"""

    def __init__(self, model: str = "mistral"):
        """Initialize the LLM client (runs locally via Ollama)"""
        self.client = OllamaClient(model=model)
        self.model = model
        self.conversation_history = []

    def ask(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Send a prompt and get a response

        Args:
            prompt: Your question/instruction
            temperature: 0.0=predictable, 1.0=creative
        """
        # MUST REMEMBER: Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": prompt
        })

        try:
            response = self.client.chat(
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=1024
            )

            assistant_message = response

            # MUST REMEMBER: Add assistant response for multi-turn
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

        except ConnectionError as e:
            print(f"❌ Ollama Connection Error: {e}")
            self.conversation_history.pop()
            raise
        except Exception as e:
            print(f"❌ Error: {e}")
            self.conversation_history.pop()
            raise

    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []


def main():
    print("=" * 60)
    print("01: BASIC LLM APPLICATION")
    print("=" * 60)

    app = BasicLLMApp()

    # Example 1
    print("\n📝 Example 1: Simple Question")
    print("-" * 40)
    response = app.ask("What is Python and why is it used for AI?")
    print(f"Response: {response[:200]}...")

    # Example 2
    print("\n📝 Example 2: Multi-turn (uses history)")
    print("-" * 40)
    response = app.ask("Can you give me a concrete example?")
    print(f"Response: {response[:200]}...")

    # Example 3
    print("\n📝 Example 3: Creative Mode (temperature=0.9)")
    print("-" * 40)
    app.reset_conversation()
    response = app.ask(
        "Write a short poem about AI",
        temperature=0.9
    )
    print(f"Response:\n{response}")

    # Example 4
    print("\n📝 Example 4: Analytical Mode (temperature=0.2)")
    print("-" * 40)
    app.reset_conversation()
    response = app.ask(
        "What are 3 differences between ML and Deep Learning?",
        temperature=0.2
    )
    print(f"Response:\n{response}")

    print("\n" + "=" * 60)
    print("✅ MUST REMEMBER:")
    print("=" * 60)
    print("""
1. Temperature: 0-0.3 (analytical), 0.4-0.7 (balanced), 0.8-1.0 (creative)
2. History: Maintain conversation_history for multi-turn
3. Error: Always wrap API calls in try-except
4. Messages: Keep role (user/assistant) + content structure
5. Reset: Call reset_conversation() when starting new topic
    """)


if __name__ == "__main__":
    main()
