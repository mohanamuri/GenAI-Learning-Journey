"""
14_supervisor_agent.py - Supervisor Agent (Coordinator Pattern)
===============================================================
MUST REMEMBER:
✓ Supervisor: coordinates multiple agents
✓ Task delegation: assign work to specialists
✓ Result aggregation: combine responses
✓ Orchestration: manage workflow

KEY: Coordinator pattern, task delegation, result aggregation
"""

from ollama_base import OllamaClient
import json


class SpecialistAgent:
    """Specialist agent for specific domain (Ollama)"""

    def __init__(self, specialty: str, description: str):
        self.specialty = specialty
        self.description = description
        self.client = OllamaClient(model="mistral")

    def process(self, task: str) -> str:
        """Process task in specialty"""
        prompt = f"You are a {self.specialty} specialist. {self.description}\n\nTask: {task}"

        try:
            response = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response

        except Exception as e:
            return f"Error: {e}"


class SupervisorAgent:
    """Supervisor that coordinates specialist agents (Ollama)"""

    def __init__(self):
        self.client = OllamaClient(model="mistral")

        # MUST REMEMBER: Create specialist agents
        self.specialists = {
            "data_scientist": SpecialistAgent(
                "Data Scientist",
                "Focus on data analysis, statistics, and ML models"
            ),
            "software_engineer": SpecialistAgent(
                "Software Engineer",
                "Focus on code quality, architecture, and best practices"
            ),
            "domain_expert": SpecialistAgent(
                "Domain Expert",
                "Focus on business context and real-world applications"
            )
        }

    def _parse_delegation(self, delegation_text: str) -> dict:
        """Parse LLM delegation to experts"""
        # Simulated: in real case, parse structured output
        result = {}

        if "data" in delegation_text.lower() or "analysis" in delegation_text.lower():
            result["data_scientist"] = delegation_text

        if "code" in delegation_text.lower() or "engineer" in delegation_text.lower():
            result["software_engineer"] = delegation_text

        if "business" in delegation_text.lower() or "domain" in delegation_text.lower():
            result["domain_expert"] = delegation_text

        return result if result else {"data_scientist": delegation_text}

    def supervise(self, task: str) -> str:
        """Supervise and delegate to specialists"""
        print(f"\n🎯 Task: {task}")
        print("=" * 60)

        # Step 1: Analyze task and create delegation plan
        print("\n📋 Step 1: Creating Delegation Plan")
        delegation_prompt = f"""
Analyze this task and decide which specialists to involve: {task}

List specialists needed: data_scientist, software_engineer, domain_expert
Return simple list of needed specialists.
        """

        try:
            response = self.client.chat(
                messages=[{"role": "user", "content": delegation_prompt}],
                max_tokens=300
            )

            delegation_plan = response
            specialists_needed = self._parse_delegation(delegation_plan)

            print(f"📍 Specialists needed: {list(specialists_needed.keys())}")

        except Exception as e:
            print(f"⚠️ Delegation planning failed: {e}")
            specialists_needed = {"data_scientist": task}

        # Step 2: Get specialist responses
        print("\n🔄 Step 2: Getting Specialist Responses")
        specialist_responses = {}

        for specialist_name, subtask in specialists_needed.items():
            if specialist_name in self.specialists:
                print(f"   → {specialist_name}: processing...")
                response = self.specialists[specialist_name].process(subtask)
                specialist_responses[specialist_name] = response[:200]

        # Step 3: Aggregate results
        print("\n📊 Step 3: Aggregating Results")
        aggregation_prompt = f"""
Synthesize these specialist responses into a coherent answer:

{json.dumps(specialist_responses, indent=2)}

Original task: {task}

Provide final synthesized response.
        """

        try:
            response = self.client.chat(
                messages=[{"role": "user", "content": aggregation_prompt}],
                max_tokens=500
            )

            final_response = response
            print(f"✅ Final Response: {final_response[:100]}...")
            return final_response

        except Exception as e:
            return f"Error aggregating results: {e}"


def main():


if __name__ == "__main__":
    main()
