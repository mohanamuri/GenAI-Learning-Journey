# Framework: CrewAI (Role-based Multi-Agent)

## What is it?
A Python library for building multi-agent systems where each agent has a role, goal, and backstory.
Agents form a "crew" and collaborate to complete tasks — each agent uses its own LLM calls.
Built on top of LangChain. Released 2024.

## What existed before?
- **Single-agent loops**: one LLM doing everything → context overload, no specialization
- **LangChain Chains**: fixed steps, no agent-to-agent collaboration
- **AutoGen**: Microsoft's multi-agent framework (conversation-based, more flexible but verbose)

## Why CrewAI?
- Role definition forces specialization: Researcher, Writer, Reviewer each have a clear scope
- Sequential/parallel/hierarchical process modes built-in
- Task outputs pass automatically between agents
- Simpler API than AutoGen for structured pipelines

## Core code pattern

```python
from crewai import Agent, Task, Crew, Process

# Define agents with roles
researcher = Agent(
    role="Research Analyst",
    goal="Find accurate, up-to-date information on the topic",
    backstory="Expert at synthesizing information from multiple sources",
    verbose=True,
    llm="ollama/llama3.2:3b"   # supports Ollama
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, concise articles from research findings",
    backstory="Turns complex research into readable content",
    llm="ollama/llama3.2:3b"
)

# Define tasks
research_task = Task(
    description="Research {topic} and return structured findings",
    expected_output="A list of 5 key facts with sources",
    agent=researcher
)

write_task = Task(
    description="Write a 200-word article using the research findings",
    expected_output="A polished article with title and body",
    agent=writer,
    context=[research_task]   # gets researcher's output as input
)

# Form crew and run
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential   # or Process.hierarchical
)

result = crew.kickoff(inputs={"topic": "TechCorp pricing"})
print(result)
```

## Process types

| Process | Description | Use when |
|---------|-------------|----------|
| `sequential` | Tasks run in order, each gets prev output | Linear pipelines |
| `hierarchical` | Manager agent delegates to workers | Dynamic task assignment |
| `parallel` | Tasks run concurrently | Independent tasks |

## When to use vs skip

**Use CrewAI when:**
- You need a quick multi-agent pipeline with minimal boilerplate
- Task dependencies are linear (A → B → C)
- You want built-in role/goal/backstory structure

**Skip CrewAI (use raw agents) when:**
- You need fine-grained control over state transitions
- You need complex conditional branching or loops
- You want to avoid LangChain dependency overhead
- Building a production system with custom retry/circuit breaker logic

## Install
```bash
pip install crewai
pip install crewai[tools]   # for built-in tools (search, scrape, etc.)
```

## Key concepts to remember
- `Agent`: role + goal + backstory + llm
- `Task`: description + expected_output + agent + context (optional)
- `Crew`: agents + tasks + process
- `kickoff(inputs={...})`: runs the crew, returns final task output
- Context: `context=[prior_task]` passes prior task output to current task
