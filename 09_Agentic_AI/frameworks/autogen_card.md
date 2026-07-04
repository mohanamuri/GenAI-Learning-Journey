# Framework: AutoGen (Conversational Multi-Agent)

## What is it?
Microsoft's multi-agent framework where agents communicate through a conversation loop.
Each agent can be an LLM, a human proxy, or a code executor.
The conversation is the coordination mechanism — agents take turns replying.
Released: 2023 (Microsoft Research).

## What existed before?
- **Single LLM**: can't self-correct or run code reliably
- **LangChain agents**: single-agent tool-calling, no agent-to-agent chat
- **CrewAI**: role-based, but sequential by default — AutoGen allows back-and-forth

## Why AutoGen?
- Back-and-forth between agents (not just A → B → C)
- Built-in human proxy: a human can step in at any point
- Code executor: LLM writes Python → executor runs it → result goes back to LLM
- Natural for debate/critique patterns: Writer ↔ Critic ↔ Writer

## Core code pattern

```python
import autogen

config_list = [{
    "model": "llama3.2:3b",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
}]

llm_config = {"config_list": config_list, "temperature": 0.3}

# AssistantAgent: LLM-powered
assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config=llm_config,
    system_message="You are a helpful assistant. Write Python code when needed.",
)

# UserProxyAgent: acts as user, can execute code
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",    # NEVER | ALWAYS | TERMINATE
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,     # set True in production
    },
    max_consecutive_auto_reply=5,
)

# Initiate conversation — agents chat until termination condition
user_proxy.initiate_chat(
    assistant,
    message="Calculate compound interest for $10,000 at 7% for 10 years."
)
```

## Agent types

| Agent | Description | Use when |
|-------|-------------|----------|
| `AssistantAgent` | LLM-powered, can suggest code | Primary reasoning agent |
| `UserProxyAgent` | Executes code, proxies human | Code execution, human-in-loop |
| `GroupChatManager` | Orchestrates multi-agent group chat | 3+ agents debating/collaborating |
| Custom agents | Subclass ConversableAgent | Domain-specific behavior |

## GroupChat pattern (3+ agents)

```python
groupchat = autogen.GroupChat(
    agents=[researcher, writer, critic],
    messages=[],
    max_round=10,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
user_proxy.initiate_chat(manager, message="Write an article on AI agents.")
```

## Termination conditions

```python
# Terminate when agent outputs "TERMINATE" in its reply
user_proxy = autogen.UserProxyAgent(
    ...
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
)
```

## When to use vs skip

**Use AutoGen when:**
- You need back-and-forth critique/revision between agents
- You need code execution in the agent loop
- Human-in-the-loop is required (UserProxyAgent)
- Building a debate or negotiation pattern between agents

**Skip AutoGen (use CrewAI or raw agents) when:**
- Tasks are strictly sequential with no back-and-forth needed
- You want simpler setup without conversation overhead
- You need fine-grained workflow control (DAG, conditions, loops)

## Install
```bash
pip install pyautogen
```

## Key concepts to remember
- `initiate_chat()`: starts the back-and-forth conversation loop
- `human_input_mode="NEVER"`: fully automated, no human intervention
- `max_consecutive_auto_reply`: prevents infinite loops
- `code_execution_config`: enables actual Python execution by UserProxy
- AutoGen = conversation-based; CrewAI = task-pipeline-based
