# Medscape Agentic AI System

**✔ Objective**
<ul>
<li>Multi-agent architecture</li>

<li>Retriever Agent (knowledge base search)</li>

<li>Analytics Agent (ROI, comparisons, stability, recommendations)</li>

<li>Orchestrator / Controller Agent (LLM-driven reasoning loop)</li>
</ul>
- ✔ Natural language → Structured recommendation 
- ✔ Tool-calling workflow 
- ✔ Complete architecture + design rationale
- ✔ Clear path to production scalability 

---
## 🔑 Core Pattern: The Agentic Loop

The orchestrator follows a simple loop:

1. **Send** query + available tools to LLM
2. **If** LLM requests tools → execute them → loop back to step 1
3. **Else** → return LLM's final answer

---

## 📋 Thought Process

**Step 1** 
The core challenge in this assignment is to design a system where the LLM acts as the decision-maker, not just a text generator. Instead of manually choosing which functions to run, the LLM must determine:
<ul>
<li>What information is required, </li>

<li>Which tools should be invoked,</li>

<li>In what order those tools should be used, and</li>

<li>When the final answer is ready.</li>
</ul>

This mirrors how a human analyst approaches questions: understanding the request, identifying relevant data sources, running analysis, and forming a recommendation.

**Step 2:  LLM Orchestration**
To model this behavior, I designed the system to function as a human analyst with assistants. The LLM performs the reasoning and planning steps:

1. Interpret the user request

2. Decide what data is needed

3. Identify which tools should be used to retrieve or compute that data

4. Execute those tools through the orchestrator

5. Combine the results into a structured, actionable recommendation

Since LLMs cannot directly access spreadsheets or databases, tools act as the hands of the system while the LLM is the “brain.” Each tool performs one well-defined task, and the LLM coordinates them through an iterative reasoning loop.

**Step 3: Tool Calling**
I decided to use the same pattern as OpenAI does when interacting with external systems. The method is to tell LLM about what it needs to know about the given tools and then give it the question. In this process, LLM will read the question, consider what it knows (for example, available tools) and responds with what tool it should call and what parameters are needed. After information is executed from the tools the results will be sent to the LLM. In this step, LLM can make a decision on to whether ask for more tools or consider it as a final response and send it to the client.

**The interface**
```python
class LLMClient:
    def chat(self, messages, tools) -> Response:
        """
        Send messages to LLM, get back either:
        - tool_calls: LLM wants to call tools
        - content: LLM's final answer
        """
        pass
```

**What LLM receives**

```python
messages = [
    {"role": "system", "content": "You are a campaign analyst..."},
    {"role": "user", "content": "Compare Cardiology vs Oncology"}
]

tools = TOOL_DEFINITIONS  # The tools it can use
```
**What LLM returns**

```python
# Option A: LLM wants to call tools
response.tool_calls = [
    ToolCall(name="tool_compare_brand_areas", 
             arguments={"brand_areas": ["Cardiology", "Oncology"]})
]

# Option B: LLM is done, here's the answer
response.content = "Based on my analysis, Cardiology outperforms..."
```

**Step 4: Architecture**
To support this design, I structured the system around four core components:

<ul>
<li>Tools: Python functions that execute specific tasks such as computing ROI, comparing brand areas, or searching the knowledge base. Each tool adheres to a single responsibility principle.</li>

<li>Tool Definitions: JSON schemas describing each tool in a format the LLM can understand. These definitions allow the LLM to reason about available capabilities.</li>

<li>LLM Client: A clean abstraction that allows the system to run with either a Mock LLM (for offline testing) or a real LLM API (for production).</li>

<li>Orchestrator: The main loop that manages tool invocation, collects outputs, and passes results back to the LLM until a final structured recommendation is produced.</li>
</ul>
This separation of responsibilities makes the system transparent, debuggable, and easily extensible. 


## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
│      "Compare Cardiology vs Oncology performance"               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM (or Mock LLM)                          │
│                                                                 │
│   Input: Query + Available Tools (in OpenAI function format)    │
│   Output: "I want to call tool_compare_brand_areas with..."     │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  For Demo: MockLLMClient simulates this decision        │   │
│   │  For Prod: OpenAIClient makes real LLM calls            │   │
│   │                                                         │   │
│   │  THE ARCHITECTURE IS IDENTICAL - ONLY SOURCE CHANGES    │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TOOL EXECUTION LAYER                         │
│                                                                 │
│   tool_compare_brand_areas()     tool_search_knowledge_base()   │
│   tool_get_campaign_metrics()    tool_analyze_stability()       │
│   tool_get_reallocation_suggestion()                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM (or Mock LLM)                          │
│                                                                 │
│   Input: Tool results                                           │
│   Output: Final synthesized response                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      FINAL RESPONSE
```

## 🎯 Design Decisions

| Decision | Choice | Why | Trade-off |
|----------|--------|-----|-----------|
| Orchestration | Custom (not LangGraph) | Transparency, easier to explain | Less battle-tested |
| LLM Client | Mock + Real abstraction | Demo works without API key | Mock logic is simplified |
| Tool granularity | 5 focused tools | Single responsibility, composable | More tools to maintain |
| Data format | JSON responses | LLM-friendly, structured | Verbose for large datasets |

---

## 🚀 Scaling to Production

**Phase 1: Better Retrieval**
- Replace keyword search with vector embeddings (e.g., ChromaDB, Pinecone)
- Semantic search for a knowledge base

**Phase 2: Infrastructure**
- Add Redis caching for frequent queries
- PostgreSQL for campaign data (replace CSV)
- FastAPI wrapper for REST API
- Kubernetes for horizontal scaling

**Phase 3: Observability**
- Logging for all LLM calls and tool executions
- Metrics dashboard for response times, tool usage
- A/B testing for prompt variations
---

## 🚀 Running the Demo

### Install Dependencies
```bash
pip install pandas pydantic
```

### Option 1: Menu Selection
```bash
python main.py
```

### Option 2: Demo Mode (sample queries)
```bash
python main.py --demo  
```

### Option 3: Interactive Mode (ask your own questions)
```bash
python main.py --interactive 
```
`exit` to quit.

### With OpenAI (Real LLM)
```python
from main import MedscapeOrchestrator, OpenAIClient

client = OpenAIClient(api_key="key")
orchestrator = MedscapeOrchestrator(llm=client)
result = orchestrator.process("Compare Cardiology vs Oncology")
```
---

## 📁 Project Structure

```
medscape_final/
├── main.py              # Entry point + CLI
├── orchestrator.py      # Orchestrator class
├── clients.py           # LLM clients - Mock + OpenAI
├── tools.py             # Tool functions + definitions
├── README.md
└── data/
    ├── campaign_performance.csv
    ├── kb_documents.jsonl
    └── sample_queries.txt
```

---

## 🔧 The Tools (Agents)

| Tool | Role | Description |
|------|------|-------------|
| `tool_get_campaign_metrics` | Analytics Agent | Get ROI metrics by dimension |
| `tool_compare_brand_areas` | Analytics Agent | Compare brand areas |
| `tool_analyze_stability` | Analytics Agent | Analyze ROI stability |
| `tool_search_knowledge_base` | Retriever Agent | Search KB for guidance |
| `tool_get_reallocation_suggestion` | Analytics Agent | Get reallocation advice |

---

## ⚡ Quick Test

```python
from main import MedscapeOrchestrator, MockLLMClient

# Create orchestrator
orchestrator = MedscapeOrchestrator(llm=MockLLMClient())

# Process query
result = orchestrator.process(
    "Compare Cardiology vs Oncology in 2025Q2",
    verbose=True
)

print(result)
```

Output:
```
🔄 Iteration 1
   LLM decided to call 1 tool(s):
   → tool_compare_brand_areas({'brand_areas': ['Cardiology', 'Oncology']...})
   ✓ {"Cardiology": {"overall_roi": 1.586...}, "Oncology": {...}}

🔄 Iteration 2
   ✅ LLM generated final response

## Analysis Results
**Cardiology**: ROI = 1.586, Best tactic = HCP_Newsletter
**Oncology**: ROI = 1.568, Best tactic = HCP_Newsletter
```
