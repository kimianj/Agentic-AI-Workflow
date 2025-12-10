import json
import os
from typing import List, Dict, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd


# =============================================================================
# SECTION 1: DATA LAYER
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load_campaign_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "campaign_performance.csv"))

def load_knowledge_base() -> List[Dict]:
    docs = []
    with open(os.path.join(DATA_DIR, "kb_documents.jsonl"), 'r') as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


# =============================================================================
# SECTION 2: TOOLS (Functions the LLM can call)
# =============================================================================
# These are the "agents" - specialized functions that do specific tasks

def tool_get_campaign_metrics(
    brand_areas: Optional[List[str]] = None,
    quarters: Optional[List[str]] = None,
    group_by: str = "tactic"
) -> str:
    """
    Analytics Agent: Get campaign performance metrics.
    
    Args:
        brand_areas: Filter by brand area (Cardiology, Oncology, Endocrinology)
        quarters: Filter by quarter (2025Q1, 2025Q2, 2025Q3)
        group_by: Dimension to group by (tactic, brand_area, quarter)
    """
    df = load_campaign_data()
    
    if brand_areas:
        df = df[df['brand_area'].isin(brand_areas)]
    if quarters:
        df = df[df['quarter'].isin(quarters)]
    
    # Aggregate
    agg = df.groupby(group_by).agg({
        'spend': 'sum',
        'revenue': 'sum',
        'conversions': 'sum'
    }).reset_index()
    
    agg['roi'] = (agg['revenue'] / agg['spend']).round(3)
    agg = agg.sort_values('roi', ascending=False)
    
    # Convert to native Python types
    result = []
    for _, row in agg.iterrows():
        result.append({
            group_by: str(row[group_by]),
            'spend': float(row['spend']),
            'revenue': float(row['revenue']),
            'conversions': int(row['conversions']),
            'roi': float(row['roi'])
        })
    
    return json.dumps(result, indent=2)


def tool_compare_brand_areas(
    brand_areas: List[str],
    quarter: Optional[str] = None
) -> str:
    """
    Analytics Agent: Compare two or more brand areas.
    
    Args:
        brand_areas: List of brand areas to compare
        quarter: Optional quarter filter
    """
    df = load_campaign_data()
    
    if quarter:
        df = df[df['quarter'] == quarter]
    df = df[df['brand_area'].isin(brand_areas)]
    
    result = {}
    for area in brand_areas:
        area_df = df[df['brand_area'] == area]
        if len(area_df) == 0:
            continue
        
        # By tactic
        by_tactic = area_df.groupby('tactic').agg({
            'spend': 'sum', 'revenue': 'sum'
        }).reset_index()
        by_tactic['roi'] = by_tactic['revenue'] / by_tactic['spend']
        
        best = by_tactic.loc[by_tactic['roi'].idxmax()]
        worst = by_tactic.loc[by_tactic['roi'].idxmin()]
        
        result[area] = {
            'total_spend': float(area_df['spend'].sum()),
            'total_revenue': float(area_df['revenue'].sum()),
            'overall_roi': round(float(area_df['revenue'].sum() / area_df['spend'].sum()), 3),
            'best_tactic': str(best['tactic']),
            'best_tactic_roi': round(float(best['roi']), 3),
            'worst_tactic': str(worst['tactic']),
            'worst_tactic_roi': round(float(worst['roi']), 3)
        }
    
    return json.dumps(result, indent=2)


def tool_analyze_stability(
    brand_areas: Optional[List[str]] = None
) -> str:
    """
    Analytics Agent: Analyze ROI stability across quarters.
    Lower std = more stable/predictable performance.
    
    Args:
        brand_areas: Optional filter
    """
    df = load_campaign_data()
    
    if brand_areas:
        df = df[df['brand_area'].isin(brand_areas)]
    
    df['roi'] = df['revenue'] / df['spend']
    
    stability = df.groupby('tactic')['roi'].agg(['mean', 'std']).reset_index()
    stability.columns = ['tactic', 'roi_mean', 'roi_std']
    stability = stability.fillna(0).sort_values('roi_std')
    
    result = []
    for _, row in stability.iterrows():
        result.append({
            'tactic': str(row['tactic']),
            'roi_mean': round(float(row['roi_mean']), 3),
            'roi_std': round(float(row['roi_std']), 3),
            'stability_rank': len(result) + 1
        })
    
    return json.dumps(result, indent=2)


def tool_search_knowledge_base(
    keywords: List[str]
) -> str:
    """
    Retriever Agent: Search knowledge base for guidance.
    
    Args:
        keywords: Keywords to search for
    """
    docs = load_knowledge_base()
    results = []
    
    for doc in docs:
        text = (doc['title'] + ' ' + doc['text']).lower()
        score = sum(text.count(kw.lower()) for kw in keywords)
        
        if score > 0:
            results.append({
                'title': doc['title'],
                'text': doc['text'],
                'relevance': score
            })
    
    results.sort(key=lambda x: x['relevance'], reverse=True)
    return json.dumps(results[:3], indent=2)


def tool_get_reallocation_suggestion(
    brand_area: str,
    percent: float = 20.0
) -> str:
    """
    Analytics Agent: Suggest spend reallocation.
    
    Args:
        brand_area: Brand area to analyze
        percent: Percentage to reallocate (default 20%)
    """
    df = load_campaign_data()
    df = df[df['brand_area'] == brand_area]
    
    by_tactic = df.groupby('tactic').agg({
        'spend': 'sum', 'revenue': 'sum'
    }).reset_index()
    by_tactic['roi'] = by_tactic['revenue'] / by_tactic['spend']
    
    best = by_tactic.loc[by_tactic['roi'].idxmax()]
    worst = by_tactic.loc[by_tactic['roi'].idxmin()]
    
    amount = float(worst['spend']) * (percent / 100)
    impact = amount * (float(best['roi']) - float(worst['roi']))
    
    return json.dumps({
        'from_tactic': str(worst['tactic']),
        'to_tactic': str(best['tactic']),
        'amount': round(amount, 2),
        'projected_impact': round(impact, 2),
        'recommendation': f"Shift ${amount:,.0f} ({percent}%) from {worst['tactic']} to {best['tactic']}"
    }, indent=2)


# =============================================================================
# SECTION 3: TOOL DEFINITIONS (OpenAI Function Calling Format)
# =============================================================================
# This is the standard format that LLMs like GPT-4 and Claude understand

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "tool_get_campaign_metrics",
            "description": "Get campaign performance metrics (spend, revenue, ROI) grouped by a dimension",
            "parameters": {
                "type": "object",
                "properties": {
                    "brand_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter: Cardiology, Oncology, or Endocrinology"
                    },
                    "quarters": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "Filter: 2025Q1, 2025Q2, or 2025Q3"
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["tactic", "brand_area", "quarter"],
                        "description": "Dimension to group results by"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_compare_brand_areas",
            "description": "Compare performance metrics between brand areas",
            "parameters": {
                "type": "object",
                "properties": {
                    "brand_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Brand areas to compare"
                    },
                    "quarter": {
                        "type": "string",
                        "description": "Optional quarter filter"
                    }
                },
                "required": ["brand_areas"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_analyze_stability",
            "description": "Analyze ROI stability across quarters. Lower std deviation = more stable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "brand_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional filter by brand area"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_search_knowledge_base",
            "description": "Search knowledge base for tactic guidance and best practices",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords to search"
                    }
                },
                "required": ["keywords"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_get_reallocation_suggestion",
            "description": "Get a spend reallocation recommendation",
            "parameters": {
                "type": "object",
                "properties": {
                    "brand_area": {
                        "type": "string",
                        "description": "Brand area to analyze"
                    },
                    "percent": {
                        "type": "number",
                        "description": "Percentage to reallocate (default 20)"
                    }
                },
                "required": ["brand_area"]
            }
        }
    }
]

# Map names to functions
TOOL_MAP = {
    "tool_get_campaign_metrics": tool_get_campaign_metrics,
    "tool_compare_brand_areas": tool_compare_brand_areas,
    "tool_analyze_stability": tool_analyze_stability,
    "tool_search_knowledge_base": tool_search_knowledge_base,
    "tool_get_reallocation_suggestion": tool_get_reallocation_suggestion
}


# =============================================================================
# SECTION 4: LLM CLIENTS (Abstraction Layer)
# =============================================================================

@dataclass
class ToolCall:
    """Represents a tool call from the LLM"""
    id: str
    name: str
    arguments: Dict


@dataclass 
class LLMResponse:
    """Represents an LLM response"""
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def chat(self, messages: List[Dict], tools: List[Dict]) -> LLMResponse:
        """Send messages to LLM and get response"""
        pass


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM for demonstration purposes.
    
    This simulates how a real LLM would decide which tools to call.
    For a real implementation, swap this with OpenAIClient or AnthropicClient.
    
    The ARCHITECTURE is the same - only the decision-making changes.
    """
    
    def chat(self, messages: List[Dict], tools: List[Dict]) -> LLMResponse:
        # Check if we already have tool results
        has_results = any(m.get("role") == "tool" for m in messages)
        
        if has_results:
            # Generate final response based on tool results
            return LLMResponse(
                content=self._generate_response(messages),
                tool_calls=None
            )
        
        # Analyze query and decide which tools to call
        user_query = next((m["content"] for m in messages if m["role"] == "user"), "")
        tool_calls = self._decide_tools(user_query.lower())
        
        return LLMResponse(content=None, tool_calls=tool_calls)
    
    def _decide_tools(self, query: str) -> List[ToolCall]:
        """
        Simulate LLM deciding which tools to call.
        
        In a real LLM, this happens through neural network inference.
        Here we simulate the decision for demo purposes.
        """
        calls = []
        
        # Comparison queries
        if "compare" in query or "vs" in query:
            areas = []
            if "cardiology" in query: areas.append("Cardiology")
            if "oncology" in query: areas.append("Oncology")
            if "endocrinology" in query: areas.append("Endocrinology")
            
            args = {"brand_areas": areas or ["Cardiology", "Oncology"]}
            
            # Extract quarter
            for q in ["2025q1", "2025q2", "2025q3"]:
                if q in query:
                    args["quarter"] = q.upper()
                    break
            
            calls.append(ToolCall("call_1", "tool_compare_brand_areas", args))
        
        # Stability queries
        elif "stability" in query or "stable" in query:
            args = {}
            if "endocrinology" in query:
                args["brand_areas"] = ["Endocrinology"]
            calls.append(ToolCall("call_1", "tool_analyze_stability", args))
        
        # Webinar vs Email comparison
        elif "webinar" in query or "email" in query:
            areas = []
            if "oncology" in query: areas.append("Oncology")
            
            calls.append(ToolCall("call_1", "tool_get_campaign_metrics", {
                "brand_areas": areas or None,
                "group_by": "tactic"
            }))
            
            # Also search KB
            keywords = []
            if "oncology" in query: keywords.append("oncology")
            if "webinar" in query: keywords.append("webinar")
            if "email" in query: keywords.append("email")
            
            if keywords:
                calls.append(ToolCall("call_2", "tool_search_knowledge_base", {
                    "keywords": keywords
                }))
        
        # Reallocation queries
        if "shift" in query or "reallocat" in query:
            area = "Cardiology"
            if "oncology" in query: area = "Oncology"
            elif "endocrinology" in query: area = "Endocrinology"
            
            calls.append(ToolCall("call_realloc", "tool_get_reallocation_suggestion", {
                "brand_area": area,
                "percent": 20.0
            }))
        
        return calls if calls else None
    
    def _generate_response(self, messages: List[Dict]) -> str:
        """Generate final response based on tool results."""
        # Collect tool results
        results = [m["content"] for m in messages if m.get("role") == "tool"]
        query = next((m["content"] for m in messages if m["role"] == "user"), "")
        
        # Simple response generation (a real LLM would do this much better)
        response_parts = ["## Analysis Results\n"]
        
        for result in results:
            try:
                data = json.loads(result)
                if isinstance(data, dict) and "recommendation" in data:
                    response_parts.append(f"**Recommendation**: {data['recommendation']}\n")
                elif isinstance(data, dict) and len(data) > 0:
                    first_key = list(data.keys())[0]
                    if isinstance(data[first_key], dict) and "overall_roi" in data[first_key]:
                        for area, metrics in data.items():
                            response_parts.append(f"**{area}**: ROI = {metrics['overall_roi']}, Best tactic = {metrics['best_tactic']}\n")
                elif isinstance(data, list) and len(data) > 0:
                    if "stability_rank" in data[0]:
                        response_parts.append("**Most Stable Tactics**:\n")
                        for item in data[:2]:
                            response_parts.append(f"- {item['tactic']}: std={item['roi_std']}, mean ROI={item['roi_mean']}\n")
                    elif "roi" in data[0]:
                        response_parts.append("**Performance by Tactic**:\n")
                        for item in data[:3]:
                            response_parts.append(f"- {item.get('tactic', 'N/A')}: ROI = {item['roi']}\n")
                    elif "title" in data[0]:
                        response_parts.append("**Knowledge Base Insights**:\n")
                        for doc in data:
                            response_parts.append(f"- {doc['title']}: {doc['text'][:100]}...\n")
            except:
                pass
        
        response_parts.append("\n## Next Best Action\n")
        response_parts.append("Based on the analysis, optimize budget allocation toward higher-ROI tactics.")
        
        return "".join(response_parts)


class OpenAIClient(BaseLLMClient):
    """
    Real OpenAI client for production use.
    
    Usage:
        client = OpenAIClient(api_key="sk-...")
        orchestrator = Orchestrator(llm=client)
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("pip install openai")
    
    def chat(self, messages: List[Dict], tools: List[Dict]) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                )
                for tc in msg.tool_calls
            ]
        
        return LLMResponse(content=msg.content, tool_calls=tool_calls)


# =============================================================================
# SECTION 5: THE ORCHESTRATOR (The Main Loop)
# =============================================================================

class MedscapeOrchestrator:
    """
    LLM-based orchestrator for campaign analysis.
    
    This implements the standard LLM agentic loop:
    1. Send query + available tools to LLM
    2. LLM decides which tools to call
    3. Execute the tools
    4. Send results back to LLM
    5. LLM generates final response (or calls more tools)
    
    The architecture is the same whether using MockLLM or real LLM.
    """
    
    def __init__(self, llm: BaseLLMClient = None):
        self.llm = llm or MockLLMClient()
        self.system_prompt = """You are a campaign performance analyst for Medscape.
Your job is to analyze campaign data and provide actionable recommendations.

When answering questions:
1. Use the available tools to gather data
2. Analyze the results
3. Provide a clear summary (2-3 sentences)
4. Give a specific, actionable next best action

Be specific with numbers. Don't be vague."""
    
    def process(self, query: str, verbose: bool = True) -> str:
        """
        Process a query using LLM orchestration.
        
        Args:
            query: Natural language question
            verbose: Print intermediate steps
        
        Returns:
            Final response string
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Agentic loop
        for iteration in range(10):  # Safety limit
            if verbose:
                print(f"\n🔄 Iteration {iteration + 1}")
            
            # LLM decides what to do
            response = self.llm.chat(messages, TOOL_DEFINITIONS)
            
            if response.tool_calls:
                if verbose:
                    print(f"   LLM decided to call {len(response.tool_calls)} tool(s):")
                
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", 
                         "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                        for tc in response.tool_calls
                    ]
                })
                
                # Execute each tool
                for tc in response.tool_calls:
                    if verbose:
                        print(f"   → {tc.name}({tc.arguments})")
                    
                    # Execute the tool
                    func = TOOL_MAP.get(tc.name)
                    if func:
                        result = func(**tc.arguments)
                    else:
                        result = json.dumps({"error": f"Unknown tool: {tc.name}"})
                    
                    if verbose:
                        preview = result[:100] + "..." if len(result) > 100 else result
                        print(f"   ✓ {preview}")
                    
                    # Add tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })
            
            else:
                # LLM gave final response
                if verbose:
                    print("   ✅ LLM generated final response")
                return response.content
        
        return "Error: Max iterations reached"


# =============================================================================
# SECTION 6: DEMO
# =============================================================================

def main():
    """Run demo with sample queries."""
    print("="*70)
    print("  MEDSCAPE AGENTIC AI SYSTEM")
    print("  Architecture: LLM Tool Orchestration")
    print("  Mode: Mock LLM (swap with OpenAIClient for production)")
    print("="*70)
    
    # Initialize with mock (no API key needed)
    orchestrator = MedscapeOrchestrator(llm=MockLLMClient())
    
    queries = [
        "Compare tactic performance for Cardiology vs Oncology in 2025Q2 and recommend where to shift 20% spend.",
        "For Endocrinology, which tactics have the highest ROI stability?",
        "Should we emphasize Webinars or Email for Oncology? Check the knowledge base."
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'#'*70}")
        print(f"  QUERY {i}")
        print(f"{'#'*70}")
        
        result = orchestrator.process(query, verbose=True)
        
        print(f"\n{'─'*60}")
        print("📊 FINAL OUTPUT:")
        print(f"{'─'*60}")
        print(result)
        
        if i < len(queries):
            input("\nPress Enter for next query...")


if __name__ == "__main__":
    main()
