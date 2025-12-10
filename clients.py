import json
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict


@dataclass
class LLMResponse:
    """Represents an LLM response."""
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]


# =============================================================================
# BASE CLIENT
# =============================================================================

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def chat(self, messages: List[Dict], tools: List[Dict]) -> LLMResponse:
        """Send messages to LLM and get response."""
        pass


# =============================================================================
# MOCK CLIENT (For Demo)
# =============================================================================

class MockLLMClient(BaseLLMClient):
    def chat(self, messages: List[Dict], tools: List[Dict]) -> LLMResponse:
        # Check if we already have tool results
        has_results = any(m.get("role") == "tool" for m in messages)
        
        if has_results:
            return LLMResponse(
                content=self._generate_response(messages),
                tool_calls=None
            )
        
        # Analyze query and decide which tools to call
        user_query = next((m["content"] for m in messages if m["role"] == "user"), "")
        tool_calls = self._decide_tools(user_query.lower())
        
        # If no tools matched, return helpful message
        if tool_calls is None:
            return LLMResponse(
                content=self._no_match_response(user_query),
                tool_calls=None
            )
        
        return LLMResponse(content=None, tool_calls=tool_calls)
    
    def _decide_tools(self, query: str) -> Optional[List[ToolCall]]:
        calls = []
        
        # Comparison queries
        if "compare" in query or "vs" in query:
            areas = []
            if "cardiology" in query: areas.append("Cardiology")
            if "oncology" in query: areas.append("Oncology")
            if "endocrinology" in query: areas.append("Endocrinology")
            
            args = {"brand_areas": areas or ["Cardiology", "Oncology"]}
            
            for q in ["2025q1", "2025q2", "2025q3"]:
                if q in query:
                    args["quarter"] = q.upper()
                    break
            
            calls.append(ToolCall("call_1", "tool_compare_brand_areas", args))
            
            # Also search KB for comparison context
            if areas:
                calls.append(ToolCall("call_kb", "tool_search_knowledge_base", {
                    "keywords": [a.lower() for a in areas] + ["comparison", "tactic"]
                }))
        
        # Stability queries
        elif "stability" in query or "stable" in query:
            args = {}
            area = None
            if "endocrinology" in query:
                args["brand_areas"] = ["Endocrinology"]
                area = "endocrinology"
            elif "cardiology" in query:
                args["brand_areas"] = ["Cardiology"]
                area = "cardiology"
            elif "oncology" in query:
                args["brand_areas"] = ["Oncology"]
                area = "oncology"
            calls.append(ToolCall("call_1", "tool_analyze_stability", args))
            
            # Search KB for stability/optimization guidance
            keywords = ["stability", "optimization"]
            if area:
                keywords.append(area)
            calls.append(ToolCall("call_kb", "tool_search_knowledge_base", {
                "keywords": keywords
            }))
        
        # Webinar vs Email comparison
        elif "webinar" in query or "email" in query:
            areas = []
            if "oncology" in query: areas.append("Oncology")
            elif "cardiology" in query: areas.append("Cardiology")
            elif "endocrinology" in query: areas.append("Endocrinology")
            
            calls.append(ToolCall("call_1", "tool_get_campaign_metrics", {
                "brand_areas": areas or None,
                "group_by": "tactic"
            }))
            
            # Search KB for tactic guidance
            keywords = []
            if "oncology" in query: keywords.append("oncology")
            if "cardiology" in query: keywords.append("cardiology")
            if "endocrinology" in query: keywords.append("endocrinology")
            if "webinar" in query: keywords.append("webinar")
            if "email" in query: keywords.append("email")
            
            if keywords:
                calls.append(ToolCall("call_kb", "tool_search_knowledge_base", {
                    "keywords": keywords
                }))
        
        # ROI or best/worst tactic queries
        elif "roi" in query or "best" in query or "worst" in query or "performance" in query or "tactic" in query:
            areas = []
            area_name = None
            if "oncology" in query: 
                areas.append("Oncology")
                area_name = "oncology"
            if "cardiology" in query: 
                areas.append("Cardiology")
                area_name = "cardiology"
            if "endocrinology" in query: 
                areas.append("Endocrinology")
                area_name = "endocrinology"
            
            calls.append(ToolCall("call_1", "tool_get_campaign_metrics", {
                "brand_areas": areas or None,
                "group_by": "tactic"
            }))
            
            # Search KB for tactic guidance
            keywords = ["tactic", "optimization"]
            if area_name:
                keywords.append(area_name)
            calls.append(ToolCall("call_kb", "tool_search_knowledge_base", {
                "keywords": keywords
            }))
        
        # Reallocation queries
        if "shift" in query or "reallocat" in query or "recommend" in query:
            area = "Cardiology"
            area_lower = "cardiology"
            if "oncology" in query: 
                area = "Oncology"
                area_lower = "oncology"
            elif "endocrinology" in query: 
                area = "Endocrinology"
                area_lower = "endocrinology"
            
            calls.append(ToolCall("call_realloc", "tool_get_reallocation_suggestion", {
                "brand_area": area,
                "percent": 20.0
            }))
            
            # Search KB for optimization tips if not already searching
            if not any(c.name == "tool_search_knowledge_base" for c in calls):
                calls.append(ToolCall("call_kb", "tool_search_knowledge_base", {
                    "keywords": [area_lower, "optimization", "shift"]
                }))
        
        # Fallback: check for brand area mentions
        if not calls:
            areas = []
            keywords = []
            if "oncology" in query: 
                areas.append("Oncology")
                keywords.append("oncology")
            if "cardiology" in query: 
                areas.append("Cardiology")
                keywords.append("cardiology")
            if "endocrinology" in query: 
                areas.append("Endocrinology")
                keywords.append("endocrinology")
            
            if areas:
                calls.append(ToolCall("call_1", "tool_get_campaign_metrics", {
                    "brand_areas": areas,
                    "group_by": "tactic"
                }))
                calls.append(ToolCall("call_kb", "tool_search_knowledge_base", {
                    "keywords": keywords + ["tactic"]
                }))
        
        return calls if calls else None
    
    def _generate_response(self, messages: List[Dict]) -> str:
        results = [m["content"] for m in messages if m.get("role") == "tool"]
        query = next((m["content"] for m in messages if m["role"] == "user"), "").lower()
        
        response_parts = []
        summary = ""
        next_action = ""
        
        for result in results:
            try:
                data = json.loads(result)
                
                # Reallocation recommendation
                if isinstance(data, dict) and "recommendation" in data:
                    response_parts.append(f"**Recommendation**: {data['recommendation']}\n")
                    response_parts.append(f"**Projected Impact**: ${data.get('projected_impact', 0):,.0f} additional revenue\n")
                    next_action = f"Execute the reallocation: {data['recommendation']}. Monitor performance for 2-4 weeks before further adjustments."
                
                # Brand area comparison
                elif isinstance(data, dict) and len(data) > 0:
                    first_key = list(data.keys())[0]
                    if isinstance(data[first_key], dict) and "overall_roi" in data[first_key]:
                        sorted_areas = sorted(data.items(), key=lambda x: x[1]['overall_roi'], reverse=True)
                        best_area = sorted_areas[0]
                        
                        # Generate summary
                        if len(sorted_areas) >= 2:
                            worse_area = sorted_areas[-1]
                            roi_diff = ((best_area[1]['overall_roi'] / worse_area[1]['overall_roi']) - 1)
                            summary = (
                                f"**{best_area[0]}** outperforms **{worse_area[0]}** with ROI of "
                                f"{best_area[1]['overall_roi']:.2f} vs {worse_area[1]['overall_roi']:.2f} "
                                f"({roi_diff:+.1%}). Best performing tactic: **{best_area[1]['best_tactic']}**."
                            )
                        else:
                            summary = (
                                f"**{best_area[0]}** has an overall ROI of {best_area[1]['overall_roi']:.2f}. "
                                f"Top tactic: **{best_area[1]['best_tactic']}** (ROI: {best_area[1]['best_tactic_roi']:.2f})."
                            )
                        
                        for area, metrics in data.items():
                            response_parts.append(
                                f"**{area}**: ROI = {metrics['overall_roi']}, "
                                f"Best = {metrics['best_tactic']} ({metrics['best_tactic_roi']}), "
                                f"Worst = {metrics['worst_tactic']} ({metrics['worst_tactic_roi']})\n"
                            )
                        
                        if len(sorted_areas) >= 2:
                            worse_area = sorted_areas[-1]
                            next_action = (
                                f"Apply **{best_area[0]}**'s successful tactics to **{worse_area[0]}**. "
                                f"Shift budget from {worse_area[1]['worst_tactic']} to {best_area[1]['best_tactic']}."
                            )
                        else:
                            next_action = (
                                f"For {best_area[0]}, shift budget from {best_area[1]['worst_tactic']} "
                                f"to {best_area[1]['best_tactic']} to maximize ROI."
                            )
                
                # List results (stability, metrics, KB)
                elif isinstance(data, list) and len(data) > 0:
                    # Stability analysis
                    if "stability_rank" in data[0]:
                        top_two = data[:2]
                        summary = (
                            f"The **two most stable tactics** are **{top_two[0]['tactic']}** "
                            f"(std: {top_two[0]['roi_std']:.3f}, mean ROI: {top_two[0]['roi_mean']:.2f}) and "
                            f"**{top_two[1]['tactic']}** (std: {top_two[1]['roi_std']:.3f}, mean ROI: {top_two[1]['roi_mean']:.2f}). "
                            f"Lower standard deviation = more consistent performance."
                        )
                        
                        response_parts.append("**Full Stability Ranking**:\n")
                        for item in data:
                            response_parts.append(
                                f"- **{item['tactic']}**: std={item['roi_std']:.3f}, "
                                f"mean ROI={item['roi_mean']:.2f}\n"
                            )
                        
                        most_stable = data[0]['tactic']
                        highest_roi = max(data, key=lambda x: x['roi_mean'])
                        
                        if highest_roi['tactic'] != most_stable:
                            next_action = (
                                f"Use **{most_stable}** as your anchor tactic for predictable returns. "
                                f"Pair with **{highest_roi['tactic']}** (highest mean ROI: {highest_roi['roi_mean']:.2f}) for growth."
                            )
                        else:
                            next_action = (
                                f"**{most_stable}** is both the most stable AND highest ROI. "
                                f"Prioritize this tactic for reliable, high-performing campaigns."
                            )
                    
                    # ROI metrics
                    elif "roi" in data[0]:
                        best = data[0]
                        worst = data[-1] if len(data) > 1 else None
                        
                        summary = f"Top performing tactic: **{best.get('tactic', 'N/A')}** with ROI of {best['roi']:.2f}."
                        if worst:
                            summary += f" Lowest: **{worst.get('tactic', 'N/A')}** (ROI: {worst['roi']:.2f})."
                        
                        response_parts.append("**Performance by Tactic**:\n")
                        for item in data:
                            response_parts.append(
                                f"- **{item.get('tactic', 'N/A')}**: ROI = {item['roi']:.2f}, "
                                f"Spend = ${item.get('spend', 0):,.0f}\n"
                            )
                        
                        if worst:
                            next_action = f"Prioritize **{best.get('tactic')}** (highest ROI). Consider reducing **{worst.get('tactic')}** spend and reallocating to top performers."
                        else:
                            next_action = f"Continue investing in **{best.get('tactic')}**."
                    
                    # Knowledge base results
                    elif "title" in data[0]:
                        response_parts.append("\n**Knowledge Base Insights**:\n")
                        for doc in data:
                            response_parts.append(f"- *{doc['title']}*: {doc['text']}\n")
            except:
                pass
        
        # Build final response with Summary first
        final_parts = []
        
        if summary:
            final_parts.append("## Summary\n")
            final_parts.append(summary + "\n")
        
        if response_parts:
            final_parts.append("\n## Detailed Analysis\n")
            final_parts.extend(response_parts)
        
        final_parts.append("\n## Next Best Action\n")
        if next_action:
            final_parts.append(next_action)
        else:
            final_parts.append("Review the data above and consider reallocating budget toward higher-ROI tactics.")
        
        return "".join(final_parts)
    
    def _no_match_response(self, query: str) -> str:
        """Generate helpful response when query doesn't match available data."""
        return """## Unable to Process Query

I couldn't find relevant data for your question. This could be because:

1. Brand area not recognized: Available brand areas are:
   - Cardiology
   - Oncology  
   - Endocrinology

2. Question type not supported: Try asking about:
   - Comparisons (e.g., "Compare Cardiology vs Oncology")
   - ROI/Performance (e.g., "What's the best tactic for Oncology?")
   - Stability (e.g., "Which tactics have stable ROI?")
   - Recommendations (e.g., "Recommend where to shift spend")

3. Time period: Available quarters are 2025Q1, 2025Q2, 2025Q3
Example queries:
- "Compare Cardiology vs Oncology in 2025Q2"
- "What's the ROI stability for Endocrinology?"
- "Should we use Webinars or Email for Oncology?"

Please try rephrasing your question."""


# =============================================================================
# OPENAI CLIENT (For Production)
# =============================================================================

class OpenAIClient(BaseLLMClient):
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
