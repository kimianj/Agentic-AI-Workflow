import json
from typing import List, Dict

from clients import BaseLLMClient, MockLLMClient
from tools import TOOL_DEFINITIONS, TOOL_MAP


class MedscapeOrchestrator:
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
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                            }
                        }
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
