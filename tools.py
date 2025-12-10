import json
import os
from typing import List, Dict, Optional
import pandas as pd


# =============================================================================
# DATA LAYER
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_campaign_data() -> pd.DataFrame:
    """Load campaign performance data."""
    return pd.read_csv(os.path.join(DATA_DIR, "campaign_performance.csv"))


def load_knowledge_base() -> List[Dict]:
    """Load knowledge base documents."""
    docs = []
    with open(os.path.join(DATA_DIR, "kb_documents.jsonl"), 'r') as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


# =============================================================================
# TOOL FUNCTIONS
# =============================================================================

def tool_get_campaign_metrics(
    brand_areas: Optional[List[str]] = None,
    quarters: Optional[List[str]] = None,
    group_by: str = "tactic"
) -> str:
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
# TOOL DEFINITIONS (OpenAI Function Calling Format)
# =============================================================================

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

# Map function names to actual functions
TOOL_MAP = {
    "tool_get_campaign_metrics": tool_get_campaign_metrics,
    "tool_compare_brand_areas": tool_compare_brand_areas,
    "tool_analyze_stability": tool_analyze_stability,
    "tool_search_knowledge_base": tool_search_knowledge_base,
    "tool_get_reallocation_suggestion": tool_get_reallocation_suggestion
}
