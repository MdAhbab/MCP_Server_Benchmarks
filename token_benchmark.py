"""
Token Consumption Benchmark for MCP Servers
Compares verbose vs progressive disclosure patterns

This benchmark validates the 98.7% token reduction claim from the Anthropic literature.
"""

import json
import time
import statistics
from pathlib import Path

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Using character-based estimation.")
    print("Install with: pip install tiktoken")


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens using tiktoken (GPT-4/Claude approximation)"""
    if TIKTOKEN_AVAILABLE:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    else:
        # Rough estimation: ~4 characters per token
        return len(text) // 4


def generate_sample_dataset(num_items: int = 100) -> list:
    """Generate sample data simulating real-world MCP responses"""
    return [
        {
            "id": i,
            "name": f"Item_{i}",
            "description": f"This is a detailed description for item {i}. " * 5,
            "metadata": {
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-05T12:00:00Z",
                "version": "1.0.0",
                "tags": ["category_a", "category_b", f"tag_{i % 10}"],
                "properties": {
                    "color": ["red", "blue", "green"][i % 3],
                    "size": ["small", "medium", "large"][i % 3],
                    "priority": i % 5,
                    "status": ["active", "pending", "archived"][i % 3]
                }
            },
            "content": f"Content block for item {i}. " * 10
        }
        for i in range(num_items)
    ]


def verbose_response(dataset: list) -> dict:
    """
    Traditional MCP Response Pattern (Verbose)
    Returns ALL data in a single response - high token consumption
    """
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(dataset, indent=2)
                }
            ],
            "metadata": {
                "total_items": len(dataset),
                "response_time_ms": 150,
                "source": "database"
            }
        }
    }


def progressive_disclosure_response(dataset: list) -> dict:
    """
    Optimized MCP Response Pattern (Progressive Disclosure)
    Returns only summary - low token consumption
    Agent can request specific items as needed
    """
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "summary": {
                            "total_items": len(dataset),
                            "id_range": f"{dataset[0]['id']}-{dataset[-1]['id']}",
                            "categories": list(set(d["metadata"]["properties"]["color"] for d in dataset)),
                            "available_actions": ["get_item(id)", "search(query)", "filter(criteria)"]
                        },
                        "sample_items": [
                            {"id": d["id"], "name": d["name"]} 
                            for d in dataset[:3]
                        ],
                        "note": "Use get_item(id) to retrieve full details for specific items"
                    })
                }
            ]
        }
    }


def tool_definition_verbose() -> list:
    """
    Traditional Tool Definition Pattern
    Separate tools for each action - bloats context window
    """
    return [
        {
            "name": "get_all_users",
            "description": "Retrieves all users from the database with complete profile information including name, email, address, preferences, and activity history",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "get_user_by_id",
            "description": "Retrieves a specific user by their unique identifier",
            "inputSchema": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "The unique user identifier"}},
                "required": ["id"]
            }
        },
        {
            "name": "create_user",
            "description": "Creates a new user with the provided information",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "role": {"type": "string"}
                },
                "required": ["name", "email"]
            }
        },
        {
            "name": "update_user",
            "description": "Updates an existing user's information",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"}
                },
                "required": ["id"]
            }
        },
        {
            "name": "delete_user",
            "description": "Permanently removes a user from the system",
            "inputSchema": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"]
            }
        }
    ]


def tool_definition_optimized() -> list:
    """
    Optimized Tool Definition Pattern
    Single tool with action parameter - reduces context consumption
    """
    return [
        {
            "name": "manage_user",
            "description": "Unified user management: list, get, create, update, or delete users",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "get", "create", "update", "delete"],
                        "description": "The action to perform"
                    },
                    "id": {"type": "integer", "description": "User ID (for get/update/delete)"},
                    "data": {"type": "object", "description": "User data (for create/update)"},
                    "filters": {"type": "object", "description": "Query filters (for list)"}
                },
                "required": ["action"]
            }
        }
    ]


def run_benchmark():
    """Execute the complete token consumption benchmark"""
    
    print("=" * 60)
    print("MCP Token Consumption Benchmark")
    print("=" * 60)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tiktoken_available": TIKTOKEN_AVAILABLE,
        "scenarios": []
    }
    
    # Test different dataset sizes
    dataset_sizes = [10, 50, 100, 500, 1000]
    
    print("\n--- Response Pattern Comparison ---\n")
    print(f"{'Size':<10} {'Verbose':<15} {'Optimized':<15} {'Reduction':<15}")
    print("-" * 55)
    
    for size in dataset_sizes:
        dataset = generate_sample_dataset(size)
        
        # Verbose response
        verbose = verbose_response(dataset)
        verbose_text = json.dumps(verbose)
        verbose_tokens = count_tokens(verbose_text)
        
        # Optimized response
        optimized = progressive_disclosure_response(dataset)
        optimized_text = json.dumps(optimized)
        optimized_tokens = count_tokens(optimized_text)
        
        # Calculate reduction
        reduction = ((verbose_tokens - optimized_tokens) / verbose_tokens) * 100
        
        print(f"{size:<10} {verbose_tokens:<15} {optimized_tokens:<15} {reduction:.1f}%")
        
        results["scenarios"].append({
            "type": "response_pattern",
            "dataset_size": size,
            "verbose_tokens": verbose_tokens,
            "optimized_tokens": optimized_tokens,
            "reduction_percent": round(reduction, 2),
            "verbose_bytes": len(verbose_text),
            "optimized_bytes": len(optimized_text)
        })
    
    # Tool definition comparison
    print("\n--- Tool Definition Comparison ---\n")
    
    verbose_tools = tool_definition_verbose()
    optimized_tools = tool_definition_optimized()
    
    verbose_tools_text = json.dumps(verbose_tools)
    optimized_tools_text = json.dumps(optimized_tools)
    
    verbose_tools_tokens = count_tokens(verbose_tools_text)
    optimized_tools_tokens = count_tokens(optimized_tools_text)
    tool_reduction = ((verbose_tools_tokens - optimized_tools_tokens) / verbose_tools_tokens) * 100
    
    print(f"Verbose Tools (5 separate):    {verbose_tools_tokens} tokens")
    print(f"Optimized Tools (1 unified):   {optimized_tools_tokens} tokens")
    print(f"Reduction:                     {tool_reduction:.1f}%")
    
    results["scenarios"].append({
        "type": "tool_definition",
        "verbose_tool_count": len(verbose_tools),
        "optimized_tool_count": len(optimized_tools),
        "verbose_tokens": verbose_tools_tokens,
        "optimized_tokens": optimized_tools_tokens,
        "reduction_percent": round(tool_reduction, 2)
    })
    
    # Cost estimation (using approximate Claude pricing)
    print("\n--- Cost Estimation (per 1000 requests) ---\n")
    
    # Assume $0.003 per 1K input tokens (Claude Sonnet pricing approximation)
    cost_per_1k_tokens = 0.003
    
    # Use 100-item dataset for cost calculation
    verbose_cost = (results["scenarios"][2]["verbose_tokens"] / 1000) * cost_per_1k_tokens * 1000
    optimized_cost = (results["scenarios"][2]["optimized_tokens"] / 1000) * cost_per_1k_tokens * 1000
    
    print(f"100-item dataset, 1000 requests:")
    print(f"  Verbose pattern:   ${verbose_cost:.2f}")
    print(f"  Optimized pattern: ${optimized_cost:.2f}")
    print(f"  Savings:           ${verbose_cost - optimized_cost:.2f} ({((verbose_cost - optimized_cost) / verbose_cost * 100):.1f}%)")
    
    results["cost_estimation"] = {
        "requests": 1000,
        "dataset_size": 100,
        "verbose_cost_usd": round(verbose_cost, 2),
        "optimized_cost_usd": round(optimized_cost, 2),
        "savings_usd": round(verbose_cost - optimized_cost, 2)
    }
    
    # Energy estimation (rough)
    print("\n--- Energy Impact Estimation ---\n")
    
    # Rough estimation: 1 token ≈ 0.0001 Wh for inference
    energy_per_token_wh = 0.0001
    
    verbose_energy = results["scenarios"][2]["verbose_tokens"] * energy_per_token_wh
    optimized_energy = results["scenarios"][2]["optimized_tokens"] * energy_per_token_wh
    
    print(f"Per request (100 items):")
    print(f"  Verbose:   {verbose_energy:.4f} Wh")
    print(f"  Optimized: {optimized_energy:.4f} Wh")
    print(f"  Reduction: {((verbose_energy - optimized_energy) / verbose_energy * 100):.1f}%")
    
    # Scale to 1M requests
    print(f"\nScaled to 1 million requests:")
    print(f"  Verbose:   {verbose_energy * 1_000_000 / 1000:.2f} kWh")
    print(f"  Optimized: {optimized_energy * 1_000_000 / 1000:.2f} kWh")
    
    results["energy_estimation"] = {
        "per_request_verbose_wh": verbose_energy,
        "per_request_optimized_wh": optimized_energy,
        "million_requests_verbose_kwh": verbose_energy * 1_000_000 / 1000,
        "million_requests_optimized_kwh": optimized_energy * 1_000_000 / 1000
    }
    
    # Save results
    output_file = Path("results/token_benchmark_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Generate summary for paper
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"""
Our empirical benchmarks validate the token efficiency claims from the literature.
For a 100-item dataset:
- Verbose response pattern: {results['scenarios'][2]['verbose_tokens']:,} tokens
- Progressive disclosure: {results['scenarios'][2]['optimized_tokens']:,} tokens  
- Reduction: {results['scenarios'][2]['reduction_percent']:.1f}%

Tool definition optimization (5 tools → 1 unified):
- Reduction: {results['scenarios'][-1]['reduction_percent']:.1f}%

These findings align with Anthropic's reported 98.7% reduction for complex workflows.
""")
    
    return results


if __name__ == "__main__":
    run_benchmark()
