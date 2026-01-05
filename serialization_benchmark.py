"""
Serialization Overhead Benchmark for MCP Servers
Compares JSON-RPC vs binary serialization formats

This benchmark quantifies the "Standardization Tax" of JSON-RPC in MCP.
"""

import json
import time
import statistics
import base64
from pathlib import Path
from typing import Dict, Any

# Optional binary serialization libraries
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False


def create_test_payloads() -> Dict[str, Any]:
    """Create test payloads of varying complexity matching MCP use cases"""
    
    # Small payload: Simple tool call
    small = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "get_file",
            "arguments": {"path": "/home/user/document.txt"}
        }
    }
    
    # Medium payload: Tool response with data
    medium = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": "x" * 1000  # 1KB of text content
                }
            ],
            "metadata": {
                "tokens_used": 150,
                "model": "claude-3-sonnet",
                "timestamp": "2025-01-05T12:00:00Z",
                "processing_time_ms": 245
            }
        }
    }
    
    # Large payload: Tool definitions (context bloat scenario)
    large = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {
                    "name": f"tool_{i}",
                    "description": f"This is the description for tool number {i}. It provides functionality for handling various operations related to domain {i % 5}. " * 3,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            f"param_{j}": {
                                "type": ["string", "integer", "boolean"][j % 3],
                                "description": f"Parameter {j} for tool {i}"
                            }
                            for j in range(10)
                        },
                        "required": [f"param_{j}" for j in range(3)]
                    }
                }
                for i in range(50)  # 50 tools
            ]
        }
    }
    
    # Binary payload: Simulating file/image data (worst case for JSON)
    binary_data = bytes(range(256)) * 100  # 25.6 KB of binary data
    binary = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {
                    "type": "binary",
                    "data": base64.b64encode(binary_data).decode('utf-8'),
                    "mime_type": "application/octet-stream"
                }
            ]
        }
    }
    
    return {
        "small": small,
        "medium": medium,
        "large": large,
        "binary": binary
    }


def benchmark_json(data: dict, iterations: int = 5000) -> Dict[str, Any]:
    """Benchmark JSON serialization/deserialization"""
    
    # Serialization
    serialize_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        serialized = json.dumps(data)
        serialize_times.append(time.perf_counter() - start)
    
    serialized = json.dumps(data)
    
    # Deserialization
    deserialize_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        json.loads(serialized)
        deserialize_times.append(time.perf_counter() - start)
    
    return {
        "format": "JSON",
        "size_bytes": len(serialized.encode('utf-8')),
        "serialize_mean_us": statistics.mean(serialize_times) * 1_000_000,
        "serialize_std_us": statistics.stdev(serialize_times) * 1_000_000,
        "deserialize_mean_us": statistics.mean(deserialize_times) * 1_000_000,
        "deserialize_std_us": statistics.stdev(deserialize_times) * 1_000_000,
        "total_mean_us": (statistics.mean(serialize_times) + statistics.mean(deserialize_times)) * 1_000_000
    }


def benchmark_msgpack(data: dict, iterations: int = 5000) -> Dict[str, Any]:
    """Benchmark MessagePack serialization"""
    if not MSGPACK_AVAILABLE:
        return {"format": "MessagePack", "error": "Not installed (pip install msgpack)"}
    
    # Serialization
    serialize_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        serialized = msgpack.packb(data, use_bin_type=True)
        serialize_times.append(time.perf_counter() - start)
    
    serialized = msgpack.packb(data, use_bin_type=True)
    
    # Deserialization
    deserialize_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        msgpack.unpackb(serialized, raw=False)
        deserialize_times.append(time.perf_counter() - start)
    
    return {
        "format": "MessagePack",
        "size_bytes": len(serialized),
        "serialize_mean_us": statistics.mean(serialize_times) * 1_000_000,
        "serialize_std_us": statistics.stdev(serialize_times) * 1_000_000,
        "deserialize_mean_us": statistics.mean(deserialize_times) * 1_000_000,
        "deserialize_std_us": statistics.stdev(deserialize_times) * 1_000_000,
        "total_mean_us": (statistics.mean(serialize_times) + statistics.mean(deserialize_times)) * 1_000_000
    }


def benchmark_pickle(data: dict, iterations: int = 5000) -> Dict[str, Any]:
    """Benchmark Python Pickle serialization (baseline comparison)"""
    
    # Serialization
    serialize_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        serialized = pickle.dumps(data)
        serialize_times.append(time.perf_counter() - start)
    
    serialized = pickle.dumps(data)
    
    # Deserialization
    deserialize_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        pickle.loads(serialized)
        deserialize_times.append(time.perf_counter() - start)
    
    return {
        "format": "Pickle",
        "size_bytes": len(serialized),
        "serialize_mean_us": statistics.mean(serialize_times) * 1_000_000,
        "serialize_std_us": statistics.stdev(serialize_times) * 1_000_000,
        "deserialize_mean_us": statistics.mean(deserialize_times) * 1_000_000,
        "deserialize_std_us": statistics.stdev(deserialize_times) * 1_000_000,
        "total_mean_us": (statistics.mean(serialize_times) + statistics.mean(deserialize_times)) * 1_000_000
    }


def calculate_base64_overhead(original_size: int) -> Dict[str, Any]:
    """Calculate Base64 encoding overhead for binary data"""
    # Base64 increases size by ~33%
    base64_size = ((original_size + 2) // 3) * 4
    overhead = (base64_size - original_size) / original_size * 100
    
    return {
        "original_bytes": original_size,
        "base64_bytes": base64_size,
        "overhead_percent": round(overhead, 2)
    }


def run_benchmark():
    """Execute the complete serialization benchmark"""
    
    print("=" * 70)
    print("MCP Serialization Overhead Benchmark")
    print("=" * 70)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iterations": 5000,
        "payloads": {},
        "base64_overhead": {}
    }
    
    payloads = create_test_payloads()
    
    for payload_name, payload_data in payloads.items():
        print(f"\n--- {payload_name.upper()} Payload ---\n")
        
        payload_results = []
        
        # JSON benchmark
        json_result = benchmark_json(payload_data)
        payload_results.append(json_result)
        
        # MessagePack benchmark
        msgpack_result = benchmark_msgpack(payload_data)
        payload_results.append(msgpack_result)
        
        # Pickle benchmark (baseline)
        pickle_result = benchmark_pickle(payload_data)
        payload_results.append(pickle_result)
        
        # Print results table
        print(f"{'Format':<15} {'Size (B)':<12} {'Ser (µs)':<12} {'Deser (µs)':<12} {'Total (µs)':<12}")
        print("-" * 63)
        
        for r in payload_results:
            if "error" not in r:
                print(f"{r['format']:<15} {r['size_bytes']:<12} {r['serialize_mean_us']:<12.2f} {r['deserialize_mean_us']:<12.2f} {r['total_mean_us']:<12.2f}")
            else:
                print(f"{r['format']:<15} {r['error']}")
        
        # Calculate overhead vs MessagePack
        if "error" not in msgpack_result:
            json_overhead = (json_result['size_bytes'] - msgpack_result['size_bytes']) / msgpack_result['size_bytes'] * 100
            json_time_overhead = (json_result['total_mean_us'] - msgpack_result['total_mean_us']) / msgpack_result['total_mean_us'] * 100
            
            print(f"\nJSON overhead vs MessagePack:")
            print(f"  Size: +{json_overhead:.1f}%")
            print(f"  Time: +{json_time_overhead:.1f}%")
        
        results["payloads"][payload_name] = payload_results
    
    # Base64 overhead analysis
    print("\n" + "=" * 70)
    print("Base64 Encoding Overhead (for binary data in JSON)")
    print("=" * 70 + "\n")
    
    binary_sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB
    
    print(f"{'Original Size':<15} {'Base64 Size':<15} {'Overhead':<15}")
    print("-" * 45)
    
    for size in binary_sizes:
        overhead = calculate_base64_overhead(size)
        size_label = f"{size/1024:.0f}KB" if size < 1048576 else f"{size/1048576:.0f}MB"
        b64_label = f"{overhead['base64_bytes']/1024:.0f}KB" if overhead['base64_bytes'] < 1048576 else f"{overhead['base64_bytes']/1048576:.1f}MB"
        print(f"{size_label:<15} {b64_label:<15} +{overhead['overhead_percent']:.1f}%")
        results["base64_overhead"][size_label] = overhead
    
    # Energy impact estimation
    print("\n" + "=" * 70)
    print("Energy Impact Estimation")
    print("=" * 70 + "\n")
    
    # Rough estimation: 1 µs of CPU time ≈ 0.00001 Wh at 100W TDP
    cpu_power_w = 65  # Typical desktop CPU TDP
    
    json_large = results["payloads"]["large"][0]  # JSON result for large payload
    
    # Per-request energy
    json_energy_wh = (json_large['total_mean_us'] / 1_000_000 / 3600) * cpu_power_w
    
    print(f"Large payload (50 tools) JSON serialization:")
    print(f"  Time per request: {json_large['total_mean_us']:.2f} µs")
    print(f"  Energy per request: {json_energy_wh:.10f} Wh")
    
    # Scale to 1M requests
    million_requests_wh = json_energy_wh * 1_000_000
    print(f"\nScaled to 1 million requests:")
    print(f"  Total CPU time: {json_large['total_mean_us'] * 1_000_000 / 1_000_000:.2f} seconds")
    print(f"  Total energy: {million_requests_wh:.4f} Wh ({million_requests_wh/1000:.6f} kWh)")
    
    results["energy_estimation"] = {
        "cpu_tdp_w": cpu_power_w,
        "per_request_wh": json_energy_wh,
        "million_requests_wh": million_requests_wh
    }
    
    # Save results
    output_file = Path("results/serialization_benchmark_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Paper summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    
    if "error" not in results["payloads"]["large"][1]:  # MessagePack available
        json_r = results["payloads"]["large"][0]
        msgpack_r = results["payloads"]["large"][1]
        size_overhead = (json_r['size_bytes'] - msgpack_r['size_bytes']) / msgpack_r['size_bytes'] * 100
        
        print(f"""
Serialization overhead analysis for MCP tool definitions (50 tools):

JSON-RPC 2.0 (MCP Standard):
  - Payload size: {json_r['size_bytes']:,} bytes
  - Round-trip time: {json_r['total_mean_us']:.2f} µs

MessagePack (Binary Alternative):
  - Payload size: {msgpack_r['size_bytes']:,} bytes
  - Round-trip time: {msgpack_r['total_mean_us']:.2f} µs

Findings:
  - JSON size overhead: +{size_overhead:.1f}% vs binary format
  - Base64 encoding overhead: +33.3% for binary data
  - This validates the "Standardization Tax" documented in the literature

Recommendation: Binary transport extensions (Protocol Buffers, MessagePack)
could reduce payload sizes by ~{size_overhead:.0f}% for tool-heavy integrations.
""")
    
    return results


if __name__ == "__main__":
    run_benchmark()
