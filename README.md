# MCP Server Benchmarking Guide for Green Computing Research

## Overview

This guide provides detailed instructions for conducting benchmarking experiments on MCP (Model Context Protocol) servers to generate empirical data for your survey paper on energy-efficient MCP architectures.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Benchmark Scenarios](#2-benchmark-scenarios)
3. [Measurement Tools](#3-measurement-tools)
4. [Detailed Benchmarking Procedures](#4-detailed-benchmarking-procedures)
5. [Data Collection Templates](#5-data-collection-templates)
6. [Analysis Methods](#6-analysis-methods)
7. [Paper Integration Guidelines](#7-paper-integration-guidelines)

---

## 1. Environment Setup

### 1.1 Hardware Requirements Documentation

Before benchmarking, document your system specifications for reproducibility:

```powershell
# Get system information
systeminfo | Select-String "OS Name|OS Version|System Type|Total Physical Memory|Processor"

# Get CPU details
Get-WmiObject -Class Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed

# Get RAM details
Get-WmiObject -Class Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum

# Get disk type (SSD/HDD)
Get-PhysicalDisk | Select-Object MediaType, Size, FriendlyName
```

### 1.2 Software Prerequisites

#### Install Node.js (for JavaScript MCP servers)
```powershell
# Using winget
winget install OpenJS.NodeJS.LTS

# Verify installation
node --version
npm --version
```

#### Install Python (for Python MCP servers)
```powershell
# Using winget
winget install Python.Python.3.12

# Verify installation
python --version
pip --version
```

#### Install Docker (for containerized benchmarks)
```powershell
# Download Docker Desktop from https://www.docker.com/products/docker-desktop/
# Or using winget
winget install Docker.DockerDesktop

# Verify installation
docker --version
```

#### Install UV (Fast Python package manager for MCP)
```powershell
# Install UV
pip install uv

# Or using PowerShell
irm https://astral.sh/uv/install.ps1 | iex
```

### 1.3 MCP SDK Installation

```powershell
# Create benchmark workspace
mkdir C:\MCP_Benchmarks
cd C:\MCP_Benchmarks

# Install MCP SDK for Python
pip install mcp

# Install MCP SDK for TypeScript/JavaScript
npm init -y
npm install @modelcontextprotocol/sdk
```

### 1.4 Install Claude Desktop (MCP Host)

Download from: https://claude.ai/download

Configure MCP servers in: `%APPDATA%\Claude\claude_desktop_config.json`

---

## 2. Benchmark Scenarios

Based on your survey, here are the key scenarios to benchmark:

### Scenario 1: Token Consumption Comparison
**Goal:** Measure token usage in traditional ReAct vs Progressive Disclosure patterns

### Scenario 2: Transport Mechanism Latency
**Goal:** Compare stdio vs HTTP vs SSE transport latencies

### Scenario 3: Serialization Overhead
**Goal:** Measure JSON-RPC serialization costs vs alternatives

### Scenario 4: Caching Effectiveness
**Goal:** Quantify performance improvements from caching strategies

### Scenario 5: Resource Utilization
**Goal:** Measure CPU, memory, and energy consumption across server types

### Scenario 6: Multi-Server Scalability
**Goal:** Test performance degradation with increasing server count

---

## 3. Measurement Tools

### 3.1 Energy Measurement Tools

#### Windows Built-in Power Monitoring
```powershell
# Enable energy estimation (requires admin)
powercfg /energy /duration 60

# View power usage
powercfg /batteryreport
```

#### Intel Power Gadget (for Intel CPUs)
Download: https://www.intel.com/content/www/us/en/developer/articles/tool/power-gadget.html

#### HWiNFO64 (Comprehensive hardware monitoring)
Download: https://www.hwinfo.com/download/

#### Open Hardware Monitor (CPU/GPU power)
```powershell
# Install via Chocolatey
choco install openhardwaremonitor
```

### 3.2 Performance Monitoring Tools

#### Custom PowerShell Monitoring Script
```powershell
# Save as: C:\MCP_Benchmarks\monitor.ps1

param(
    [int]$DurationSeconds = 60,
    [int]$IntervalMs = 1000,
    [string]$OutputFile = "metrics.csv"
)

$metrics = @()
$endTime = (Get-Date).AddSeconds($DurationSeconds)

while ((Get-Date) -lt $endTime) {
    $cpu = Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue
    $mem = Get-Counter '\Memory\Available MBytes' -ErrorAction SilentlyContinue
    $disk = Get-Counter '\PhysicalDisk(_Total)\Disk Bytes/sec' -ErrorAction SilentlyContinue
    
    $metrics += [PSCustomObject]@{
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
        CPU_Percent = [math]::Round($cpu.CounterSamples[0].CookedValue, 2)
        Available_Memory_MB = [math]::Round($mem.CounterSamples[0].CookedValue, 2)
        Disk_Bytes_Sec = [math]::Round($disk.CounterSamples[0].CookedValue, 2)
    }
    
    Start-Sleep -Milliseconds $IntervalMs
}

$metrics | Export-Csv -Path $OutputFile -NoTypeInformation
Write-Host "Metrics saved to $OutputFile"
```

### 3.3 Network Monitoring

```powershell
# Install Wireshark for detailed packet analysis
winget install WiresharkFoundation.Wireshark

# Or use built-in netstat for connection monitoring
netstat -an | findstr "ESTABLISHED"
```

### 3.4 Process-Specific Monitoring

```powershell
# Save as: C:\MCP_Benchmarks\process_monitor.ps1

param(
    [string]$ProcessName,
    [int]$DurationSeconds = 60,
    [string]$OutputFile = "process_metrics.csv"
)

$metrics = @()
$endTime = (Get-Date).AddSeconds($DurationSeconds)

while ((Get-Date) -lt $endTime) {
    $process = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
    
    if ($process) {
        $metrics += [PSCustomObject]@{
            Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
            CPU_Time_Seconds = $process.CPU
            Working_Set_MB = [math]::Round($process.WorkingSet64 / 1MB, 2)
            Private_Memory_MB = [math]::Round($process.PrivateMemorySize64 / 1MB, 2)
            Thread_Count = $process.Threads.Count
            Handle_Count = $process.HandleCount
        }
    }
    
    Start-Sleep -Milliseconds 500
}

$metrics | Export-Csv -Path $OutputFile -NoTypeInformation
```

---

## 4. Detailed Benchmarking Procedures

### 4.1 Benchmark 1: Token Consumption Analysis

#### Setup: Create Test MCP Servers

**Traditional Server (Verbose Response):**
```python
# Save as: C:\MCP_Benchmarks\servers\verbose_server.py

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
import asyncio
import json

server = Server("verbose-server")

# Simulate large dataset
SAMPLE_DATA = [
    {"id": i, "name": f"Item_{i}", "description": f"Description for item {i} " * 10, 
     "metadata": {"created": "2025-01-01", "modified": "2025-01-05", "tags": ["tag1", "tag2", "tag3"]}}
    for i in range(100)
]

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_all_data",
            description="Returns ALL data items with full details",
            inputSchema={"type": "object", "properties": {}}
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_all_data":
        # Return everything (inefficient - high token count)
        return [TextContent(type="text", text=json.dumps(SAMPLE_DATA, indent=2))]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

**Optimized Server (Progressive Disclosure):**
```python
# Save as: C:\MCP_Benchmarks\servers\optimized_server.py

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
import asyncio
import json

server = Server("optimized-server")

# Same dataset
SAMPLE_DATA = [
    {"id": i, "name": f"Item_{i}", "description": f"Description for item {i} " * 10,
     "metadata": {"created": "2025-01-01", "modified": "2025-01-05", "tags": ["tag1", "tag2", "tag3"]}}
    for i in range(100)
]

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_summary",
            description="Returns summary statistics only",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_item",
            description="Returns a specific item by ID",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"]
            }
        ),
        Tool(
            name="search_items",
            description="Search items with filters, returns IDs only",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10}
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_summary":
        # Return only summary (efficient - low token count)
        summary = {
            "total_items": len(SAMPLE_DATA),
            "id_range": f"{SAMPLE_DATA[0]['id']}-{SAMPLE_DATA[-1]['id']}",
            "sample_names": [d["name"] for d in SAMPLE_DATA[:3]]
        }
        return [TextContent(type="text", text=json.dumps(summary))]
    
    elif name == "get_item":
        item_id = arguments.get("id", 0)
        item = next((d for d in SAMPLE_DATA if d["id"] == item_id), None)
        return [TextContent(type="text", text=json.dumps(item) if item else "Not found")]
    
    elif name == "search_items":
        limit = arguments.get("limit", 10)
        # Return only IDs
        ids = [d["id"] for d in SAMPLE_DATA[:limit]]
        return [TextContent(type="text", text=json.dumps({"matching_ids": ids}))]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

#### Token Counting Script
```python
# Save as: C:\MCP_Benchmarks\token_counter.py

import tiktoken
import json

def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens using tiktoken (GPT-4/Claude approximation)"""
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))

def analyze_response(response_file: str):
    with open(response_file, 'r') as f:
        data = json.load(f)
    
    text = json.dumps(data)
    tokens = count_tokens(text)
    
    print(f"Response size: {len(text)} characters")
    print(f"Token count: {tokens}")
    print(f"Estimated cost (at $0.01/1K tokens): ${tokens * 0.01 / 1000:.4f}")
    
    return tokens

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_response(sys.argv[1])
```

#### Run Token Benchmark
```powershell
# Install tiktoken
pip install tiktoken

# Test verbose server response
python -c "
import json
data = [{'id': i, 'name': f'Item_{i}', 'description': f'Description for item {i} ' * 10} for i in range(100)]
with open('verbose_response.json', 'w') as f:
    json.dump(data, f, indent=2)
"

# Test optimized server response
python -c "
import json
summary = {'total_items': 100, 'id_range': '0-99', 'sample_names': ['Item_0', 'Item_1', 'Item_2']}
with open('optimized_response.json', 'w') as f:
    json.dump(summary, f)
"

# Count tokens
python token_counter.py verbose_response.json
python token_counter.py optimized_response.json
```

### 4.2 Benchmark 2: Transport Latency Comparison

#### STDIO Transport Server
```python
# Save as: C:\MCP_Benchmarks\servers\stdio_latency_server.py

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
import asyncio
import time

server = Server("stdio-latency-test")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="ping",
            description="Simple ping for latency measurement",
            inputSchema={"type": "object", "properties": {"timestamp": {"type": "number"}}}
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "ping":
        client_time = arguments.get("timestamp", 0)
        server_time = time.time() * 1000
        return [TextContent(type="text", text=f'{{"server_time": {server_time}, "client_time": {client_time}}}')]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

#### HTTP Transport Server
```python
# Save as: C:\MCP_Benchmarks\servers\http_latency_server.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
import uvicorn

app = FastAPI()

@app.post("/mcp/tools/ping")
async def ping(request: Request):
    data = await request.json()
    client_time = data.get("timestamp", 0)
    server_time = time.time() * 1000
    
    return JSONResponse({
        "result": {
            "server_time": server_time,
            "client_time": client_time,
            "processing_time_ms": server_time - client_time
        }
    })

@app.get("/mcp/tools")
async def list_tools():
    return {"tools": [{"name": "ping", "description": "Latency test"}]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
```

#### Latency Benchmark Client
```python
# Save as: C:\MCP_Benchmarks\latency_benchmark.py

import asyncio
import aiohttp
import subprocess
import time
import statistics
import json

async def benchmark_http(url: str, iterations: int = 100):
    """Benchmark HTTP transport latency"""
    latencies = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(iterations):
            start = time.perf_counter()
            async with session.post(
                f"{url}/mcp/tools/ping",
                json={"timestamp": time.time() * 1000}
            ) as response:
                await response.json()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    return latencies

def benchmark_stdio(server_script: str, iterations: int = 100):
    """Benchmark STDIO transport latency"""
    latencies = []
    
    # Start server process
    process = subprocess.Popen(
        ["python", server_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    for i in range(iterations):
        start = time.perf_counter()
        
        # Send JSON-RPC request
        request = json.dumps({
            "jsonrpc": "2.0",
            "id": i,
            "method": "tools/call",
            "params": {"name": "ping", "arguments": {"timestamp": time.time() * 1000}}
        }) + "\n"
        
        process.stdin.write(request)
        process.stdin.flush()
        
        # Read response
        response = process.stdout.readline()
        
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    process.terminate()
    return latencies

def analyze_latencies(latencies: list, name: str):
    """Analyze and report latency statistics"""
    print(f"\n=== {name} Latency Analysis ===")
    print(f"Iterations: {len(latencies)}")
    print(f"Mean: {statistics.mean(latencies):.3f} ms")
    print(f"Median: {statistics.median(latencies):.3f} ms")
    print(f"Std Dev: {statistics.stdev(latencies):.3f} ms")
    print(f"Min: {min(latencies):.3f} ms")
    print(f"Max: {max(latencies):.3f} ms")
    print(f"P95: {sorted(latencies)[int(len(latencies) * 0.95)]:.3f} ms")
    print(f"P99: {sorted(latencies)[int(len(latencies) * 0.99)]:.3f} ms")
    
    return {
        "name": name,
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "std": statistics.stdev(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "p95": sorted(latencies)[int(len(latencies) * 0.95)],
        "p99": sorted(latencies)[int(len(latencies) * 0.99)]
    }

async def main():
    results = []
    
    # Benchmark HTTP
    print("Starting HTTP benchmark...")
    http_latencies = await benchmark_http("http://127.0.0.1:8080", iterations=100)
    results.append(analyze_latencies(http_latencies, "HTTP Transport"))
    
    # Save results
    with open("latency_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to latency_results.json")

if __name__ == "__main__":
    asyncio.run(main())
```

### 4.3 Benchmark 3: Serialization Overhead

```python
# Save as: C:\MCP_Benchmarks\serialization_benchmark.py

import json
import time
import statistics
import sys

# Optional: Install msgpack and protobuf for comparison
# pip install msgpack protobuf

def benchmark_json_serialization(data: dict, iterations: int = 10000):
    """Benchmark JSON serialization/deserialization"""
    
    # Serialization
    serialize_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        serialized = json.dumps(data)
        serialize_times.append(time.perf_counter() - start)
    
    # Deserialization
    deserialize_times = []
    json_str = json.dumps(data)
    for _ in range(iterations):
        start = time.perf_counter()
        json.loads(json_str)
        deserialize_times.append(time.perf_counter() - start)
    
    return {
        "format": "JSON",
        "size_bytes": len(json_str.encode('utf-8')),
        "serialize_mean_us": statistics.mean(serialize_times) * 1_000_000,
        "deserialize_mean_us": statistics.mean(deserialize_times) * 1_000_000,
    }

def benchmark_msgpack_serialization(data: dict, iterations: int = 10000):
    """Benchmark MessagePack serialization"""
    try:
        import msgpack
    except ImportError:
        return {"format": "MessagePack", "error": "Not installed"}
    
    serialize_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        serialized = msgpack.packb(data)
        serialize_times.append(time.perf_counter() - start)
    
    deserialize_times = []
    packed = msgpack.packb(data)
    for _ in range(iterations):
        start = time.perf_counter()
        msgpack.unpackb(packed)
        deserialize_times.append(time.perf_counter() - start)
    
    return {
        "format": "MessagePack",
        "size_bytes": len(packed),
        "serialize_mean_us": statistics.mean(serialize_times) * 1_000_000,
        "deserialize_mean_us": statistics.mean(deserialize_times) * 1_000_000,
    }

def create_test_payloads():
    """Create test payloads of varying complexity"""
    
    # Small payload (simple tool call)
    small = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "get_file", "arguments": {"path": "/test.txt"}}
    }
    
    # Medium payload (tool response with data)
    medium = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {"type": "text", "text": "x" * 1000}  # 1KB of text
            ],
            "metadata": {
                "tokens": 150,
                "model": "claude-3",
                "timestamp": "2025-01-05T12:00:00Z"
            }
        }
    }
    
    # Large payload (tool definitions)
    large = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {
                    "name": f"tool_{i}",
                    "description": f"Description for tool {i} " * 20,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            f"param_{j}": {"type": "string", "description": f"Parameter {j}"}
                            for j in range(10)
                        }
                    }
                }
                for i in range(50)  # 50 tools
            ]
        }
    }
    
    return {"small": small, "medium": medium, "large": large}

def main():
    payloads = create_test_payloads()
    results = []
    
    for size_name, payload in payloads.items():
        print(f"\n=== Benchmarking {size_name.upper()} payload ===")
        
        json_result = benchmark_json_serialization(payload)
        json_result["payload_size"] = size_name
        results.append(json_result)
        print(f"JSON: {json_result['size_bytes']} bytes, "
              f"serialize: {json_result['serialize_mean_us']:.2f}µs, "
              f"deserialize: {json_result['deserialize_mean_us']:.2f}µs")
        
        msgpack_result = benchmark_msgpack_serialization(payload)
        if "error" not in msgpack_result:
            msgpack_result["payload_size"] = size_name
            results.append(msgpack_result)
            print(f"MessagePack: {msgpack_result['size_bytes']} bytes, "
                  f"serialize: {msgpack_result['serialize_mean_us']:.2f}µs, "
                  f"deserialize: {msgpack_result['deserialize_mean_us']:.2f}µs")
            
            # Calculate overhead
            overhead = (json_result['size_bytes'] - msgpack_result['size_bytes']) / msgpack_result['size_bytes'] * 100
            print(f"JSON overhead vs MessagePack: {overhead:.1f}%")
    
    # Save results
    with open("serialization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to serialization_results.json")

if __name__ == "__main__":
    main()
```

### 4.4 Benchmark 4: Caching Effectiveness

```python
# Save as: C:\MCP_Benchmarks\caching_benchmark.py

import time
import statistics
import json
from functools import lru_cache
import hashlib

# Simulated expensive operation
def expensive_database_query(query: str) -> dict:
    """Simulate expensive database query (100ms)"""
    time.sleep(0.1)  # Simulate latency
    return {
        "query": query,
        "results": [{"id": i, "data": f"result_{i}"} for i in range(10)],
        "timestamp": time.time()
    }

# In-memory cache implementation
class SimpleCache:
    def __init__(self, ttl_seconds: int = 60):
        self._cache = {}
        self._ttl = ttl_seconds
    
    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str):
        hashed = self._hash_key(key)
        if hashed in self._cache:
            value, timestamp = self._cache[hashed]
            if time.time() - timestamp < self._ttl:
                return value
            del self._cache[hashed]
        return None
    
    def set(self, key: str, value):
        hashed = self._hash_key(key)
        self._cache[hashed] = (value, time.time())

cache = SimpleCache(ttl_seconds=300)

def cached_query(query: str) -> dict:
    """Query with caching"""
    cached_result = cache.get(query)
    if cached_result:
        return cached_result
    
    result = expensive_database_query(query)
    cache.set(query, result)
    return result

def benchmark_without_cache(queries: list, iterations: int = 5):
    """Benchmark without caching"""
    times = []
    for _ in range(iterations):
        for query in queries:
            start = time.perf_counter()
            expensive_database_query(query)
            times.append(time.perf_counter() - start)
    return times

def benchmark_with_cache(queries: list, iterations: int = 5):
    """Benchmark with caching"""
    cache._cache.clear()  # Clear cache
    times = []
    
    for iteration in range(iterations):
        for query in queries:
            start = time.perf_counter()
            cached_query(query)
            times.append(time.perf_counter() - start)
    
    return times

def main():
    # Test queries (some repeated to test cache hits)
    queries = [
        "SELECT * FROM users WHERE id = 1",
        "SELECT * FROM users WHERE id = 2",
        "SELECT * FROM users WHERE id = 1",  # Repeat
        "SELECT * FROM orders WHERE user_id = 1",
        "SELECT * FROM users WHERE id = 1",  # Repeat
        "SELECT * FROM products WHERE category = 'electronics'",
        "SELECT * FROM orders WHERE user_id = 1",  # Repeat
    ]
    
    print("=== Caching Effectiveness Benchmark ===\n")
    
    # Benchmark without cache
    print("Running without cache...")
    no_cache_times = benchmark_without_cache(queries, iterations=3)
    
    # Benchmark with cache
    print("Running with cache...")
    with_cache_times = benchmark_with_cache(queries, iterations=3)
    
    # Analyze results
    print("\n=== Results ===")
    print(f"\nWithout Cache:")
    print(f"  Total time: {sum(no_cache_times)*1000:.2f} ms")
    print(f"  Mean per query: {statistics.mean(no_cache_times)*1000:.2f} ms")
    
    print(f"\nWith Cache:")
    print(f"  Total time: {sum(with_cache_times)*1000:.2f} ms")
    print(f"  Mean per query: {statistics.mean(with_cache_times)*1000:.2f} ms")
    
    # Calculate improvement
    improvement = (sum(no_cache_times) - sum(with_cache_times)) / sum(no_cache_times) * 100
    speedup = sum(no_cache_times) / sum(with_cache_times)
    
    print(f"\n=== Improvement ===")
    print(f"  Time reduction: {improvement:.1f}%")
    print(f"  Speedup factor: {speedup:.1f}x")
    
    # Cache statistics
    unique_queries = len(set(queries))
    total_queries = len(queries) * 3  # iterations
    cache_hits = total_queries - unique_queries * 3
    hit_rate = cache_hits / total_queries * 100
    
    print(f"\n=== Cache Statistics ===")
    print(f"  Unique queries: {unique_queries}")
    print(f"  Total queries: {total_queries}")
    print(f"  Estimated cache hits: {cache_hits}")
    print(f"  Hit rate: {hit_rate:.1f}%")
    
    # Save results
    results = {
        "without_cache": {
            "total_ms": sum(no_cache_times) * 1000,
            "mean_ms": statistics.mean(no_cache_times) * 1000
        },
        "with_cache": {
            "total_ms": sum(with_cache_times) * 1000,
            "mean_ms": statistics.mean(with_cache_times) * 1000
        },
        "improvement_percent": improvement,
        "speedup_factor": speedup,
        "cache_hit_rate": hit_rate
    }
    
    with open("caching_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```

### 4.5 Benchmark 5: Energy Consumption Measurement

```python
# Save as: C:\MCP_Benchmarks\energy_benchmark.py

import subprocess
import time
import json
import os
import statistics

def get_cpu_power_estimate():
    """
    Estimate CPU power using Windows Performance Counters
    Note: For accurate measurements, use Intel Power Gadget or HWiNFO
    """
    try:
        result = subprocess.run(
            ['powershell', '-Command', 
             "(Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples[0].CookedValue"],
            capture_output=True, text=True, timeout=5
        )
        cpu_percent = float(result.stdout.strip())
        
        # Rough estimate: Assume TDP of 65W, scale by usage
        # This is a ROUGH estimate - use proper power monitoring tools for accuracy
        tdp_watts = 65  # Adjust based on your CPU
        estimated_power = (cpu_percent / 100) * tdp_watts
        
        return cpu_percent, estimated_power
    except Exception as e:
        return None, None

def run_workload(workload_func, duration_seconds: int = 30):
    """Run a workload and measure power consumption"""
    measurements = []
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    # Start workload in background
    import threading
    workload_running = True
    
    def workload_thread():
        while workload_running:
            workload_func()
    
    thread = threading.Thread(target=workload_thread)
    thread.start()
    
    # Collect measurements
    while time.time() < end_time:
        cpu_percent, power_watts = get_cpu_power_estimate()
        if cpu_percent is not None:
            measurements.append({
                "timestamp": time.time() - start_time,
                "cpu_percent": cpu_percent,
                "estimated_power_watts": power_watts
            })
        time.sleep(0.5)
    
    workload_running = False
    thread.join(timeout=5)
    
    return measurements

def idle_workload():
    """Baseline idle workload"""
    time.sleep(0.1)

def light_workload():
    """Light JSON processing workload"""
    data = {"items": [{"id": i, "value": f"item_{i}"} for i in range(100)]}
    for _ in range(100):
        json.dumps(data)
        json.loads(json.dumps(data))
    time.sleep(0.01)

def heavy_workload():
    """Heavy computation workload"""
    # Simulate heavy serialization/computation
    data = {"items": [{"id": i, "value": f"item_{i}" * 100} for i in range(1000)]}
    for _ in range(10):
        serialized = json.dumps(data)
        json.loads(serialized)

def analyze_measurements(measurements: list, name: str):
    """Analyze energy measurements"""
    if not measurements:
        return None
    
    cpu_values = [m["cpu_percent"] for m in measurements]
    power_values = [m["estimated_power_watts"] for m in measurements]
    
    duration = measurements[-1]["timestamp"] - measurements[0]["timestamp"]
    avg_power = statistics.mean(power_values)
    energy_wh = (avg_power * duration) / 3600  # Convert to Watt-hours
    
    return {
        "name": name,
        "duration_seconds": duration,
        "avg_cpu_percent": statistics.mean(cpu_values),
        "avg_power_watts": avg_power,
        "peak_power_watts": max(power_values),
        "energy_wh": energy_wh,
        "energy_joules": avg_power * duration
    }

def main():
    print("=== Energy Consumption Benchmark ===")
    print("Note: For accurate power measurements, use Intel Power Gadget or HWiNFO")
    print("These are rough estimates based on CPU utilization.\n")
    
    results = []
    
    # Baseline (idle)
    print("Measuring baseline (idle)...")
    idle_measurements = run_workload(idle_workload, duration_seconds=15)
    idle_result = analyze_measurements(idle_measurements, "Idle Baseline")
    if idle_result:
        results.append(idle_result)
        print(f"  Avg CPU: {idle_result['avg_cpu_percent']:.1f}%")
        print(f"  Avg Power: {idle_result['avg_power_watts']:.1f}W")
    
    # Light workload
    print("\nMeasuring light workload (simple JSON operations)...")
    light_measurements = run_workload(light_workload, duration_seconds=15)
    light_result = analyze_measurements(light_measurements, "Light Workload")
    if light_result:
        results.append(light_result)
        print(f"  Avg CPU: {light_result['avg_cpu_percent']:.1f}%")
        print(f"  Avg Power: {light_result['avg_power_watts']:.1f}W")
    
    # Heavy workload
    print("\nMeasuring heavy workload (intensive JSON operations)...")
    heavy_measurements = run_workload(heavy_workload, duration_seconds=15)
    heavy_result = analyze_measurements(heavy_measurements, "Heavy Workload")
    if heavy_result:
        results.append(heavy_result)
        print(f"  Avg CPU: {heavy_result['avg_cpu_percent']:.1f}%")
        print(f"  Avg Power: {heavy_result['avg_power_watts']:.1f}W")
    
    # Calculate energy overhead
    if idle_result and heavy_result:
        overhead = heavy_result['avg_power_watts'] - idle_result['avg_power_watts']
        print(f"\n=== Energy Overhead ===")
        print(f"Heavy workload overhead: {overhead:.1f}W above idle")
        print(f"Percentage increase: {(overhead/idle_result['avg_power_watts'])*100:.1f}%")
    
    # Save results
    with open("energy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to energy_results.json")
    print("\nFor publication-quality measurements, consider:")
    print("1. Using Intel Power Gadget for actual CPU power readings")
    print("2. Using a hardware power meter (e.g., Kill-A-Watt)")
    print("3. Running longer duration tests (5-10 minutes)")
    print("4. Multiple runs with statistical analysis")

if __name__ == "__main__":
    main()
```

### 4.6 Benchmark 6: Multi-Server Scalability

```python
# Save as: C:\MCP_Benchmarks\scalability_benchmark.py

import asyncio
import time
import statistics
import json

class MockMCPServer:
    """Mock MCP server for scalability testing"""
    def __init__(self, server_id: int, num_tools: int = 10):
        self.server_id = server_id
        self.tools = [
            {
                "name": f"server_{server_id}_tool_{i}",
                "description": f"Tool {i} from server {server_id}",
                "inputSchema": {"type": "object", "properties": {}}
            }
            for i in range(num_tools)
        ]
    
    async def initialize(self):
        """Simulate initialization handshake"""
        await asyncio.sleep(0.01)  # Simulate network latency
        return self.tools
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Simulate tool invocation"""
        await asyncio.sleep(0.005)  # Simulate processing
        return {"result": f"Response from {tool_name}"}

async def benchmark_initialization(num_servers: int):
    """Benchmark initialization time with multiple servers"""
    servers = [MockMCPServer(i) for i in range(num_servers)]
    
    start = time.perf_counter()
    
    # Initialize all servers concurrently
    all_tools = await asyncio.gather(*[s.initialize() for s in servers])
    
    end = time.perf_counter()
    
    total_tools = sum(len(tools) for tools in all_tools)
    
    return {
        "num_servers": num_servers,
        "total_tools": total_tools,
        "init_time_ms": (end - start) * 1000
    }

async def benchmark_tool_selection(num_servers: int, tools_per_server: int = 10):
    """Benchmark time to find and invoke a tool"""
    servers = [MockMCPServer(i, tools_per_server) for i in range(num_servers)]
    
    # Collect all tools
    all_tools = []
    for s in servers:
        tools = await s.initialize()
        for t in tools:
            all_tools.append((s, t))
    
    # Simulate tool selection (linear search - worst case)
    target_tool = f"server_{num_servers-1}_tool_{tools_per_server-1}"
    
    times = []
    for _ in range(100):
        start = time.perf_counter()
        
        # Find tool (linear search)
        for server, tool in all_tools:
            if tool["name"] == target_tool:
                await server.call_tool(tool["name"], {})
                break
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "num_servers": num_servers,
        "total_tools": len(all_tools),
        "mean_selection_time_ms": statistics.mean(times),
        "p99_selection_time_ms": sorted(times)[99]
    }

async def benchmark_context_window_usage(num_servers: int, tools_per_server: int = 10):
    """Estimate context window tokens for tool definitions"""
    servers = [MockMCPServer(i, tools_per_server) for i in range(num_servers)]
    
    all_tools = []
    for s in servers:
        tools = await s.initialize()
        all_tools.extend(tools)
    
    # Estimate tokens (rough: 4 chars per token)
    json_str = json.dumps(all_tools)
    estimated_tokens = len(json_str) / 4
    
    return {
        "num_servers": num_servers,
        "total_tools": len(all_tools),
        "json_bytes": len(json_str),
        "estimated_tokens": int(estimated_tokens),
        "context_percentage": (estimated_tokens / 128000) * 100  # Assume 128K context
    }

async def main():
    print("=== MCP Server Scalability Benchmark ===\n")
    
    server_counts = [1, 5, 10, 25, 50, 100]
    
    results = {
        "initialization": [],
        "tool_selection": [],
        "context_usage": []
    }
    
    for num_servers in server_counts:
        print(f"Testing with {num_servers} servers...")
        
        # Initialization benchmark
        init_result = await benchmark_initialization(num_servers)
        results["initialization"].append(init_result)
        
        # Tool selection benchmark
        selection_result = await benchmark_tool_selection(num_servers)
        results["tool_selection"].append(selection_result)
        
        # Context usage estimate
        context_result = await benchmark_context_window_usage(num_servers)
        results["context_usage"].append(context_result)
    
    # Print summary
    print("\n=== Initialization Time ===")
    print(f"{'Servers':<10} {'Tools':<10} {'Time (ms)':<15}")
    for r in results["initialization"]:
        print(f"{r['num_servers']:<10} {r['total_tools']:<10} {r['init_time_ms']:<15.2f}")
    
    print("\n=== Tool Selection Time ===")
    print(f"{'Servers':<10} {'Tools':<10} {'Mean (ms)':<15} {'P99 (ms)':<15}")
    for r in results["tool_selection"]:
        print(f"{r['num_servers']:<10} {r['total_tools']:<10} {r['mean_selection_time_ms']:<15.2f} {r['p99_selection_time_ms']:<15.2f}")
    
    print("\n=== Context Window Usage ===")
    print(f"{'Servers':<10} {'Tools':<10} {'Est. Tokens':<15} {'% of 128K':<15}")
    for r in results["context_usage"]:
        print(f"{r['num_servers']:<10} {r['total_tools']:<10} {r['estimated_tokens']:<15} {r['context_percentage']:<15.1f}")
    
    # Save results
    with open("scalability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to scalability_results.json")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Data Collection Templates

### 5.1 Benchmark Run Log Template

```markdown
# MCP Benchmark Run Log

## Run Information
- **Date:** YYYY-MM-DD HH:MM
- **Run ID:** [unique identifier]
- **Operator:** [name]

## System Configuration
- **OS:** Windows 11 Pro 23H2
- **CPU:** [model] @ [speed] GHz, [cores] cores
- **RAM:** [size] GB DDR[version]
- **Storage:** [type] [size]
- **Power Mode:** [balanced/performance]

## Software Versions
- **Python:** 3.12.x
- **Node.js:** 20.x.x
- **MCP SDK:** x.x.x
- **Docker:** x.x.x

## Test Parameters
- **Warmup Iterations:** [number]
- **Test Iterations:** [number]
- **Cooldown Period:** [seconds]

## Results Summary
| Benchmark | Metric | Value | Unit |
|-----------|--------|-------|------|
| Token Consumption | Verbose | | tokens |
| Token Consumption | Optimized | | tokens |
| Latency | STDIO | | ms |
| Latency | HTTP | | ms |
| Serialization | JSON | | µs |
| Caching | Hit Rate | | % |
| Energy | Idle | | W |
| Energy | Load | | W |

## Observations
[Notes about anomalies, issues, or interesting findings]
```

### 5.2 Results CSV Template

```csv
run_id,timestamp,benchmark_type,metric_name,value,unit,notes
001,2025-01-05T10:00:00,token_consumption,verbose_tokens,150000,tokens,
001,2025-01-05T10:00:00,token_consumption,optimized_tokens,2000,tokens,
001,2025-01-05T10:05:00,latency,stdio_mean,5.2,ms,
001,2025-01-05T10:05:00,latency,http_mean,12.4,ms,
```

---

## 6. Analysis Methods

### 6.1 Statistical Analysis Script

```python
# Save as: C:\MCP_Benchmarks\analyze_results.py

import json
import statistics
import os
from pathlib import Path

def load_results(results_dir: str = "."):
    """Load all result files"""
    results = {}
    
    for filename in ["latency_results.json", "serialization_results.json", 
                     "caching_results.json", "energy_results.json",
                     "scalability_results.json"]:
        filepath = Path(results_dir) / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                results[filename.replace("_results.json", "")] = json.load(f)
    
    return results

def generate_latex_table(results: dict):
    """Generate LaTeX table for paper"""
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Empirical Benchmark Results}
\label{tab:benchmarks}
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{Optimized} & \textbf{Improvement} \\
\midrule
"""
    
    # Add rows based on available data
    if "caching" in results:
        cache = results["caching"]
        latex += f"Response Time (ms) & {cache['without_cache']['mean_ms']:.2f} & {cache['with_cache']['mean_ms']:.2f} & {cache['improvement_percent']:.1f}\\% \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex

def main():
    results = load_results()
    
    print("=== Benchmark Analysis ===\n")
    
    for category, data in results.items():
        print(f"\n--- {category.upper()} ---")
        print(json.dumps(data, indent=2))
    
    # Generate LaTeX
    latex = generate_latex_table(results)
    print("\n=== LaTeX Table ===")
    print(latex)
    
    # Save LaTeX
    with open("benchmark_table.tex", "w") as f:
        f.write(latex)

if __name__ == "__main__":
    main()
```

### 6.2 Visualization Script

```python
# Save as: C:\MCP_Benchmarks\visualize_results.py

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_scalability(results_file: str = "scalability_results.json"):
    """Generate scalability charts"""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initialization time
    servers = [r['num_servers'] for r in data['initialization']]
    init_times = [r['init_time_ms'] for r in data['initialization']]
    
    axes[0].plot(servers, init_times, 'b-o')
    axes[0].set_xlabel('Number of Servers')
    axes[0].set_ylabel('Initialization Time (ms)')
    axes[0].set_title('MCP Initialization Scalability')
    axes[0].grid(True)
    
    # Tool selection time
    selection_times = [r['mean_selection_time_ms'] for r in data['tool_selection']]
    
    axes[1].plot(servers, selection_times, 'r-o')
    axes[1].set_xlabel('Number of Servers')
    axes[1].set_ylabel('Selection Time (ms)')
    axes[1].set_title('Tool Selection Latency')
    axes[1].grid(True)
    
    # Context usage
    context_pct = [r['context_percentage'] for r in data['context_usage']]
    
    axes[2].bar(range(len(servers)), context_pct)
    axes[2].set_xticks(range(len(servers)))
    axes[2].set_xticklabels(servers)
    axes[2].set_xlabel('Number of Servers')
    axes[2].set_ylabel('Context Window Usage (%)')
    axes[2].set_title('Context Window Consumption')
    axes[2].axhline(y=50, color='r', linestyle='--', label='50% threshold')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('scalability_charts.png', dpi=300)
    plt.savefig('scalability_charts.pdf')
    print("Charts saved to scalability_charts.png/pdf")

def plot_comparison_bar(data: dict, output_file: str = "comparison_chart.png"):
    """Generate comparison bar chart"""
    
    categories = list(data.keys())
    baseline = [data[c]['baseline'] for c in categories]
    optimized = [data[c]['optimized'] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline', color='#ff6b6b')
    bars2 = ax.bar(x + width/2, optimized, width, label='Optimized', color='#4ecdc4')
    
    ax.set_ylabel('Value')
    ax.set_title('MCP Optimization Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Chart saved to {output_file}")

if __name__ == "__main__":
    # Install matplotlib if needed: pip install matplotlib
    
    if Path("scalability_results.json").exists():
        plot_scalability()
    else:
        print("Run scalability_benchmark.py first to generate data")
```

---

## 7. Paper Integration Guidelines

### 7.1 Adding Benchmarks Section to Your Paper

Add this section after your existing "Comparative Performance Assessment" section:

```latex
\section{Empirical Validation}

\subsection{Experimental Setup}

To validate theoretical findings from the literature, we conducted empirical benchmarks on a controlled test environment. Experiments were performed on a system with [CPU model], [RAM] GB RAM, running Windows 11. All tests were repeated [N] times with results averaged.

\subsection{Token Consumption Validation}

We implemented prototype MCP servers following both traditional verbose response patterns and progressive disclosure architectures. Token counts were measured using the tiktoken library with the cl100k\_base encoding.

\begin{table}[h]
\centering
\caption{Token Consumption Comparison}
\label{tab:token_benchmark}
\begin{tabular}{lrr}
\toprule
\textbf{Scenario} & \textbf{Tokens} & \textbf{Reduction} \\
\midrule
Verbose Response (100 items) & X,XXX & -- \\
Progressive Disclosure & XXX & XX.X\% \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Transport Latency Analysis}

Latency measurements comparing STDIO and HTTP transport mechanisms revealed [findings].

\subsection{Caching Effectiveness}

Response caching demonstrated [X]x performance improvement with a [Y]\% cache hit rate for repeated queries, validating recommendations from the optimization literature \cite{catchmetrics2025}.

\subsection{Scalability Observations}

Testing with increasing numbers of MCP servers (1 to 100) revealed [observations about context window consumption and initialization overhead].
```

### 7.2 Adding Figures

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{scalability_charts.pdf}
\caption{MCP Server Scalability Benchmarks: (a) Initialization time scales linearly with server count, (b) Tool selection latency increases with tool registry size, (c) Context window consumption becomes critical beyond 50 servers.}
\label{fig:scalability}
\end{figure}
```

### 7.3 Updating Abstract and Contributions

Update your abstract to mention empirical validation:

```latex
"...We validate theoretical findings through empirical benchmarks demonstrating [key finding 1] and [key finding 2]..."
```

Add to your contributions:

```latex
"...(5) empirical validation of optimization strategies through controlled benchmarks..."
```

---

## 8. Quick Start Commands

Run all benchmarks in sequence:

```powershell
# Navigate to benchmark directory
cd C:\MCP_Benchmarks

# Install dependencies
pip install tiktoken aiohttp matplotlib

# Run benchmarks
python serialization_benchmark.py
python caching_benchmark.py
python energy_benchmark.py
python scalability_benchmark.py

# Analyze and visualize
python analyze_results.py
python visualize_results.py
```

---

## 9. Advanced Benchmarks (Optional)

### 9.1 Real MCP Server Testing with Claude Desktop

Configure Claude Desktop to use your test servers:

```json
// %APPDATA%\Claude\claude_desktop_config.json
{
  "mcpServers": {
    "verbose-test": {
      "command": "python",
      "args": ["C:\\MCP_Benchmarks\\servers\\verbose_server.py"]
    },
    "optimized-test": {
      "command": "python", 
      "args": ["C:\\MCP_Benchmarks\\servers\\optimized_server.py"]
    }
  }
}
```

### 9.2 Docker-Based Solver Benchmarks

```dockerfile
# Save as: C:\MCP_Benchmarks\solver\Dockerfile
FROM python:3.12-slim

RUN pip install mcp ortools z3-solver

COPY solver_server.py /app/
WORKDIR /app

CMD ["python", "solver_server.py"]
```

### 9.3 Network Capture for Protocol Analysis

```powershell
# Capture MCP traffic with Wireshark
# Filter: tcp.port == 8080
# Export: Statistics > Protocol Hierarchy
```

---

## 10. Checklist for Paper Submission

- [ ] Run all benchmarks minimum 3 times
- [ ] Document system specifications
- [ ] Calculate statistical significance (p-values if comparing groups)
- [ ] Generate publication-quality figures (300 DPI, PDF format)
- [ ] Update paper with empirical validation section
- [ ] Add benchmark methodology to reproducibility appendix
- [ ] Cite this benchmarking methodology in your paper
- [ ] Make benchmark code available (GitHub repository)

---

## Contact and Support

For issues with this benchmarking guide, consult:
- MCP SDK Documentation: https://modelcontextprotocol.io
- Python MCP: https://github.com/modelcontextprotocol/python-sdk
- TypeScript MCP: https://github.com/modelcontextprotocol/typescript-sdk
