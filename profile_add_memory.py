#!/usr/bin/env python3
"""
Profile the add_memory operation to identify performance bottlenecks
"""

import time
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys
import cProfile
import pstats
from io import StringIO

async def profile_add_memory():
    """Profile a single add_memory operation"""
    
    server_params = StdioServerParameters(
        command="python3",
        args=["memoryos-mcp/server_new.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # Get available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")
            
            # Test data
            test_conversation = {
                "user_input": "What's the weather like today?",
                "agent_response": "I don't have access to real-time weather data, but I can help you find weather information if you tell me your location."
            }
            
            print("\n=== Starting add_memory profiling ===")
            
            # Measure the time for add_memory operation
            start_time = time.time()
            
            result = await session.call_tool("add_memory", test_conversation)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print(f"\nElapsed time: {elapsed_time:.2f} seconds")
            print(f"Result: {result}")
            
            if elapsed_time > 5:
                print(f"\n⚠️ WARNING: add_memory took {elapsed_time:.2f} seconds, which is too slow!")
            
            return elapsed_time

async def profile_multiple_adds():
    """Profile multiple add_memory operations to see if there's a pattern"""
    
    server_params = StdioServerParameters(
        command="python3",
        args=["memoryos-mcp/server_new.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            conversations = [
                {
                    "user_input": f"Question {i}: What is {i} + {i}?",
                    "agent_response": f"The answer is {i*2}."
                }
                for i in range(1, 16)  # Test with 15 conversations
            ]
            
            times = []
            
            for i, conv in enumerate(conversations, 1):
                print(f"\nAdding conversation {i}/15...")
                start_time = time.time()
                
                result = await session.call_tool("add_memory", conv)
                
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
                
                print(f"  Time: {elapsed_time:.2f}s")
                
                if i == 10:
                    print("\n  ⚠️ Short-term memory should be full now (capacity=10)")
                    print("  Next add should trigger processing to mid-term memory")
            
            print("\n=== Summary ===")
            print(f"Average time: {sum(times)/len(times):.2f}s")
            print(f"Min time: {min(times):.2f}s")
            print(f"Max time: {max(times):.2f}s")
            print(f"Times for each add: {[f'{t:.2f}s' for t in times]}")
            
            # Identify slow operations
            slow_ops = [(i+1, t) for i, t in enumerate(times) if t > 5]
            if slow_ops:
                print(f"\n⚠️ Slow operations (>5s):")
                for idx, t in slow_ops:
                    print(f"  - Operation {idx}: {t:.2f}s")

if __name__ == "__main__":
    print("Starting MemoryOS add_memory profiling...")
    print("=" * 50)
    
    # Run single test
    print("\n1. Testing single add_memory operation:")
    asyncio.run(profile_add_memory())
    
    # Run multiple tests to see pattern
    print("\n" + "=" * 50)
    print("2. Testing multiple add_memory operations:")
    asyncio.run(profile_multiple_adds())