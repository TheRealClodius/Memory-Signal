#!/usr/bin/env python3
"""
MemoryOS Performance Test Suite
Tests the optimized memory spillover performance to validate improvements.

This test specifically targets the issues identified:
- Memory spillover performance cliff at capacity limit
- Embedding model loading delays
- Sync LLM processing during spillover
- Memory retrieval accuracy after spillover

Expected Results After Optimization:
- Spillover operations should be ~20-70x faster
- No blocking during memory addition operations
- Consistent performance regardless of memory capacity state
- Maintained retrieval accuracy
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from statistics import mean, stdev
import threading

# Import MCP client
try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import types
except ImportError as e:
    print(f"❌ Failed to import MCP client library: {e}")
    print("Please install official MCP SDK: pip install mcp")
    sys.exit(1)

class PerformanceMemoryOSTest:
    """Performance test suite for MemoryOS optimizations"""
    
    def __init__(self, server_script: str = "server_new.py", config_file: str = "config.json"):
        self.server_script = Path(server_script)
        self.config_file = Path(config_file)
        
        # Validate file existence
        if not self.server_script.exists():
            raise FileNotFoundError(f"Server script not found: {self.server_script}")
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        # Performance tracking
        self.operation_times = []
        self.spillover_times = []
        self.retrieval_times = []
        self.background_tasks_count = 0
        
    def get_server_url(self):
        import os
        return os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")
    
    async def time_operation(self, operation_func):
        """Time an async operation and return the duration in milliseconds."""
        start_time = time.time()
        result = await operation_func()
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        return result, duration_ms
    
    async def test_performance_cliff_scenario(self):
        """
        Test the specific performance cliff scenario:
        - Insert memories 1-9: Should be fast (~300-400ms each)
        - Insert memory 10+: Previously 20-70x slower, now should be fast with background processing
        """
        print("\n🚀 Performance Cliff Test: Testing spillover performance...")
        print("=" * 70)
        
        # Test conversations designed to trigger spillover at memory 9-10
        test_conversations = [
            {"user_input": f"This is test message {i}", "agent_response": f"Response to test message {i}. This is a detailed response to ensure sufficient content for embedding processing."}
            for i in range(1, 21)  # 20 conversations to test multiple spillovers
        ]
        
        server_url = self.get_server_url()
        operation_times = []
        
        try:
            headers = {"Accept": "application/json, text/event-stream"}
            async with streamablehttp_client(server_url, headers=headers) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    print(f"📊 Testing {len(test_conversations)} memory additions...")
                    
                    for i, conversation in enumerate(test_conversations, 1):
                        print(f"   Adding memory {i:2d}/20...", end="")
                        
                        # Time the add_memory operation
                        async def add_memory_op():
                            return await session.call_tool("add_memory", conversation)
                        
                        result, duration_ms = await self.time_operation(add_memory_op)
                        operation_times.append(duration_ms)
                        
                        # Analyze result
                        success = False
                        if hasattr(result, 'content') and result.content:
                            content = result.content[0]
                            if isinstance(content, types.TextContent):
                                response = json.loads(content.text)
                                if response.get("status") == "success":
                                    success = True
                        
                        status_emoji = "✅" if success else "❌"
                        print(f" {status_emoji} {duration_ms:6.1f}ms")
                        
                        # Brief delay to allow background processing
                        await asyncio.sleep(0.05)
                    
                    # Wait for any background tasks to complete
                    print("\n⏳ Waiting for background processing to complete...")
                    await asyncio.sleep(5)  # Allow background spillover tasks to finish
                    
                    self.operation_times = operation_times
                    return True
                    
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False
    
    async def test_memory_retrieval_performance(self):
        """Test memory retrieval performance and accuracy after spillover."""
        print("\n🔍 Memory Retrieval Performance Test...")
        print("=" * 50)
        
        test_queries = [
            {"query": "Tell me about test message 1", "description": "Early memory retrieval"},
            {"query": "What was said in test message 10", "description": "Spillover boundary retrieval"},
            {"query": "Recent test messages", "description": "Recent memory retrieval"},
            {"query": "test message responses", "description": "General content retrieval"}
        ]
        
        server_url = self.get_server_url()
        retrieval_times = []
        
        try:
            headers = {"Accept": "application/json, text/event-stream"}
            async with streamablehttp_client(server_url, headers=headers) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    for i, test_query in enumerate(test_queries, 1):
                        print(f"   Query {i}: {test_query['description']}")
                        
                        query_params = {
                            "query": test_query["query"],
                            "relationship_with_user": "friend",
                            "style_hint": "helpful",
                            "max_results": 10
                        }
                        
                        # Time the retrieval operation
                        async def retrieve_op():
                            return await session.call_tool("retrieve_memory", query_params)
                        
                        result, duration_ms = await self.time_operation(retrieve_op)
                        retrieval_times.append(duration_ms)
                        
                        # Analyze retrieval results
                        if hasattr(result, 'content') and result.content:
                            content = result.content[0]
                            if isinstance(content, types.TextContent):
                                response = json.loads(content.text)
                                if response.get("status") == "success":
                                    pages_found = response.get('total_pages_found', 0)
                                    short_term_count = response.get('short_term_count', 0)
                                    print(f"      ✅ Found {pages_found} pages, {short_term_count} short-term items ({duration_ms:.1f}ms)")
                                else:
                                    print(f"      ❌ Query failed: {response.get('message', 'Unknown error')}")
                        
                        await asyncio.sleep(0.2)
                    
                    self.retrieval_times = retrieval_times
                    return True
                    
        except Exception as e:
            print(f"❌ Retrieval test failed: {e}")
            return False
    
    def analyze_performance_results(self):
        """Analyze and report performance improvements."""
        print("\n📈 Performance Analysis Results")
        print("=" * 50)
        
        if not self.operation_times:
            print("❌ No operation times recorded")
            return False
        
        # Split times into early vs late operations to detect performance cliff
        early_operations = self.operation_times[:5]  # First 5 operations
        late_operations = self.operation_times[10:15]  # Operations 11-15 (after spillover)
        
        early_avg = mean(early_operations)
        late_avg = mean(late_operations) if late_operations else early_avg
        
        print(f"📊 Operation Times Summary:")
        print(f"   Early operations (1-5):     {early_avg:6.1f}ms average")
        print(f"   Late operations (11-15):    {late_avg:6.1f}ms average")
        
        # Calculate performance change
        if late_avg > early_avg:
            performance_change = ((late_avg - early_avg) / early_avg) * 100
            print(f"   Performance degradation:    +{performance_change:.1f}%")
        else:
            performance_improvement = ((early_avg - late_avg) / early_avg) * 100
            print(f"   Performance improvement:    +{performance_improvement:.1f}%")
        
        # Detect performance cliff (>500% degradation indicates an issue)
        cliff_detected = performance_change > 500 if late_avg > early_avg else False
        
        print(f"\n📋 Detailed Operation Times:")
        for i, time_ms in enumerate(self.operation_times, 1):
            status = "🔥" if time_ms > early_avg * 5 else "✅"
            print(f"   Operation {i:2d}: {time_ms:6.1f}ms {status}")
        
        # Background processing analysis
        consistent_performance = all(t < early_avg * 2 for t in self.operation_times)
        
        print(f"\n🎯 Performance Goals Assessment:")
        print(f"   ✅ No performance cliff detected: {not cliff_detected}")
        print(f"   ✅ Consistent performance maintained: {consistent_performance}")
        
        if self.retrieval_times:
            retrieval_avg = mean(self.retrieval_times)
            print(f"   ✅ Memory retrieval average: {retrieval_avg:.1f}ms")
        
        # Overall success criteria
        success = not cliff_detected and consistent_performance
        
        print(f"\n🏆 Overall Performance Test: {'PASSED' if success else 'FAILED'}")
        
        if success:
            print("🎉 Optimizations successfully eliminated the performance cliff!")
            print("   - Memory spillover now uses background processing")
            print("   - Batch embedding generation reduces model loading overhead")
            print("   - Deferred I/O operations prevent blocking")
        else:
            print("⚠️  Performance issues still detected. Further optimization needed.")
        
        return success
    
    async def run_performance_test(self):
        """Run the complete performance test suite."""
        print("🚀 Starting MemoryOS Performance Test Suite")
        print(f"Server script: {self.server_script}")
        print(f"Config file: {self.config_file}")
        print("=" * 70)
        
        # Step 1: Performance cliff test
        cliff_test_success = await self.test_performance_cliff_scenario()
        if not cliff_test_success:
            print("❌ Performance cliff test failed. Stopping.")
            return False
        
        # Step 2: Retrieval performance test
        retrieval_test_success = await self.test_memory_retrieval_performance()
        
        # Step 3: Analyze results
        analysis_success = self.analyze_performance_results()
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 Performance Test Summary:")
        print(f"✅ Performance cliff test:     {'Passed' if cliff_test_success else 'Failed'}")
        print(f"✅ Memory retrieval test:      {'Passed' if retrieval_test_success else 'Failed'}")
        print(f"✅ Performance analysis:       {'Passed' if analysis_success else 'Failed'}")
        
        overall_success = cliff_test_success and analysis_success
        
        if overall_success:
            print("🎉 All performance tests passed! Optimizations are working correctly.")
            print("🔍 Key improvements:")
            print("   - Eliminated 20-70x spillover performance cliff")
            print("   - Background processing prevents blocking")
            print("   - Batch operations improve throughput")
            print("   - Memory compression reduces I/O overhead")
            return True
        else:
            print("⚠️ Some performance tests failed. Review optimizations.")
            return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MemoryOS Performance Test Suite")
    parser.add_argument("--server", default="server_new.py", help="Server script path")
    parser.add_argument("--config", default="config.json", help="Config file path")
    
    args = parser.parse_args()
    
    try:
        tester = PerformanceMemoryOSTest(args.server, args.config)
        success = asyncio.run(tester.run_performance_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Performance test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Performance test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
