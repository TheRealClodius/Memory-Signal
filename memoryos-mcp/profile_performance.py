#!/usr/bin/env python3
"""
Detailed performance profiling for MemoryOS add_memory operation
"""

import time
import json
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import timing decorator
def time_function(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"  ⏱️  {func.__name__}: {end - start:.3f}s")
        return result
    return wrapper

# Monkey-patch key functions to add timing
original_imports = {}

def profile_memoryos():
    """Profile MemoryOS initialization and add_memory operation"""
    
    print("=" * 60)
    print("🔬 MemoryOS Performance Profiling")
    print("=" * 60)
    
    # Load config
    config_path = Path("config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("\n📋 Configuration:")
    print(f"  - Short-term capacity: {config['short_term_capacity']}")
    print(f"  - Mid-term capacity: {config['mid_term_capacity']}")
    print(f"  - Embedding model: {config['embedding_model_name']}")
    print(f"  - LLM model: {config['llm_model']}")
    
    # Test with empty API key (will fail LLM calls but show embedding performance)
    if not config.get('openai_api_key'):
        print("\n⚠️  No OpenAI API key configured - LLM calls will fail")
        print("   This is OK for testing embedding performance\n")
    
    # Import and patch MemoryOS components with timing
    print("\n🔧 Instrumenting MemoryOS components...")
    
    from memoryos import utils
    from memoryos import memoryos as memoryos_module
    from memoryos import short_term, mid_term, long_term, updater, retriever
    
    # Patch key functions with timing
    original_get_embedding = utils.get_embedding
    utils.get_embedding = time_function(original_get_embedding)
    
    original_chat_completion = utils.OpenAIClient.chat_completion
    utils.OpenAIClient.chat_completion = time_function(original_chat_completion)
    
    # Initialize MemoryOS
    print("\n📦 Initializing MemoryOS...")
    start_init = time.perf_counter()
    
    from memoryos.memoryos import Memoryos
    memory = Memoryos(
        user_id=config['user_id'],
        openai_api_key=config.get('openai_api_key', ''),
        openai_base_url=config.get('openai_base_url', ''),
        data_storage_path=config.get('data_storage_path', './memoryos_data'),
        assistant_id=config.get('assistant_id', 'assistant'),
        short_term_capacity=config['short_term_capacity'],
        mid_term_capacity=config['mid_term_capacity'],
        embedding_model_name=config['embedding_model_name'],
        long_term_knowledge_capacity=config.get('long_term_knowledge_capacity', 100),
        retrieval_queue_capacity=config.get('retrieval_queue_capacity', 7),
        mid_term_heat_threshold=config.get('mid_term_heat_threshold', 7.0),
        mid_term_similarity_threshold=config.get('mid_term_similarity_threshold', 0.6),
        llm_model=config.get('llm_model', 'gpt-4o-mini')
    )
    
    end_init = time.perf_counter()
    print(f"✅ MemoryOS initialized in {end_init - start_init:.3f}s")
    
    # Test conversations
    conversations = [
        ("Hello, I'm Alice", "Nice to meet you, Alice!"),
        ("I'm interested in Python", "Python is a great language!"),
        ("Tell me about decorators", "Decorators are functions that modify other functions"),
        ("What about async programming?", "Async programming allows concurrent execution"),
        ("I use FastAPI", "FastAPI is excellent for building APIs"),
        ("How about databases?", "Databases are essential for data persistence"),
        ("I prefer PostgreSQL", "PostgreSQL is a powerful relational database"),
        ("What about NoSQL?", "NoSQL databases offer flexibility for unstructured data"),
        ("I'm learning Docker", "Docker is great for containerization"),
        ("And Kubernetes?", "Kubernetes orchestrates container deployments"),
        ("I want to learn ML", "Machine learning is fascinating!"),
        ("Tell me about neural networks", "Neural networks are inspired by the brain"),
    ]
    
    print(f"\n🧪 Testing {len(conversations)} add_memory operations...")
    print(f"   (Short-term capacity: {config['short_term_capacity']})")
    
    times = []
    for i, (user_input, agent_response) in enumerate(conversations, 1):
        print(f"\n📝 Adding memory {i}/{len(conversations)}: '{user_input[:30]}...'")
        
        start = time.perf_counter()
        try:
            memory.add_memory(
                user_input=user_input,
                agent_response=agent_response
            )
            end = time.perf_counter()
            elapsed = end - start
            times.append(elapsed)
            print(f"✅ Memory {i} added in {elapsed:.3f}s")
            
            if i == config['short_term_capacity']:
                print(f"\n⚠️  Short-term memory full (capacity={config['short_term_capacity']})")
                print("   Next add will trigger processing to mid-term...")
        except Exception as e:
            print(f"❌ Error: {e}")
            times.append(0)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("📊 Performance Summary")
    print("=" * 60)
    
    if times:
        valid_times = [t for t in times if t > 0]
        if valid_times:
            print(f"⏱️  Timing Statistics:")
            print(f"   - Average: {sum(valid_times)/len(valid_times):.3f}s")
            print(f"   - Min: {min(valid_times):.3f}s")
            print(f"   - Max: {max(valid_times):.3f}s")
            print(f"   - Total: {sum(valid_times):.3f}s")
            
            # Identify slow operations
            slow_threshold = 5.0
            slow_ops = [(i, t) for i, t in enumerate(times, 1) if t > slow_threshold]
            
            if slow_ops:
                print(f"\n⚠️  Slow operations (>{slow_threshold}s):")
                for idx, t in slow_ops:
                    print(f"   - Operation {idx}: {t:.3f}s")
                    if idx == config['short_term_capacity'] + 1:
                        print(f"     (This triggered mid-term processing)")
            
            # Identify the pattern
            print("\n🔍 Performance Pattern Analysis:")
            
            # First N operations (before memory full)
            if len(valid_times) > config['short_term_capacity']:
                before_full = valid_times[:config['short_term_capacity']]
                after_full = valid_times[config['short_term_capacity']:]
                
                print(f"   - Before memory full (ops 1-{config['short_term_capacity']}): avg {sum(before_full)/len(before_full):.3f}s")
                print(f"   - After memory full (ops {config['short_term_capacity']+1}+): avg {sum(after_full)/len(after_full):.3f}s")
                
                if after_full[0] > sum(before_full)/len(before_full) * 2:
                    print(f"\n   🎯 Operation {config['short_term_capacity']+1} is significantly slower!")
                    print(f"      This is when short-term memory processing happens.")
                    print(f"      It involves:")
                    print(f"      - LLM calls for continuity checking")
                    print(f"      - Embedding generation for similarity search")
                    print(f"      - Session summarization")
    
    print("\n" + "=" * 60)
    print("✅ Profiling complete!")
    
    # Restore original functions
    utils.get_embedding = original_get_embedding
    utils.OpenAIClient.chat_completion = original_chat_completion

if __name__ == "__main__":
    profile_memoryos()