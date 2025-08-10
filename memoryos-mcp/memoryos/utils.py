import time
import uuid
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import inspect
from functools import wraps
from . import prompts
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def clean_reasoning_model_output(text):
    """
    清理推理模型输出中的<think>标签
    适配推理模型（如o1系列）的输出格式
    """
    if not text:
        return text
    
    import re
    # 移除<think>...</think>标签及其内容
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 清理可能产生的多余空白行
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    # 移除开头和结尾的空白
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

# ---- LLM Client (supports OpenAI and Gemini) ----
class LLMClient:
    def __init__(self, provider="openai", api_key=None, base_url=None, gemini_api_key=None, max_workers=5):
        self.provider = provider.lower()
        self.api_key = api_key
        self.gemini_api_key = gemini_api_key
        self.base_url = base_url if base_url else "https://api.openai.com/v1"
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        
        if self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.provider == "gemini":
            try:
                from google import genai
                import os
                # Try multiple API key sources
                api_key = self.gemini_api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
                if api_key:
                    self.genai_client = genai.Client(api_key=api_key)
                else:
                    self.genai_client = genai.Client()  # Will try default env vars
            except ImportError:
                raise ImportError("google-genai package is required for Gemini. Install with: pip install google-genai")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Supported: 'openai', 'gemini'")

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2000):
        print(f"Calling {self.provider.upper()} API. Model: {model}")
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                raw_content = response.choices[0].message.content.strip()
                
            elif self.provider == "gemini":
                # Convert OpenAI message format to Gemini format
                gemini_content = self._convert_to_gemini_format(messages)
                
                response = self.genai_client.models.generate_content(
                    model=model,
                    contents=gemini_content,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )
                raw_content = response.text.strip() if hasattr(response, 'text') and response.text else ""
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # 自动清理推理模型的<think>标签
            cleaned_content = clean_reasoning_model_output(raw_content)
            return cleaned_content
            
        except Exception as e:
            print(f"Error calling {self.provider.upper()} API: {e}")
            # Fallback or error handling
            return "Error: Could not get response from LLM."

    def _convert_to_gemini_format(self, messages):
        """Convert OpenAI message format to Gemini format."""
        if not messages:
            return ""
        
        # Gemini expects a single text prompt, so we'll combine the messages
        combined_text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                combined_text += f"System Instructions: {content}\n\n"
            elif role == "user":
                combined_text += f"User: {content}\n\n"
            elif role == "assistant":
                combined_text += f"Assistant: {content}\n\n"
        
        return combined_text.strip()

    def chat_completion_async(self, model, messages, temperature=0.7, max_tokens=2000):
        """异步版本的chat_completion"""
        return self.executor.submit(self.chat_completion, model, messages, temperature, max_tokens)

    def batch_chat_completion(self, requests):
        """
        并行处理多个LLM请求
        requests: List of dict with keys: model, messages, temperature, max_tokens
        """
        futures = []
        for req in requests:
            future = self.chat_completion_async(
                model=req.get("model", "gpt-4o-mini"),
                messages=req["messages"],
                temperature=req.get("temperature", 0.7),
                max_tokens=req.get("max_tokens", 2000)
            )
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in batch completion: {e}")
                results.append("Error: Could not get response from LLM.")
        
        return results

    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)

# ---- Parallel Processing Utilities ----
def run_parallel_tasks(tasks, max_workers=3):
    """
    并行执行任务列表
    tasks: List of callable functions
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in parallel task: {e}")
                results.append(None)
        return results

# ---- Basic Utilities ----
def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def generate_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def ensure_directory_exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ---- Embedding Utilities ----
_model_cache = {}
_embedding_cache = {}  # 添加embedding缓存

def _get_valid_kwargs(func, kwargs):
    """Helper to filter kwargs for a given function's signature."""
    try:
        sig = inspect.signature(func)
        param_keys = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in param_keys}
    except (ValueError, TypeError):
        # Fallback for functions/methods where signature inspection is not straightforward
        return kwargs

def get_embedding(text, model_name="all-MiniLM-L6-v2", use_cache=True, **kwargs):
    """
    获取文本的embedding向量。
    支持多种主流模型，能自动适应不同库的调用方式。
    - SentenceTransformer模型: e.g., 'all-MiniLM-L6-v2', 'Qwen/Qwen3-Embedding-0.6B'
    - FlagEmbedding模型: e.g., 'BAAI/bge-m3'

    :param text: 输入文本。
    :param model_name: Hugging Face上的模型名称。
    :param use_cache: 是否使用内存缓存。
    :param kwargs: 传递给模型构造函数或encode方法的额外参数。
                   - for Qwen: `model_kwargs`, `tokenizer_kwargs`, `prompt_name="query"`
                   - for BGE-M3: `use_fp16=True`, `max_length=8192`
    :return: 文本的embedding向量 (numpy array)。
    """
    model_config_key = json.dumps({"model_name": model_name, **kwargs}, sort_keys=True)
    
    if use_cache:
        cache_key = f"{model_config_key}::{hash(text)}"
        if cache_key in _embedding_cache:
            return _embedding_cache[cache_key]
    
    # --- Model Loading ---
    model_init_key = json.dumps({"model_name": model_name, **{k:v for k,v in kwargs.items() if k not in ['batch_size', 'max_length']}}, sort_keys=True)
    if model_init_key not in _model_cache:
        print(f"Loading model: {model_name}...")
        if 'bge-m3' in model_name.lower():
            try:
                from FlagEmbedding import BGEM3FlagModel
                init_kwargs = _get_valid_kwargs(BGEM3FlagModel.__init__, kwargs)
                print(f"-> Using BGEM3FlagModel with init kwargs: {init_kwargs}")
                _model_cache[model_init_key] = BGEM3FlagModel(model_name, **init_kwargs)
            except ImportError:
                raise ImportError("Please install FlagEmbedding: 'pip install -U FlagEmbedding' to use bge-m3 model.")
        else: # Default handler for SentenceTransformer-based models (like Qwen, all-MiniLM, etc.)
            try:
                from sentence_transformers import SentenceTransformer
                init_kwargs = _get_valid_kwargs(SentenceTransformer.__init__, kwargs)
                print(f"-> Using SentenceTransformer with init kwargs: {init_kwargs}")
                _model_cache[model_init_key] = SentenceTransformer(model_name, **init_kwargs)
            except ImportError:
                raise ImportError("Please install sentence-transformers: 'pip install -U sentence-transformers' to use this model.")
            
    model = _model_cache[model_init_key]
    
    # --- Encoding ---
    embedding = None
    if 'bge-m3' in model_name.lower():
        encode_kwargs = _get_valid_kwargs(model.encode, kwargs)
        print(f"-> Encoding with BGEM3FlagModel using kwargs: {encode_kwargs}")
        result = model.encode([text], **encode_kwargs)
        embedding = result['dense_vecs'][0]
    else: # Default to SentenceTransformer-based models
        encode_kwargs = _get_valid_kwargs(model.encode, kwargs)
        print(f"-> Encoding with SentenceTransformer using kwargs: {encode_kwargs}")
        embedding = model.encode([text], **encode_kwargs)[0]

    if use_cache:
        cache_key = f"{model_config_key}::{hash(text)}"
        _embedding_cache[cache_key] = embedding
        if len(_embedding_cache) > 10000:
            keys_to_remove = list(_embedding_cache.keys())[:1000]
            for key in keys_to_remove:
                try:
                    del _embedding_cache[key]
                except KeyError:
                    pass
            print("Cleaned embedding cache to prevent memory overflow")
    
    return embedding


def get_batch_embeddings(texts, model_name="all-MiniLM-L6-v2", use_cache=True, **kwargs):
    """
    PERFORMANCE OPTIMIZATION: Batch embedding generation for multiple texts.
    This is significantly faster than calling get_embedding() multiple times.
    
    :param texts: List of input texts.
    :param model_name: Hugging Face model name.
    :param use_cache: Whether to use memory cache.
    :param kwargs: Additional parameters for model.
    :return: List of embedding vectors (numpy arrays).
    """
    if not texts:
        return []
    
    model_config_key = json.dumps({"model_name": model_name, **kwargs}, sort_keys=True)
    
    # Check cache for all texts
    cached_results = {}
    texts_to_encode = []
    text_indices = {}
    
    if use_cache:
        for i, text in enumerate(texts):
            cache_key = f"{model_config_key}::{hash(text)}"
            if cache_key in _embedding_cache:
                cached_results[i] = _embedding_cache[cache_key]
            else:
                texts_to_encode.append(text)
                text_indices[len(texts_to_encode) - 1] = i
    else:
        texts_to_encode = texts
        text_indices = {i: i for i in range(len(texts))}
    
    # Load model if needed
    model_init_key = json.dumps({"model_name": model_name, **{k:v for k,v in kwargs.items() if k not in ['batch_size', 'max_length']}}, sort_keys=True)
    if model_init_key not in _model_cache:
        print(f"Loading model for batch processing: {model_name}...")
        if 'bge-m3' in model_name.lower():
            try:
                from FlagEmbedding import BGEM3FlagModel
                init_kwargs = _get_valid_kwargs(BGEM3FlagModel.__init__, kwargs)
                _model_cache[model_init_key] = BGEM3FlagModel(model_name, **init_kwargs)
            except ImportError:
                raise ImportError("Please install FlagEmbedding: 'pip install -U FlagEmbedding' to use bge-m3 model.")
        else:
            try:
                from sentence_transformers import SentenceTransformer
                init_kwargs = _get_valid_kwargs(SentenceTransformer.__init__, kwargs)
                _model_cache[model_init_key] = SentenceTransformer(model_name, **init_kwargs)
            except ImportError:
                raise ImportError("Please install sentence-transformers: 'pip install -U sentence-transformers' to use this model.")
    
    model = _model_cache[model_init_key]
    
    # Batch encode uncached texts
    new_embeddings = []
    if texts_to_encode:
        print(f"Batch encoding {len(texts_to_encode)} texts with {model_name}")
        if 'bge-m3' in model_name.lower():
            encode_kwargs = _get_valid_kwargs(model.encode, kwargs)
            result = model.encode(texts_to_encode, **encode_kwargs)
            new_embeddings = result['dense_vecs']
        else:
            encode_kwargs = _get_valid_kwargs(model.encode, kwargs)
            new_embeddings = model.encode(texts_to_encode, **encode_kwargs)
        
        # Cache new embeddings
        if use_cache:
            for i, embedding in enumerate(new_embeddings):
                text = texts_to_encode[i]
                cache_key = f"{model_config_key}::{hash(text)}"
                _embedding_cache[cache_key] = embedding
    
    # Combine cached and new results
    results = [None] * len(texts)
    for i in range(len(texts)):
        if i in cached_results:
            results[i] = cached_results[i]
        else:
            # Find the corresponding new embedding
            for new_idx, original_idx in text_indices.items():
                if original_idx == i:
                    results[i] = new_embeddings[new_idx]
                    break
    
    # Clean cache if too large
    if use_cache and len(_embedding_cache) > 10000:
        keys_to_remove = list(_embedding_cache.keys())[:1000]
        for key in keys_to_remove:
            try:
                del _embedding_cache[key]
            except KeyError:
                pass
        print("Cleaned embedding cache to prevent memory overflow")
    
    return results


def clear_embedding_cache():
    """清空embedding缓存"""
    global _embedding_cache
    _embedding_cache.clear()
    print("Embedding cache cleared")


def compress_memory_data(data):
    """
    PERFORMANCE OPTIMIZATION: Compress memory data to reduce I/O overhead.
    Uses gzip compression for JSON data.
    """
    import gzip
    import pickle
    
    try:
        # Convert to JSON string first, then compress
        json_str = json.dumps(data, ensure_ascii=False)
        compressed = gzip.compress(json_str.encode('utf-8'))
        return compressed
    except Exception as e:
        print(f"Error compressing data: {e}")
        return None


def decompress_memory_data(compressed_data):
    """
    PERFORMANCE OPTIMIZATION: Decompress memory data.
    """
    import gzip
    
    try:
        # Decompress and parse JSON
        json_str = gzip.decompress(compressed_data).decode('utf-8')
        return json.loads(json_str)
    except Exception as e:
        print(f"Error decompressing data: {e}")
        return None


def optimize_memory_structure(memory_data):
    """
    PERFORMANCE OPTIMIZATION: Optimize memory structure to reduce redundancy.
    """
    if not isinstance(memory_data, dict):
        return memory_data
    
    # Remove redundant timestamp info if all entries have similar patterns
    # Compress repeated strings
    # Optimize embedding storage (can be converted to smaller data types)
    
    optimized_data = {}
    for key, value in memory_data.items():
        if key == "sessions" and isinstance(value, dict):
            optimized_sessions = {}
            for session_id, session_data in value.items():
                if isinstance(session_data, dict):
                    optimized_session = session_data.copy()
                    
                    # Optimize embeddings - convert to float16 for smaller storage
                    if "details" in optimized_session:
                        for page in optimized_session["details"]:
                            if "page_embedding" in page and page["page_embedding"]:
                                try:
                                    import numpy as np
                                    embedding = np.array(page["page_embedding"], dtype=np.float32)
                                    # Convert to float16 for ~50% size reduction
                                    page["page_embedding"] = embedding.astype(np.float16).tolist()
                                except:
                                    pass  # Keep original if conversion fails
                    
                    optimized_sessions[session_id] = optimized_session
                else:
                    optimized_sessions[session_id] = session_data
            optimized_data[key] = optimized_sessions
        else:
            optimized_data[key] = value
    
    return optimized_data

def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# ---- Time Decay Function ----
def compute_time_decay(event_timestamp_str, current_timestamp_str, tau_hours=24):
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        t_event = datetime.strptime(event_timestamp_str, fmt)
        t_current = datetime.strptime(current_timestamp_str, fmt)
        delta_hours = (t_current - t_event).total_seconds() / 3600.0
        return np.exp(-delta_hours / tau_hours)
    except ValueError: # Handle cases where timestamp might be invalid
        return 0.1 # Default low recency


# ---- LLM-based Utility Functions ----

def gpt_summarize_dialogs(dialogs, client: LLMClient, model="gpt-4o-mini"):
    dialog_text = "\n".join([f"User: {d.get('user_input','')} Assistant: {d.get('agent_response','')}" for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.SUMMARIZE_DIALOGS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.SUMMARIZE_DIALOGS_USER_PROMPT.format(dialog_text=dialog_text)}
    ]
    print("Calling LLM to generate topic summary...")
    return client.chat_completion(model=model, messages=messages)

def gpt_generate_multi_summary(text, client: LLMClient, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": prompts.MULTI_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.MULTI_SUMMARY_USER_PROMPT.format(text=text)}
    ]
    print("Calling LLM to generate multi-topic summary...")
    response_text = client.chat_completion(model=model, messages=messages)
    try:
        summaries = json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse multi-summary JSON: {response_text}")
        summaries = [] # Return empty list or a default structure
    return {"input": text, "summaries": summaries}


def gpt_user_profile_analysis(dialogs, client: LLMClient, model="gpt-4o-mini", existing_user_profile="None"):
    """
    Analyze and update user personality profile from dialogs
    结合现有画像和新对话，直接输出更新后的完整画像
    """
    conversation = "\n".join([f"User: {d.get('user_input','')} (Timestamp: {d.get('timestamp', '')})\nAssistant: {d.get('agent_response','')} (Timestamp: {d.get('timestamp', '')})" for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.PERSONALITY_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.PERSONALITY_ANALYSIS_USER_PROMPT.format(
            conversation=conversation,
            existing_user_profile=existing_user_profile
        )}
    ]
    print("Calling LLM for user profile analysis and update...")
    result_text = client.chat_completion(model=model, messages=messages)
    return result_text.strip() if result_text else "None"


def gpt_knowledge_extraction(dialogs, client: LLMClient, model="gpt-4o-mini"):
    """Extract user private data and assistant knowledge from dialogs"""
    conversation = "\n".join([f"User: {d.get('user_input','')} (Timestamp: {d.get('timestamp', '')})\nAssistant: {d.get('agent_response','')} (Timestamp: {d.get('timestamp', '')})" for d in dialogs])
    messages = [
        {"role": "system", "content": prompts.KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.KNOWLEDGE_EXTRACTION_USER_PROMPT.format(
            conversation=conversation
        )}
    ]
    print("Calling LLM for knowledge extraction...")
    result_text = client.chat_completion(model=model, messages=messages)
    
    private_data = "None"
    assistant_knowledge = "None"

    try:
        if "【User Private Data】" in result_text:
            private_data_start = result_text.find("【User Private Data】") + len("【User Private Data】")
            if "【Assistant Knowledge】" in result_text:
                private_data_end = result_text.find("【Assistant Knowledge】")
                private_data = result_text[private_data_start:private_data_end].strip()
                
                assistant_knowledge_start = result_text.find("【Assistant Knowledge】") + len("【Assistant Knowledge】")
                assistant_knowledge = result_text[assistant_knowledge_start:].strip()
            else:
                private_data = result_text[private_data_start:].strip()
        elif "【Assistant Knowledge】" in result_text:
             assistant_knowledge_start = result_text.find("【Assistant Knowledge】") + len("【Assistant Knowledge】")
             assistant_knowledge = result_text[assistant_knowledge_start:].strip()

    except Exception as e:
        print(f"Error parsing knowledge extraction: {e}. Raw result: {result_text}")

    return {
        "private": private_data if private_data else "None", 
        "assistant_knowledge": assistant_knowledge if assistant_knowledge else "None"
    }


# Keep the old function for backward compatibility, but mark as deprecated
def gpt_personality_analysis(dialogs, client: LLMClient, model="gpt-4o-mini", known_user_traits="None"):
    """
    DEPRECATED: Use gpt_user_profile_analysis and gpt_knowledge_extraction instead.
    This function is kept for backward compatibility only.
    """
    # Call the new functions
    profile = gpt_user_profile_analysis(dialogs, client, model, known_user_traits)
    knowledge_data = gpt_knowledge_extraction(dialogs, client, model)
    
    return {
        "profile": profile,
        "private": knowledge_data["private"],
        "assistant_knowledge": knowledge_data["assistant_knowledge"]
    }


def gpt_update_profile(old_profile, new_analysis, client: LLMClient, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": prompts.UPDATE_PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.UPDATE_PROFILE_USER_PROMPT.format(old_profile=old_profile, new_analysis=new_analysis)}
    ]
    print("Calling LLM to update user profile...")
    return client.chat_completion(model=model, messages=messages)

def gpt_extract_theme(answer_text, client: LLMClient, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": prompts.EXTRACT_THEME_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.EXTRACT_THEME_USER_PROMPT.format(answer_text=answer_text)}
    ]
    print("Calling LLM to extract theme...")
    return client.chat_completion(model=model, messages=messages)

def llm_extract_keywords(text, client: LLMClient, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": prompts.EXTRACT_KEYWORDS_SYSTEM_PROMPT},
        {"role": "user", "content": prompts.EXTRACT_KEYWORDS_USER_PROMPT.format(text=text)}
    ]
    print("Calling LLM to extract keywords...")
    response = client.chat_completion(model=model, messages=messages)
    return [kw.strip() for kw in response.split(',') if kw.strip()]

# ---- Functions from dynamic_update.py (to be used by Updater class) ----
def check_conversation_continuity(previous_page, current_page, client: LLMClient, model="gpt-4o-mini"):
    prev_user = previous_page.get("user_input", "") if previous_page else ""
    prev_agent = previous_page.get("agent_response", "") if previous_page else ""
    
    user_prompt = prompts.CONTINUITY_CHECK_USER_PROMPT.format(
        prev_user=prev_user,
        prev_agent=prev_agent,
        curr_user=current_page.get("user_input", ""),
        curr_agent=current_page.get("agent_response", "")
    )
    messages = [
        {"role": "system", "content": prompts.CONTINUITY_CHECK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat_completion(model=model, messages=messages, temperature=0.0, max_tokens=10)
    return response.strip().lower() == "true"

def generate_page_meta_info(last_page_meta, current_page, client: LLMClient, model="gpt-4o-mini"):
    current_conversation = f"User: {current_page.get('user_input', '')}\nAssistant: {current_page.get('agent_response', '')}"
    user_prompt = prompts.META_INFO_USER_PROMPT.format(
        last_meta=last_page_meta if last_page_meta else "None",
        new_dialogue=current_conversation
    )
    messages = [
        {"role": "system", "content": prompts.META_INFO_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    return client.chat_completion(model=model, messages=messages, temperature=0.3, max_tokens=100).strip() 