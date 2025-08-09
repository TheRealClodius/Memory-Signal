
import sys
import os
import json
import argparse
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv
# Ensure the current directory is in sys.path so that the `memoryos` package can be imported
sys.path.insert(0, os.path.dirname(__file__))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    print(f"ERROR: Failed to import FastMCP. Exception: {e}", file=sys.stderr)
    print("请安装最新版本的MCP: pip install --upgrade mcp", file=sys.stderr)
    sys.exit(1)

try:
    from memoryos import Memoryos
    from memoryos.utils import get_timestamp
except ImportError as e:
    print(f"无法导入MemoryOS模块: {e}", file=sys.stderr)
    print("请确保项目结构正确，memoryos目录应包含所有必要文件", file=sys.stderr)
    sys.exit(1)

# MemoryOS实例 - 将在初始化时设置
memoryos_instance: Optional[Memoryos] = None

def init_memoryos(config_path: str) -> Memoryos:
    """Initialize the MemoryOS instance."""
    # Load environment variables from .env.local if present
    try:
        env_path = os.path.join(os.path.dirname(__file__), '.env.local')
        load_dotenv(dotenv_path=env_path)
    except Exception:
        # Non-fatal if dotenv isn't available or file missing; continue with config values
        pass
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    required_fields = ['user_id', 'openai_api_key', 'data_storage_path']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"配置文件缺少必需字段: {field}")
    
    return Memoryos(
        user_id=config['user_id'],
        openai_api_key=os.getenv('OPENAI_API_KEY', config['openai_api_key']),
        data_storage_path=os.getenv('DATA_STORAGE_PATH', config['data_storage_path']),
        openai_base_url=os.getenv('OPENAI_BASE_URL', config.get('openai_base_url')),
        assistant_id=os.getenv('ASSISTANT_ID', config.get('assistant_id', 'default_assistant_profile')),
        short_term_capacity=config.get('short_term_capacity', 10),
        mid_term_capacity=config.get('mid_term_capacity', 2000),
        long_term_knowledge_capacity=config.get('long_term_knowledge_capacity', 100),
        retrieval_queue_capacity=config.get('retrieval_queue_capacity', 7),
        mid_term_heat_threshold=config.get('mid_term_heat_threshold', 5.0),
        llm_model=os.getenv('LLM_MODEL', config.get('llm_model', 'gpt-4o-mini')),
        embedding_model_name=os.getenv('EMBEDDING_MODEL_NAME', config.get('embedding_model_name', 'all-MiniLM-L6-v2'))
    )

# Create FastMCP server instance
mcp = FastMCP("MemoryOS")

@mcp.tool()
def add_memory(user_input: str, agent_response: str, timestamp: Optional[str] = None, meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Add a new memory (a user input and assistant response pair) to MemoryOS.

    Args:
        user_input: The user's input or question.
        agent_response: The assistant's response.
        timestamp: Optional timestamp in the format YYYY-MM-DD HH:MM:SS.
        meta_data: Optional metadata (JSON object).

    Returns:
        A dictionary containing the operation result.
    """
    global memoryos_instance
    
    if memoryos_instance is None:
        return {
            "status": "error",
            "message": "MemoryOS is not initialized. Please check the configuration file."
        }
    
    try:
        if not user_input or not agent_response:
            return {
                "status": "error",
                "message": "user_input and agent_response are required"
            }
        
        memoryos_instance.add_memory(
            user_input=user_input,
            agent_response=agent_response,
            timestamp=timestamp or get_timestamp(),
            meta_data=meta_data or {}
        )
        
        result = {
            "status": "success",
            "message": "Memory has been successfully added to MemoryOS",
            "timestamp": timestamp or get_timestamp(),
            "details": {
                "user_input_length": len(user_input),
                "agent_response_length": len(agent_response),
                "has_meta_data": meta_data is not None
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error adding memory: {str(e)}"
        }

@mcp.tool()
def retrieve_memory(query: str, relationship_with_user: str = "friend", style_hint: str = "", max_results: int = 10) -> Dict[str, Any]:
    """
    Retrieve relevant memories and context from MemoryOS based on a query, including
    short-term memory, mid-term memory, and long-term knowledge.

    Args:
        query: The retrieval query describing what to find.
        relationship_with_user: Relationship to the user (e.g., friend, assistant, colleague).
        style_hint: Optional response style hint.
        max_results: Maximum number of items to return.

    Returns:
        A dictionary with the retrieval results, including:
        - short_term_memory: All QA pairs currently in short-term memory.
        - retrieved_pages: Relevant pages retrieved from mid-term memory.
        - retrieved_user_knowledge: Relevant entries from the user's long-term knowledge.
        - retrieved_assistant_knowledge: Relevant entries from the assistant's knowledge base.
    """
    global memoryos_instance
    
    if memoryos_instance is None:
        return {
            "status": "error",
            "message": "MemoryOS is not initialized. Please check the configuration file."
        }
    
    try:
        if not query:
            return {
                "status": "error",
                "message": "query parameter is required"
            }
        
        # 使用retriever获取相关上下文
        retrieval_results = memoryos_instance.retriever.retrieve_context(
            user_query=query,
            user_id=memoryos_instance.user_id
        )
        
        # 获取短期记忆内容
        short_term_history = memoryos_instance.short_term_memory.get_all()
        
        # 获取用户画像
        user_profile = memoryos_instance.get_user_profile_summary()
        
        # 组织返回结果
        result = {
            "status": "success",
            "query": query,
            "timestamp": get_timestamp(),
            "user_profile": user_profile if user_profile and user_profile.lower() != "none" else "No detailed user profile",
            "short_term_memory": short_term_history,
            "short_term_count": len(short_term_history),
            "retrieved_pages": [{
                'user_input': page['user_input'],
                'agent_response': page['agent_response'],
                'timestamp': page['timestamp'],
                'meta_info': page['meta_info']
            } for page in retrieval_results["retrieved_pages"][:max_results]],

            "retrieved_user_knowledge": [{
                    'knowledge': k['knowledge'],
                    'timestamp': k['timestamp']
                } for k in retrieval_results["retrieved_user_knowledge"][:max_results]],

            "retrieved_assistant_knowledge": [{
                'knowledge': k['knowledge'],
                'timestamp': k['timestamp']
            } for k in retrieval_results["retrieved_assistant_knowledge"][:max_results]],
            
            # 添加总数统计字段
            "total_pages_found": len(retrieval_results["retrieved_pages"]),
            "total_user_knowledge_found": len(retrieval_results["retrieved_user_knowledge"]),
            "total_assistant_knowledge_found": len(retrieval_results["retrieved_assistant_knowledge"])
        }
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving memory: {str(e)}"
        }

@mcp.tool()
def get_user_profile(include_knowledge: bool = True, include_assistant_knowledge: bool = False) -> Dict[str, Any]:
    """
    Get the user's profile information, including personality traits, preferences,
    and related knowledge.

    Args:
        include_knowledge: Whether to include user knowledge entries.
        include_assistant_knowledge: Whether to include assistant knowledge entries.

    Returns:
        A dictionary containing the user's profile information.
    """
    global memoryos_instance
    
    if memoryos_instance is None:
        return {
            "status": "error",
            "message": "MemoryOS is not initialized. Please check the configuration file."
        }
    
    try:
        # 获取用户画像
        user_profile = memoryos_instance.get_user_profile_summary()
        
        result = {
            "status": "success",
            "timestamp": get_timestamp(),
            "user_id": memoryos_instance.user_id,
            "assistant_id": memoryos_instance.assistant_id,
            "user_profile": user_profile if user_profile and user_profile.lower() != "none" else "No detailed user profile"
        }
        
        if include_knowledge:
            user_knowledge = memoryos_instance.user_long_term_memory.get_user_knowledge()
            result["user_knowledge"] = [
                {
                    "knowledge": item["knowledge"],
                    "timestamp": item["timestamp"]
                }
                for item in user_knowledge
            ]
            result["user_knowledge_count"] = len(user_knowledge)
        
        if include_assistant_knowledge:
            assistant_knowledge = memoryos_instance.get_assistant_knowledge_summary()
            result["assistant_knowledge"] = [
                {
                    "knowledge": item["knowledge"],
                    "timestamp": item["timestamp"]
                }
                for item in assistant_knowledge
            ]
            result["assistant_knowledge_count"] = len(assistant_knowledge)
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting user profile: {str(e)}"
        }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MemoryOS MCP Server")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json",
        help="配置文件路径 (默认: config.json)"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=os.getenv("ENV_FILE", ".env.local"),
        help="可选的环境变量文件 (默认: .env.local 或环境变量 ENV_FILE)"
    )
    parser.add_argument(
        "--transport",
        type=str,
        default=os.getenv("TRANSPORT", "stdio"),
        choices=["stdio", "http"],
        help="MCP 传输方式: stdio 或 http (默认: stdio，或从环境变量 TRANSPORT 读取)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="HTTP 主机 (默认: 0.0.0.0 或环境变量 HOST)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="HTTP 端口 (默认: 8000 或环境变量 PORT)"
    )
    
    args = parser.parse_args()
    
    global memoryos_instance
    
    try:
        # Initialize MemoryOS (load the specified env file first)
        try:
            load_dotenv(dotenv_path=args.env_file)
        except Exception:
            pass
        memoryos_instance = init_memoryos(args.config)
        print(f"MemoryOS MCP Server 已启动，用户ID: {memoryos_instance.user_id}", file=sys.stderr)
        print(f"配置文件: {args.config}", file=sys.stderr)
        print(f"传输方式: {args.transport}", file=sys.stderr)
        
        # Start the MCP server
        if args.transport == "http":
            # Configure host/port via settings and use streamable-http transport
            mcp.settings.host = args.host
            mcp.settings.port = args.port
            mcp.run(transport="streamable-http")
        else:
            mcp.run(transport="stdio")
        
    except KeyboardInterrupt:
        print("服务器被用户中断", file=sys.stderr)
    except Exception as e:
        print(f"启动服务器时发生错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()