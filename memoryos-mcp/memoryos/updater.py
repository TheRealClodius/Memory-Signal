from .utils import (
    generate_id, get_timestamp, 
    gpt_generate_multi_summary, check_conversation_continuity, generate_page_meta_info, OpenAIClient,
    run_parallel_tasks
)
from .short_term import ShortTermMemory
from .mid_term import MidTermMemory
from .long_term import LongTermMemory

from concurrent.futures import ThreadPoolExecutor, as_completed

class Updater:
    def __init__(self, 
                 short_term_memory: ShortTermMemory, 
                 mid_term_memory: MidTermMemory, 
                 long_term_memory: LongTermMemory, 
                 client: OpenAIClient,
                 topic_similarity_threshold=0.5,
                 llm_model="gpt-4o-mini"):
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.client = client
        self.topic_similarity_threshold = topic_similarity_threshold
        self.last_evicted_page_for_continuity = None # Tracks the actual last page object for continuity checks
        self.llm_model = llm_model

    def _process_page_embedding_and_keywords(self, page_data):
        """处理单个页面的embedding生成（关键词由multi-summary提供）"""
        page_id = page_data.get("page_id", generate_id("page"))
        
        # 检查是否已有embedding
        if "page_embedding" in page_data and page_data["page_embedding"]:
            print(f"Updater: Page {page_id} already has embedding, skipping computation")
            return page_data
        
        # 只处理embedding，关键词由multi-summary统一提供
        if not ("page_embedding" in page_data and page_data["page_embedding"]):
            full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
            try:
                embedding = self._get_embedding_for_page(full_text)
                if embedding is not None:
                    from .utils import normalize_vector
                    page_data["page_embedding"] = normalize_vector(embedding).tolist()
                    print(f"Updater: Generated embedding for page {page_id}")
            except Exception as e:
                print(f"Error generating embedding for page {page_id}: {e}")
        
        # 设置空的关键词列表（将由multi-summary的关键词填充）
        if "page_keywords" not in page_data:
            page_data["page_keywords"] = []
        
        return page_data

    def _get_embedding_for_page(self, text):
        """获取页面embedding的辅助方法"""
        from .utils import get_embedding
        return get_embedding(text)

    def _batch_process_page_embeddings(self, pages_list):
        """
        PERFORMANCE OPTIMIZATION: Batch process embeddings for multiple pages.
        This reduces the number of model loading calls and improves throughput.
        """
        pages_needing_embeddings = []
        pages_with_embeddings = []
        
        for page_data in pages_list:
            if "page_embedding" in page_data and page_data["page_embedding"]:
                pages_with_embeddings.append(page_data)
            else:
                pages_needing_embeddings.append(page_data)
        
        if not pages_needing_embeddings:
            print("Updater: All pages already have embeddings")
            return pages_list
        
        # Prepare texts for batch embedding
        texts_for_embedding = []
        for page_data in pages_needing_embeddings:
            full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
            texts_for_embedding.append(full_text)
        
        print(f"Updater: Batch processing embeddings for {len(texts_for_embedding)} pages")
        
        try:
            from .utils import get_batch_embeddings, normalize_vector
            
            # Get model configuration from mid_term_memory
            model_name = self.mid_term_memory.embedding_model_name
            model_kwargs = self.mid_term_memory.embedding_model_kwargs
            
            # Batch generate embeddings
            embeddings = get_batch_embeddings(
                texts_for_embedding, 
                model_name=model_name, 
                **model_kwargs
            )
            
            # Assign embeddings to pages
            for i, page_data in enumerate(pages_needing_embeddings):
                if i < len(embeddings) and embeddings[i] is not None:
                    page_data["page_embedding"] = normalize_vector(embeddings[i]).tolist()
                    page_id = page_data.get("page_id", "unknown")
                    print(f"Updater: Assigned batch embedding to page {page_id}")
                
                # Set empty keywords (will be filled by multi-summary)
                if "page_keywords" not in page_data:
                    page_data["page_keywords"] = []
        
        except Exception as e:
            print(f"Error in batch embedding processing: {e}")
            # Fallback to individual processing
            print("Falling back to individual embedding processing...")
            for page_data in pages_needing_embeddings:
                self._process_page_embedding_and_keywords(page_data)
        
        # Combine all pages
        all_processed_pages = pages_with_embeddings + pages_needing_embeddings
        return all_processed_pages

    def _update_linked_pages_meta_info(self, start_page_id, new_meta_info):
        """
        Updates meta_info for a chain of connected pages starting from start_page_id.
        This is a simplified version. Assumes that once a chain is broken (no pre_page),
        we don't need to go further back. Updates forward as well.
        """
        # Go backward
        q = [start_page_id]
        visited = {start_page_id}
        
        head = 0
        while head < len(q):
            current_page_id = q[head]
            head += 1
            page = self.mid_term_memory.get_page_by_id(current_page_id)
            if page:
                page["meta_info"] = new_meta_info
                # Check previous page
                prev_id = page.get("pre_page")
                if prev_id and prev_id not in visited:
                    q.append(prev_id)
                    visited.add(prev_id)
                # Check next page
                next_id = page.get("next_page")
                if next_id and next_id not in visited:
                    q.append(next_id)
                    visited.add(next_id)
        if q: # If any pages were updated
            self.mid_term_memory.save() # Save mid-term memory after updates

    def process_short_term_to_mid_term(self):
        """
        PERFORMANCE OPTIMIZED: Process spillover from short-term to mid-term memory
        with batch operations and parallel processing to avoid the performance cliff.
        """
        evicted_qas = []
        while self.short_term_memory.is_full():
            qa = self.short_term_memory.pop_oldest()
            if qa and qa.get("user_input") and qa.get("agent_response"):
                evicted_qas.append(qa)
        
        if not evicted_qas:
            print("Updater: No QAs evicted from short-term memory.")
            return

        print(f"Updater: OPTIMIZED processing {len(evicted_qas)} QAs from short-term to mid-term.")
        
        # 1. Create page structures FIRST (without LLM calls)
        current_batch_pages = []
        temp_last_page_in_batch = self.last_evicted_page_for_continuity

        for qa_pair in evicted_qas:
            current_page_obj = {
                "page_id": generate_id("page"),
                "user_input": qa_pair.get("user_input", ""),
                "agent_response": qa_pair.get("agent_response", ""),
                "timestamp": qa_pair.get("timestamp", get_timestamp()),
                "preloaded": False, # Default for new pages from short-term
                "analyzed": False,  # Default for new pages from short-term
                "pre_page": None,
                "next_page": None,
                "meta_info": None
            }
            current_batch_pages.append(current_page_obj)
        
        # Update last evicted page for continuity
        if current_batch_pages:
            self.last_evicted_page_for_continuity = current_batch_pages[-1]

        # 2. BATCH PROCESS EMBEDDINGS (Major optimization)
        print("Updater: Batch processing embeddings for all pages...")
        current_batch_pages = self._batch_process_page_embeddings(current_batch_pages)
        
        # 3. PARALLEL LLM PROCESSING (continuity, meta-info, multi-summary)
        print("Updater: Starting parallel LLM processing...")
        
        def task_continuity_and_meta():
            """Process continuity checks and meta-info generation"""
            temp_last = self.last_evicted_page_for_continuity
            
            for i, current_page_obj in enumerate(current_batch_pages):
                # For batch processing, we'll use simpler heuristics for continuity
                # to avoid N individual LLM calls during spillover
                
                prev_page = current_batch_pages[i-1] if i > 0 else temp_last
                if prev_page and i > 0:  # Simple continuity: if consecutive in this batch
                    current_page_obj["pre_page"] = prev_page["page_id"]
                    # Generate meta-info based on context
                    last_meta = prev_page.get("meta_info")
                    try:
                        current_page_obj["meta_info"] = generate_page_meta_info(
                            last_meta, current_page_obj, self.client, model=self.llm_model
                        )
                    except Exception as e:
                        print(f"Error generating meta-info: {e}")
                        current_page_obj["meta_info"] = f"Conversation continuation from {current_page_obj['timestamp']}"
                else:
                    # Start of new chain
                    try:
                        current_page_obj["meta_info"] = generate_page_meta_info(
                            None, current_page_obj, self.client, model=self.llm_model
                        )
                    except Exception as e:
                        print(f"Error generating meta-info: {e}")
                        current_page_obj["meta_info"] = f"New conversation topic at {current_page_obj['timestamp']}"
            
            return current_batch_pages
        
        def task_multi_summary():
            """Generate multi-topic summary"""
            input_text = "\n".join([
                f"User: {p.get('user_input','')}\nAssistant: {p.get('agent_response','')}" 
                for p in current_batch_pages
            ])
            
            try:
                return gpt_generate_multi_summary(input_text, self.client, model=self.llm_model)
            except Exception as e:
                print(f"Error in multi-summary generation: {e}")
                return {"summaries": []}
        
        # Execute parallel tasks
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_continuity = executor.submit(task_continuity_and_meta)
            future_summary = executor.submit(task_multi_summary)
            
            try:
                processed_pages = future_continuity.result(timeout=30)  # Add timeout
                multi_summary_result = future_summary.result(timeout=30)
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                # Fallback to basic processing
                processed_pages = current_batch_pages
                multi_summary_result = {"summaries": []}
        
        # 4. Insert pages into MidTermMemory based on summaries
        if multi_summary_result and multi_summary_result.get("summaries"):
            print(f"Updater: Processing {len(multi_summary_result['summaries'])} themes for insertion")
            for summary_item in multi_summary_result["summaries"]:
                theme_summary = summary_item.get("content", "General summary of recent interactions.")
                theme_keywords = summary_item.get("keywords", [])
                theme_name = summary_item.get("theme", "General")
                print(f"Updater: Processing theme '{theme_name}' for mid-term insertion.")
                
                # Insert with batch-processed embeddings
                self.mid_term_memory.insert_pages_into_session(
                    summary_for_new_pages=theme_summary,
                    keywords_for_new_pages=theme_keywords,
                    pages_to_insert=processed_pages,
                    similarity_threshold=self.topic_similarity_threshold
                )
        else:
            # Fallback: single session
            print("Updater: Using fallback single session insertion")
            fallback_summary = "General conversation segment from short-term memory."
            fallback_keywords = []
            self.mid_term_memory.insert_pages_into_session(
                summary_for_new_pages=fallback_summary,
                keywords_for_new_pages=fallback_keywords,
                pages_to_insert=processed_pages,
                similarity_threshold=self.topic_similarity_threshold
            )
        
        # 5. Finalize connections (optimized)
        print("Updater: Finalizing page connections...")
        connections_updated = False
        for page in processed_pages:
            if page.get("pre_page"):
                self.mid_term_memory.update_page_connections(page["pre_page"], page["page_id"])
                connections_updated = True
        
        # Finalize all deferred operations after spillover is complete
        self.mid_term_memory.finalize_deferred_operations()
        
        print(f"Updater: OPTIMIZED spillover complete for {len(processed_pages)} pages")

    def update_long_term_from_analysis(self, user_id, profile_analysis_result):
        """
        Updates long-term memory based on the results of a personality/knowledge analysis.
        profile_analysis_result is expected to be a dict with keys like "profile", "private", "assistant_knowledge".
        """
        if not profile_analysis_result:
            print("Updater: No analysis result provided for long-term update.")
            return

        new_profile_text = profile_analysis_result.get("profile")
        if new_profile_text and new_profile_text.lower() != "none":
            print(f"Updater: Updating user profile for {user_id} in LongTermMemory.")
            # 直接使用新的分析结果作为完整画像，因为它应该已经是集成后的结果
            self.long_term_memory.update_user_profile(user_id, new_profile_text, merge=False)
        
        user_private_knowledge = profile_analysis_result.get("private")
        if user_private_knowledge and user_private_knowledge.lower() != "none":
            print(f"Updater: Adding user private knowledge for {user_id} to LongTermMemory.")
            # Split if multiple lines, assuming each line is a distinct piece of knowledge
            for line in user_private_knowledge.split('\n'):
                if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                    self.long_term_memory.add_user_knowledge(line.strip()) 

        assistant_knowledge_text = profile_analysis_result.get("assistant_knowledge")
        if assistant_knowledge_text and assistant_knowledge_text.lower() != "none":
            print("Updater: Adding assistant knowledge to LongTermMemory.")
            for line in assistant_knowledge_text.split('\n'):
                if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                    self.long_term_memory.add_assistant_knowledge(line.strip())

        # LongTermMemory.save() is called by its add/update methods 