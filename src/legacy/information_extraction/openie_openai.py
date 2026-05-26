import json
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput, TopicRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_ner_from_response(real_response):
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    if match is None:
        # If pattern doesn't match, return an empty list
        return []
    return eval(match.group())["named_entities"]


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI, topic_extraction_mode: str = 'default',
                 topic_extraction_llm: CacheOpenAI = None):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        # Use separate LLM for topic extraction if provided, otherwise use the same llm_model
        self.topic_extraction_llm = topic_extraction_llm if topic_extraction_llm is not None else llm_model
        self.topic_extraction_mode = topic_extraction_mode
        # Determine which topic extraction prompt to use
        if topic_extraction_mode == 'fine_grained':
            self.topic_prompt_name = 'topic_extraction_fine_grained'
        else:
            self.topic_prompt_name = 'topic_extraction'
        self.last_openie_stats: Dict[str, Dict[str, Any]] = {}

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=ner_input_message,
            )
            metadata['cache_hit'] = cache_hit
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is None:
                # If pattern doesn't match, return an empty list
                return []
            return eval(match.group())["triples"]

        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
            )
            metadata['cache_hit'] = cache_hit
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        return {"ner": ner_output, "triplets": triple_output}
    
    def _validate_think_memory_output(self, response: str) -> Tuple[bool, str, str]:
        """
        验证 LLM 输出是否包含 <think> 和 <memory> 标签

        Returns:
            Tuple[bool, str, str]: (是否有效, think内容, memory内容)
        """
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        memory_match = re.search(r'<memory>(.*?)</memory>', response, re.DOTALL)

        think = think_match.group(1).strip() if think_match else ""
        memory = memory_match.group(1).strip() if memory_match else ""

        # 验证：两个字段都必须存在且非空
        is_valid = bool(think_match and memory_match and think and memory)

        return is_valid, think, memory

    # def topic_extraterrestrial
    def topic_extraction(self, chunk_key, chunk_value, max_retries=3): # dev add
        """
        提取 topic，包含 think 和 memory 两部分

        Args:
            chunk_key: chunk 的唯一标识
            chunk_value: chunk 的文本内容
            max_retries: 最大重试次数（默认3次）

        Returns:
            TopicRawOutput: 包含 think 和 memory 的结果
        """
        topic_extration_input_message = self.prompt_template_manager.render(name=self.topic_prompt_name, passage=chunk_value)
        raw_response = ""
        metadata = {}
        retry_count = 0

        while retry_count < max_retries:
            try:
                # LLM INFERENCE - Use topic_extraction_llm for this step
                # -> Tuple[List[TextChatMessage], dict]:
                raw_response, metadata, cache_hit = self.topic_extraction_llm.infer(
                    messages=topic_extration_input_message,
                )
                metadata['cache_hit'] = cache_hit
                metadata['retry_count'] = retry_count

                if metadata['finish_reason'] == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response

                # 验证输出是否包含 think 和 memory 标签
                is_valid, think, memory = self._validate_think_memory_output(real_response)

                if is_valid:
                    # 输出有效，返回结果
                    topic = memory  # 保留 topic 字段用于兼容性

                    print(f"[dev] [OpenIE] [topic_extration] SUCCESS (retry={retry_count}) chunk_key={chunk_key}") # dev add
                    print(f"[dev] [OpenIE] [topic_extration] think (len={len(think)}): \n{think[:200]}...") # dev add
                    print(f"[dev] [OpenIE] [topic_extration] memory (len={len(memory)}): \n{memory[:200]}...") # dev add

                    return TopicRawOutput(
                        chunk_id=chunk_key,
                        response=raw_response,
                        topic=topic,
                        think=think,
                        memory=memory,
                        metadata=metadata
                    )
                else:
                    # 输出无效，记录并重试
                    retry_count += 1
                    logger.warning(f"[topic_extraction] Invalid output for chunk {chunk_key}, retry {retry_count}/{max_retries}")
                    logger.warning(f"[topic_extraction] Response preview: {real_response[:200]}...")

                    if retry_count < max_retries:
                        continue
                    else:
                        # 达到最大重试次数，标记为失败
                        logger.error(f"[topic_extraction] Failed after {max_retries} retries for chunk {chunk_key}")
                        metadata['extraction_failed'] = True
                        metadata['failure_reason'] = 'max_retries_exceeded'

                        # 使用原始文本作为 memory，think 为空
                        return TopicRawOutput(
                            chunk_id=chunk_key,
                            response=raw_response,
                            topic=chunk_value,
                            think="",
                            memory=chunk_value,
                            metadata=metadata
                        )

            except Exception as e:
                # 发生异常，记录并重试
                retry_count += 1
                logger.warning(f"[topic_extraction] Exception for chunk {chunk_key}, retry {retry_count}/{max_retries}: {e}")

                if retry_count < max_retries:
                    continue
                else:
                    # 达到最大重试次数，返回错误结果
                    logger.error(f"[topic_extraction] Exception after {max_retries} retries for chunk {chunk_key}: {e}")
                    metadata.update({'error': str(e), 'extraction_failed': True, 'failure_reason': 'exception'})
                    return TopicRawOutput(
                        chunk_id=chunk_key,
                        response=raw_response,
                        topic=chunk_value,
                        think="",
                        memory=chunk_value,
                        metadata=metadata
                    )

        # 不应该到达这里，但为了安全起见
        return TopicRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            topic=chunk_value,
            think="",
            memory=chunk_value,
            metadata={'extraction_failed': True, 'failure_reason': 'unknown'}
        )
        
        
        

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput], Dict[str, TopicRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput], Dict[str, TopicRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
                - A dict with keys as the chunk ids and values as the topic extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}
        
        # dev add: topic extration
        print(f"[dev] [OpenIE] [batch_openie] chunk_passages (len={len(chunk_passages)}): \n{chunk_passages[next(iter(chunk_passages))]}") # dev add
        print(f"[dev] [OpenIE] [batch_openie] chunk_passages (key_type={type(next(iter(chunk_passages)))})") # dev add
        print(f"[dev] [OpenIE] [batch_openie] chunk_passages (value_type={type(chunk_passages[next(iter(chunk_passages))])})") # dev add

        # 存储 topic extraction 结果，包含 think 和 memory
        topic_extraction_results = {}

        topic_prompt_tokens = 0
        topic_completion_tokens = 0
        topic_cache_hit = 0
        topic_start = time.time()

        chunk_idx = 0
        chunk_number = len(chunk_passages)
        for chunk_key, chunk_value in chunk_passages.items(): # dev add

            chunk_idx += 1
            if chunk_idx % 10 == 0:
                print(f"[dev] [OpenIE] [batch_openie] process: {chunk_idx}/{chunk_number}") # dev add

            topic_result = self.topic_extraction(chunk_key, chunk_value)
            topic_extraction_results[chunk_key] = topic_result
            topic_metadata = getattr(topic_result, "metadata", {}) or {}
            topic_prompt_tokens += topic_metadata.get('prompt_tokens', 0)
            topic_completion_tokens += topic_metadata.get('completion_tokens', 0)
            if topic_metadata.get('cache_hit'):
                topic_cache_hit += 1
            # 使用 memory 替代原来的 topic 用于后续的 NER 和三元组提取
            chunk_passages[chunk_key] = topic_result.memory # dev modified: 使用 memory 而不是 topic

        topic_time = time.time() - topic_start
        
        ner_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        ner_start = time.time()
        with ThreadPoolExecutor() as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1

                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        ner_time = time.time() - ner_start

        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        triple_start = time.time()
        with ThreadPoolExecutor() as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(self.triple_extraction, ner_result.chunk_id,
                                chunk_passages[ner_result.chunk_id],
                                ner_result.unique_entities): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        triple_time = time.time() - triple_start

        self.last_openie_stats = {
            'topic': {
                'prompt_tokens': topic_prompt_tokens,
                'completion_tokens': topic_completion_tokens,
                'cache_hits': topic_cache_hit,
                'time_s': topic_time,
            },
            'ner': {
                'prompt_tokens': ner_results_list and sum(res.metadata.get('prompt_tokens', 0) for res in ner_results_list) or 0,
                'completion_tokens': ner_results_list and sum(res.metadata.get('completion_tokens', 0) for res in ner_results_list) or 0,
                'cache_hits': ner_results_list and sum(1 for res in ner_results_list if res.metadata.get('cache_hit')) or 0,
                'time_s': ner_time,
            },
            'triple': {
                'prompt_tokens': triple_results_list and sum(res.metadata.get('prompt_tokens', 0) for res in triple_results_list) or 0,
                'completion_tokens': triple_results_list and sum(res.metadata.get('completion_tokens', 0) for res in triple_results_list) or 0,
                'cache_hits': triple_results_list and sum(1 for res in triple_results_list if res.metadata.get('cache_hit')) or 0,
                'time_s': triple_time,
            },
        }
        self.last_openie_stats['total'] = {
            'prompt_tokens': (
                self.last_openie_stats['topic']['prompt_tokens']
                + self.last_openie_stats['ner']['prompt_tokens']
                + self.last_openie_stats['triple']['prompt_tokens']
            ),
            'completion_tokens': (
                self.last_openie_stats['topic']['completion_tokens']
                + self.last_openie_stats['ner']['completion_tokens']
                + self.last_openie_stats['triple']['completion_tokens']
            ),
            'cache_hits': (
                self.last_openie_stats['topic']['cache_hits']
                + self.last_openie_stats['ner']['cache_hits']
                + self.last_openie_stats['triple']['cache_hits']
            ),
            'time_s': topic_time + ner_time + triple_time,
        }

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        # 返回 topic_extraction_results 用于后续存储
        return ner_results_dict, triple_results_dict, topic_extraction_results
