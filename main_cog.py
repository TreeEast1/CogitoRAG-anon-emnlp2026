#!/usr/bin/env python3
"""
统一入口脚本: main_cog.py

用途:
- 统一调度数据集 / LLM / Embedding
- 可配置 topic / NER / triple 的 prompt 模板
- 可配置奖励函数 (entity_alpha / entity_beta)
- 可配置 dense 融合参数 (alpha / gamma / rrf)

示例:
  python main_cog.py --dataset musique --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2 \
    --topic_prompt_name topic_extraction_fine_grained_v3 --ner_prompt_name ner_v2 --triple_prompt_name triple_extraction_v2 \
    --entity_alpha 4.0 --entity_beta 3.0 --dense_fuse_gamma 0.5
"""

import os
import sys
import json
import argparse
import logging
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

# 兼容直接运行脚本时的本地导入
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.cogitorag import TAG, BaseConfig, string_to_bool

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name and dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples: List) -> List:
    gold_answers = []
    for sample in samples:
        gold_ans = None
        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])
        gold_answers.append(gold_ans)
    return gold_answers


def patch_openie_templates(
    ner_prompt: str | None,
    triple_prompt: str | None,
    topic_prompt: str | None
) -> None:
    """
    Monkey patch OpenIE to use custom prompt templates for NER / Triple / Topic.
    """
    if not any([ner_prompt, triple_prompt, topic_prompt]):
        return

    import src.cogitorag.information_extraction.openie_openai as openie_module

    def patched_ner(self, chunk_key: str, passage: str):
        prompt_name = ner_prompt or 'ner'
        ner_input_message = self.prompt_template_manager.render(name=prompt_name, passage=passage)
        raw_response = ""
        metadata = {}
        try:
            raw_response, metadata, cache_hit = self.llm_model.infer(messages=ner_input_message)
            metadata['cache_hit'] = cache_hit
            metadata['template'] = prompt_name
            if metadata['finish_reason'] == 'length':
                real_response = openie_module.fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_entities = openie_module._extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))
        except Exception as e:
            logging.warning(e)
            metadata.update({'error': str(e)})
            return openie_module.NerRawOutput(
                chunk_id=chunk_key, response=raw_response, unique_entities=[], metadata=metadata
            )
        return openie_module.NerRawOutput(
            chunk_id=chunk_key, response=raw_response, unique_entities=unique_entities, metadata=metadata
        )

    def patched_triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]):
        def _extract_triples_from_response(real_response):
            pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is None:
                return []
            return eval(match.group())["triples"]

        prompt_name = triple_prompt or 'triple_extraction'
        messages = self.prompt_template_manager.render(
            name=prompt_name,
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        try:
            raw_response, metadata, cache_hit = self.llm_model.infer(messages=messages)
            metadata['cache_hit'] = cache_hit
            metadata['template'] = prompt_name
            if metadata['finish_reason'] == 'length':
                real_response = openie_module.fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = openie_module.filter_invalid_triples(triples=extracted_triples)
        except Exception as e:
            logging.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return openie_module.TripleRawOutput(
                chunk_id=chunk_key, response=raw_response, metadata=metadata, triples=[]
            )

        return openie_module.TripleRawOutput(
            chunk_id=chunk_key, response=raw_response, metadata=metadata, triples=triplets
        )

    def patched_topic_extraction(self, chunk_key, chunk_value, max_retries=3):
        prompt_name = topic_prompt or getattr(self, 'topic_prompt_name', 'topic_extraction')
        retry_count = 0
        metadata = {}
        raw_response = ""

        while retry_count < max_retries:
            try:
                messages = self.prompt_template_manager.render(name=prompt_name, passage=chunk_value)
                raw_response, metadata, cache_hit = self.topic_extraction_llm.infer(messages=messages)
                metadata['cache_hit'] = cache_hit
                metadata['retry_count'] = retry_count
                metadata['template'] = prompt_name

                if metadata['finish_reason'] == 'length':
                    real_response = openie_module.fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response

                is_valid, think, memory = self._validate_think_memory_output(real_response)

                if is_valid:
                    return openie_module.TopicRawOutput(
                        chunk_id=chunk_key,
                        response=raw_response,
                        topic=memory,
                        think=think,
                        memory=memory,
                        metadata=metadata
                    )

                memory_match = re.search(r'<memory>(.*?)</memory>', real_response, re.DOTALL)
                fallback_memory = memory_match.group(1).strip() if memory_match else ""
                if fallback_memory:
                    metadata['fallback_reason'] = 'missing_think_tag'
                    return openie_module.TopicRawOutput(
                        chunk_id=chunk_key,
                        response=raw_response,
                        topic=fallback_memory,
                        think="(missing_think_tag)",
                        memory=fallback_memory,
                        metadata=metadata
                    )
                elif real_response.strip():
                    metadata['fallback_reason'] = 'missing_tags'
                    return openie_module.TopicRawOutput(
                        chunk_id=chunk_key,
                        response=raw_response,
                        topic=real_response.strip(),
                        think="(missing_tags)",
                        memory=real_response.strip(),
                        metadata=metadata
                    )

                retry_count += 1
                logging.warning(
                    f"Chunk {chunk_key} topic extraction failed (attempt {retry_count}/{max_retries}). "
                    f"Missing or empty think/memory tags."
                )

            except Exception as e:
                retry_count += 1
                logging.warning(
                    f"Chunk {chunk_key} topic extraction exception (attempt {retry_count}/{max_retries}): {e}"
                )
                metadata = {'error': str(e), 'retry_count': retry_count, 'template': prompt_name}

        return openie_module.TopicRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            topic=chunk_value,
            think="(fallback_original)",
            memory=chunk_value,
            metadata=metadata
        )

    openie_module.OpenIE.ner = patched_ner
    openie_module.OpenIE.triple_extraction = patched_triple_extraction
    if topic_prompt:
        openie_module.OpenIE.topic_extraction = patched_topic_extraction


def load_corpus_and_samples(dataset_name: str, corpus_path: str | None, qa_path: str | None) -> Tuple[List[Dict], List[Dict]]:
    corpus_path = corpus_path or f"reproduce/dataset/{dataset_name}_corpus.json"
    qa_path = qa_path or f"reproduce/dataset/{dataset_name}.json"

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    with open(qa_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    return corpus, samples


def build_passage_to_memory_map(openie_results_path: str) -> Dict[str, str]:
    with open(openie_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = data.get('docs', []) if isinstance(data, dict) else []
    passage_to_memory: Dict[str, str] = {}
    for doc in docs:
        passage = doc.get('passage')
        if not passage:
            continue
        memory = doc.get('memory', passage)
        passage_to_memory[passage] = memory

    print(f"[INFO] 构建 passage->memory 映射: {len(passage_to_memory)} 条")
    return passage_to_memory


def sanitize_args_for_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    raw = vars(args).copy()
    sensitive_keys = {"api_key", "topic_api_key"}
    sanitized = {}
    for k, v in raw.items():
        if k in sensitive_keys and v:
            sanitized[k] = "***"
        else:
            sanitized[k] = v
    return sanitized


def build_command_text_from_args(args_dict: Dict[str, Any], explicit_command: str | None = None) -> str:
    if explicit_command:
        return explicit_command

    argv_tokens: List[str] = []
    for key, value in args_dict.items():
        if key == "run_command_text" or value is None or value is False:
            continue
        flag = f"--{key}"
        if value is True:
            argv_tokens.append(flag)
        else:
            argv_tokens.extend([flag, str(value)])
    return ("python main_cog.py " + " ".join(argv_tokens)).strip()


def build_config(args: argparse.Namespace, corpus_len: int) -> BaseConfig:
    config = BaseConfig(
        save_dir=args.save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        embedding_base_url=args.embedding_base_url,
        azure_endpoint=args.azure_endpoint,
        azure_embedding_endpoint=args.azure_embedding_endpoint,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=string_to_bool(args.force_index_from_scratch),
        force_openie_from_scratch=string_to_bool(args.force_openie_from_scratch),
        rerank_dspy_file_path=args.rerank_dspy_file_path,
        use_llm_rerank=bool(args.enable_llm_rerank),
        retrieval_top_k=args.retrieval_top_k,
        linking_top_k=args.linking_top_k,
        max_qa_steps=args.max_qa_steps,
        qa_top_k=args.qa_top_k,
        qa_prompt_name=args.qa_prompt_name,
        graph_type=args.graph_type,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_seq_len=args.embedding_max_seq_len,
        embedding_model_dtype=args.embedding_model_dtype,
        max_new_tokens=args.max_new_tokens,
        corpus_len=corpus_len,
        openie_mode=args.openie_mode,
        topic_extraction_mode=args.topic_extraction_mode,
        topic_extraction_llm_name=args.topic_llm_name,
        topic_extraction_azure_endpoint=args.topic_azure_endpoint,
        topic_extraction_api_key=args.topic_api_key,
        topic_extraction_api_version=args.topic_api_version,
        ppr_topk=args.ppr_topk,
        dense_rerank_topk=args.dense_rerank_topk,
        dense_fuse_alpha=args.dense_fuse_alpha,
        passage_node_weight=args.passage_node_weight,
    )

    if args.dense_fuse_gamma is not None:
        setattr(config, "dense_fuse_gamma", args.dense_fuse_gamma)
    if args.dense_fuse_rrf:
        setattr(config, "dense_fuse_rrf", True)
        setattr(config, "dense_fuse_rrf_k", args.dense_fuse_rrf_k)
    if args.entity_alpha is not None:
        setattr(config, "entity_alpha", args.entity_alpha)
    if args.entity_beta is not None:
        setattr(config, "entity_beta", args.entity_beta)
    if args.graph_working_dir:
        setattr(config, "graph_working_dir", args.graph_working_dir)
    if args.memory_map_path:
        setattr(config, "memory_map_path", args.memory_map_path)
    if args.run_command_text:
        setattr(config, "run_command_text", args.run_command_text)

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="CogitoRAG unified runner (main_cog)")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--llm_base_url', type=str, default=os.getenv('LLM_BASE_URL'), help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default=os.getenv('LLM_NAME', 'gpt-4o-mini'), help='LLM name')
    parser.add_argument('--embedding_name', type=str, default=os.getenv('EMBEDDING_NAME', 'nvidia/NV-Embed-v2'), help='Embedding model name')
    parser.add_argument('--embedding_base_url', type=str, default=os.getenv('EMBEDDING_BASE_URL'), help='Embedding base URL')
    parser.add_argument('--azure_endpoint', type=str, default=os.getenv('AZURE_ENDPOINT'), help='Azure LLM endpoint')
    parser.add_argument('--azure_embedding_endpoint', type=str, default=os.getenv('AZURE_EMBEDDING_ENDPOINT'), help='Azure embedding endpoint')
    parser.add_argument('--api_key', type=str, default=os.getenv('OPENAI_API_KEY'), help='API key (sets OPENAI_API_KEY)')

    parser.add_argument('--force_index_from_scratch', type=str, default='false')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online')
    parser.add_argument('--topic_extraction_mode', choices=['default', 'fine_grained'], default='fine_grained')

    parser.add_argument('--topic_prompt_name', type=str, default='topic_extraction_fine_grained_v3', help='Topic prompt template name')
    parser.add_argument('--ner_prompt_name', type=str, default='ner_v2', help='NER prompt template name')
    parser.add_argument('--triple_prompt_name', type=str, default='triple_extraction_v2', help='Triple extraction prompt template name')

    parser.add_argument('--topic_llm_name', type=str, default=os.getenv('TOPIC_LLM_NAME', 'gpt-4o-mini'))
    parser.add_argument('--topic_azure_endpoint', type=str, default=os.getenv('TOPIC_AZURE_ENDPOINT'))
    parser.add_argument('--topic_api_key', type=str, default=os.getenv('TOPIC_API_KEY', os.getenv('OPENAI_API_KEY')))
    parser.add_argument('--topic_api_version', type=str, default='2024-02-15-preview')

    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='outputs')
    parser.add_argument('--corpus_path', type=str, default=None)
    parser.add_argument('--qa_path', type=str, default=None)
    parser.add_argument('--enable_split', type=str, default='true')
    parser.add_argument('--index_mode', choices=['topic', 'basic', 'none'], default='topic')

    parser.add_argument('--rerank_dspy_file_path', type=str, default='src/cogitorag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json')
    parser.add_argument('--enable_llm_rerank', action='store_true', help='Enable LLM rerank (DSPyFilter). Disabled by default.')

    parser.add_argument('--retrieval_top_k', type=int, default=200)
    parser.add_argument('--linking_top_k', type=int, default=5)
    parser.add_argument('--max_qa_steps', type=int, default=3)
    parser.add_argument('--qa_top_k', type=int, default=5)
    parser.add_argument('--qa_prompt_name', type=str, default='rag_qa_musique', help='QA prompt template name')
    parser.add_argument('--qa_output_json', type=str, default=None, help='Path to save QA predictions JSON (GraphRAG-Bench format)')
    parser.add_argument('--graph_working_dir', type=str, default=None, help='Override TAG working directory to an existing graph directory containing graph.pickle')
    parser.add_argument('--memory_map_path', type=str, default=None, help='Path to openie_results_ner_*.json used to build passage->memory mapping for QA prompt augmentation')
    parser.add_argument('--run_command_text', type=str, default=None, help='Optional explicit command string to store in run metadata (falls back to sys.argv)')
    parser.add_argument('--graph_type', type=str, default='facts_and_sim_passage_node_unidirectional')
    parser.add_argument('--embedding_batch_size', type=int, default=5)
    parser.add_argument('--embedding_max_seq_len', type=int, default=2048)
    parser.add_argument('--embedding_model_dtype', type=str, default='auto')
    parser.add_argument('--max_new_tokens', type=int, default=None)

    parser.add_argument('--ppr_topk', type=int, default=12)
    parser.add_argument('--dense_rerank_topk', type=int, default=200)
    parser.add_argument('--dense_fuse_alpha', type=float, default=0.5)
    parser.add_argument('--dense_fuse_gamma', type=float, default=0.95)
    parser.add_argument('--dense_fuse_rrf', action='store_true', help='Enable RRF fusion')
    parser.add_argument('--dense_fuse_rrf_k', type=float, default=60.0)

    parser.add_argument('--entity_alpha', type=float, default=None)
    parser.add_argument('--entity_beta', type=float, default=None)
    parser.add_argument('--passage_node_weight', type=float, default=0.05)

    parser.add_argument('--cuda_visible_devices', type=str, default=None)
    parser.add_argument('--hf_endpoint', type=str, default=os.getenv('HF_ENDPOINT', 'https://hf-mirror.com'))
    parser.add_argument('--hf_home', type=str, default=os.getenv('HF_HOME'))
    parser.add_argument('--transformers_cache', type=str, default=os.getenv('TRANSFORMERS_CACHE'))

    args = parser.parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    if args.transformers_cache:
        os.environ["TRANSFORMERS_CACHE"] = args.transformers_cache

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    topic_prompt = args.topic_prompt_name
    if not topic_prompt:
        topic_prompt = "topic_extraction_fine_grained" if args.topic_extraction_mode == "fine_grained" else "topic_extraction"

    patch_openie_templates(
        ner_prompt=args.ner_prompt_name,
        triple_prompt=args.triple_prompt_name,
        topic_prompt=topic_prompt
    )

    save_dir = args.save_dir
    dataset_name = args.dataset
    save_dir = os.path.join(save_dir, dataset_name)
    args.save_dir = save_dir

    run_start_ts = time.time()
    run_start_iso = datetime.now().isoformat(timespec="seconds")

    corpus, samples = load_corpus_and_samples(dataset_name, args.corpus_path, args.qa_path)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    all_queries = [s['question'] for s in samples]
    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert len(all_queries) == len(gold_docs) == len(gold_answers)
    except Exception:
        gold_docs = None

    select_number = args.max_samples
    if select_number:
        all_queries = all_queries[:select_number]
        gold_answers = gold_answers[:select_number]
        if gold_docs is not None:
            gold_docs = gold_docs[:select_number]
        samples = samples[:select_number]

    config = build_config(args=args, corpus_len=len(corpus))

    logging.basicConfig(level=logging.INFO)

    sanitized_args = sanitize_args_for_metadata(args)
    run_metadata: Dict[str, Any] = {
        "timestamp_start": run_start_iso,
        "dataset": dataset_name,
        "args": sanitized_args,
        "save_dir": args.save_dir,
        "graph_working_dir": args.graph_working_dir,
        "memory_map_path": args.memory_map_path,
        "qa_use_memory": os.environ.get("QA_USE_MEMORY", "").lower() in {"1", "true", "yes"},
        "qa_prompt_name_requested": args.qa_prompt_name,
        "topic_prompt_name": topic_prompt,
        "ner_prompt_name": args.ner_prompt_name,
        "triple_prompt_name": args.triple_prompt_name,
        "llm_name": args.llm_name,
        "embedding_name": args.embedding_name,
        "qa_top_k": args.qa_top_k,
        "retrieval_top_k": args.retrieval_top_k,
        "max_qa_steps": args.max_qa_steps,
        "enable_split": string_to_bool(args.enable_split),
        "index_mode": args.index_mode,
        "output_files": {},
    }
    run_metadata["command"] = build_command_text_from_args(sanitized_args, args.run_command_text)

    cognitive_rag = TAG(global_config=config)

    if args.memory_map_path:
        passage_to_memory = build_passage_to_memory_map(args.memory_map_path)
        cognitive_rag.passage_to_memory = passage_to_memory

    if hasattr(cognitive_rag, "last_run_metadata"):
        cognitive_rag.last_run_metadata = run_metadata

    if args.index_mode == "topic":
        cognitive_rag.topic_index(docs)
    elif args.index_mode == "basic":
        cognitive_rag.index(docs)

    enable_split = string_to_bool(args.enable_split)
    outputs = cognitive_rag.rag_qa(
        queries=all_queries,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
        enable_split=enable_split
    )

    queries_solutions = None
    all_response_message = None
    all_metadata = None
    retrieval_metrics = None
    qa_metrics = None

    if isinstance(outputs, tuple):
        if len(outputs) >= 3:
            queries_solutions, all_response_message, all_metadata = outputs[:3]
        if len(outputs) >= 5:
            retrieval_metrics, qa_metrics = outputs[3], outputs[4]

    if queries_solutions is not None and all_response_message is not None:
        qa_predictions = []
        for sample, qsol, response in zip(samples, queries_solutions, all_response_message):
            retrieved_passages = list(getattr(qsol, 'docs', [])[:args.qa_top_k]) if getattr(qsol, 'docs', None) else []
            answer_text = getattr(qsol, 'answer', None) or response
            qa_predictions.append({
                "id": sample.get("id"),
                "question": sample.get("question"),
                "source": sample.get("source"),
                "context": retrieved_passages,
                "evidence": sample.get("evidence", []),
                "question_type": sample.get("question_type"),
                "generated_answer": answer_text,
                "ground_truth": sample.get("answer", "")
            })

        qa_output_path = args.qa_output_json or os.path.join(args.save_dir, f"qa_predictions_{dataset_name}.json")
        os.makedirs(os.path.dirname(qa_output_path), exist_ok=True)
        with open(qa_output_path, "w", encoding="utf-8") as f:
            json.dump(qa_predictions, f, indent=2, ensure_ascii=False)
        run_metadata.setdefault("output_files", {})["qa_predictions_json"] = qa_output_path

    if retrieval_metrics is not None or qa_metrics is not None:
        print("\n" + "=" * 70)
        print(f"实验配置 -> 数据集: {dataset_name} | LLM: {args.llm_name} | Embedding: {args.embedding_name}")
        print(f"           gamma: {args.dense_fuse_gamma} | ppr_topk: {args.ppr_topk} | qa_top_k: {args.qa_top_k}")
        if args.entity_alpha is not None or args.entity_beta is not None:
            print(f"           entity_alpha: {args.entity_alpha} | entity_beta: {args.entity_beta}")
        print("-" * 70)
        if retrieval_metrics:
            print("检索评价:")
            for k, v in retrieval_metrics.items():
                print(f"  {k:<12}: {v:.4f}")
        if qa_metrics:
            print("QA 评价:")
            for k, v in qa_metrics.items():
                print(f"  {k:<12}: {v:.4f}")

        # 时间与 Token 统计
        index_stats = getattr(cognitive_rag, 'last_index_stats', {})
        retrieval_stats = getattr(cognitive_rag, 'last_retrieval_stats', {})
        qa_stats = getattr(cognitive_rag, 'last_qa_stats', {})
        print("时间统计:")
        print(f"  {'index_time':<20}: {index_stats.get('index_time_s', 0):.2f}s")
        print(f"  {'retrieval_time':<20}: {retrieval_stats.get('time_s', 0):.2f}s")
        print(f"  {'qa_time':<20}: {qa_stats.get('time_s', 0):.2f}s")
        total_time = index_stats.get('index_time_s', 0) + retrieval_stats.get('time_s', 0) + qa_stats.get('time_s', 0)
        print(f"  {'total_time':<20}: {total_time:.2f}s")
        print("Token 统计:")
        total_prompt = retrieval_stats.get('prompt_tokens', 0) + qa_stats.get('prompt_tokens', 0) + index_stats.get('index_prompt_tokens', 0)
        total_completion = retrieval_stats.get('completion_tokens', 0) + qa_stats.get('completion_tokens', 0) + index_stats.get('index_completion_tokens', 0)
        print(f"  {'retrieval_tokens':<20}: prompt={retrieval_stats.get('prompt_tokens', 0)}, completion={retrieval_stats.get('completion_tokens', 0)}")
        print(f"  {'qa_tokens':<20}: prompt={qa_stats.get('prompt_tokens', 0)}, completion={qa_stats.get('completion_tokens', 0)}")
        print(f"  {'total_tokens':<20}: prompt={total_prompt}, completion={total_completion}, total={total_prompt + total_completion}")

        # 保存详细结果到 JSON
        summary = {
            "dataset": dataset_name,
            "gamma": args.dense_fuse_gamma,
            "llm": args.llm_name,
            "embedding": args.embedding_name,
            "ppr_topk": args.ppr_topk,
            "qa_top_k": args.qa_top_k,
            "entity_alpha": args.entity_alpha,
            "entity_beta": args.entity_beta,
            "retrieval_metrics": retrieval_metrics if retrieval_metrics else {},
            "qa_metrics": qa_metrics if qa_metrics else {},
            "time": {
                "index_s": round(index_stats.get('index_time_s', 0), 2),
                "retrieval_s": round(retrieval_stats.get('time_s', 0), 2),
                "qa_s": round(qa_stats.get('time_s', 0), 2),
                "total_s": round(total_time, 2),
            },
            "tokens": {
                "retrieval_prompt": retrieval_stats.get('prompt_tokens', 0),
                "retrieval_completion": retrieval_stats.get('completion_tokens', 0),
                "qa_prompt": qa_stats.get('prompt_tokens', 0),
                "qa_completion": qa_stats.get('completion_tokens', 0),
                "total_prompt": total_prompt,
                "total_completion": total_completion,
                "total": total_prompt + total_completion,
            },
        }
        summary_path = os.path.join(args.save_dir, f"summary_gamma_{args.dense_fuse_gamma}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        run_metadata.setdefault("output_files", {})["summary_json"] = summary_path
        print(f"\n详细结果已保存: {summary_path}")
        print("=" * 70)

    run_end_ts = time.time()
    run_end_iso = datetime.now().isoformat(timespec="seconds")
    run_metadata["timestamp_end"] = run_end_iso
    run_metadata["duration_s"] = round(run_end_ts - run_start_ts, 3)
    run_metadata["graph_pickle_path"] = getattr(cognitive_rag, "_graph_pickle_filename", None)
    run_metadata["qa_prompt_name_resolved"] = getattr(cognitive_rag, "last_qa_prompt_name", None)
    run_metadata["qa_prompt_path"] = getattr(cognitive_rag, "last_qa_prompt_path", None)
    run_metadata["stage_stats"] = {
        "index": getattr(cognitive_rag, "last_index_stats", {}),
        "retrieval": getattr(cognitive_rag, "last_retrieval_stats", {}),
        "qa": getattr(cognitive_rag, "last_qa_stats", {}),
    }

    if getattr(cognitive_rag, "last_result_json_path", None):
        run_metadata.setdefault("output_files", {})["result_json"] = cognitive_rag.last_result_json_path
    if getattr(cognitive_rag, "last_result_with_meta_json_path", None):
        run_metadata.setdefault("output_files", {})["result_with_meta_json"] = cognitive_rag.last_result_with_meta_json_path

    if hasattr(cognitive_rag, "last_run_metadata"):
        cognitive_rag.last_run_metadata = run_metadata

    result_with_meta_path = getattr(cognitive_rag, "last_result_with_meta_json_path", None)
    result_json_path = getattr(cognitive_rag, "last_result_json_path", None)
    if result_with_meta_path and result_json_path:
        with open(result_json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        with open(result_with_meta_path, "w", encoding="utf-8") as f:
            json.dump({"run_metadata": run_metadata, "records": records}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
