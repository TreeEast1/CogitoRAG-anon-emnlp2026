import json
import os
import sys
import logging
from dataclasses import dataclass, field, asdict
# from datetime import datetime # dev delete
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time
import copy

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig
from .utils.qa_utils import extract_final_answer

from datetime import datetime # dev add
import shutil # dev add
import subprocess # dev add

logger = logging.getLogger(__name__)

class TAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific CogitoRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"CogitoRAG init with config:\n  {_print_config}\n")
        self.last_index_stats = {}
        self.last_retrieval_stats = {}
        self.last_qa_stats = {}
        self.last_split_stats = {}
        self.last_qa_prompt_name = None
        self.last_qa_prompt_name_requested = getattr(self.global_config, "qa_prompt_name", None)
        self.last_qa_prompt_path = None
        self.last_result_json_path = None
        self.last_result_with_meta_json_path = None
        self.last_run_metadata = {}
        self._rerank_token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "cache_hits": 0}

        graph_working_dir = getattr(self.global_config, "graph_working_dir", None)
        if graph_working_dir:
            self.working_dir = graph_working_dir
        else:
            llm_label = self.global_config.llm_name.replace("/", "_")
            embedding_label = self.global_config.embedding_model_name.replace("/", "_")
            self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)


        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        # Create separate LLM for topic extraction if configured
        self.topic_extraction_llm = None
        if hasattr(self.global_config, 'topic_extraction_llm_name') and self.global_config.topic_extraction_llm_name:
            from .llm.openai_gpt import CacheOpenAI
            cache_dir = os.path.join(self.global_config.save_dir, "llm_cache")
            self.topic_extraction_llm = CacheOpenAI.from_custom_config(
                cache_dir=cache_dir,
                llm_name=self.global_config.topic_extraction_llm_name,
                azure_endpoint=self.global_config.topic_extraction_azure_endpoint,
                api_key=self.global_config.topic_extraction_api_key,
                api_version=getattr(self.global_config, 'topic_extraction_api_version', "2024-02-15-preview")
            )
            print(
                f"[TAG] Using separate LLM for topic extraction: {self.global_config.topic_extraction_llm_name} "
                f"(endpoint={self.global_config.topic_extraction_azure_endpoint})"
            )

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(
                llm_model=self.llm_model,
                topic_extraction_mode=self.global_config.topic_extraction_mode,
                topic_extraction_llm=self.topic_extraction_llm
            )
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)

        self.graph = self.initialize_graph()

        print("step3: graph loaded")

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            print("step3.1: get embedding class")
            EmbeddingModelClass = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name
            )
            print("step3.2: got embedding class", EmbeddingModelClass)
            self.embedding_model = EmbeddingModelClass(
                global_config=self.global_config,
                embedding_model_name=self.global_config.embedding_model_name
            )
            print("step4: embedding model loaded")


        # if self.global_config.openie_mode == 'offline':
        #     self.embedding_model = None
        # else:
        #     self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
        #         embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
        #                                                                       embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.think_storage_dir = os.path.join(self.global_config.save_dir, 'think_storage')
        os.makedirs(self.think_storage_dir, exist_ok=True)

        self.failed_extraction_dir = os.path.join(self.global_config.save_dir, 'failed_extractions')
        os.makedirs(self.failed_extraction_dir, exist_ok=True)

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None


    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def TAG_index(self, docs: List[str]):
        """
        
        python -m graphrag init --root .
        mkdir ./input
        python -m graphrag index --root .
        python -m graphrag query --method local --query "好的教育评价体系的标准是什么" --rerank
        
        """
        
        print(f"[dev] [TAG] [TAG_index]")
        
        TAG_data_dir = os.path.join(self.global_config.save_dir, "cognitive_graph_tag_data")
        if not os.path.exists(TAG_data_dir):
            os.makedirs(TAG_data_dir)

        time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        time_dir = os.path.join(TAG_data_dir, time_stamp)
        os.makedirs(time_dir, exist_ok=True)
        
        input_dir = os.path.join(time_dir, "input")
        os.makedirs(input_dir, exist_ok=True)

        print(f"Created input directory at: {input_dir}")
        
        static_dir = os.path.join(time_dir, "static")
        os.makedirs(static_dir, exist_ok=True)
        
        print(f"Created static directory at: {static_dir}")
        
        settings_index_path = os.path.join(TAG_data_dir, "settings_index.yaml")
        settings_path = f"{time_dir}/settings.yaml"


        subprocess.run(["python", "-m", "graphrag", "init", "--root", time_dir], check=True)

        settings_index_path = os.path.join(TAG_data_dir, "settings_index.yaml")
        settings_path = os.path.join(time_dir, "settings.yaml")
        shutil.copyfile(settings_index_path, settings_path)
        print(f"Copied {settings_index_path} to {settings_path}")

        # for i, doc in enumerate(docs, 1):
        #     file_path = os.path.join(input_dir, f"{i}.txt")
        #     with open(file_path, "w", encoding="utf-8") as f:
        #         f.write(doc)
        # print(f"Wrote {len(docs)} documents to {input_dir}")
        docs_per_file = 500
        for i in range(0, len(docs), docs_per_file):
            file_number = i // docs_per_file + 1
            file_path = os.path.join(input_dir, f"{file_number}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                for doc in docs[i:i+docs_per_file]:
                    f.write(doc + "\n")
        print(f"Wrote {len(docs)} documents to {input_dir}")

        subprocess.run(["python", "-m", "graphrag", "index", "--root", time_dir], check=True)
        print("Indexing completed.")
        
        return time_dir
    
    
    def topic_index(self, docs: List[str]):
        """
        Indexes the given documents using the CogitoRAG graph-indexing pipeline, generating an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """
        index_start_time = time.time()

        print(f"[dev] [TAG] [topic_index] len(docs): {len(docs)}")

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            print(f"[dev] [TAG] [topic_index] self.global_config.openie_mode == 'offline'") # dev add
            
            self.pre_openie(docs)
        else:
            print(f"[dev] [TAG] [topic_index] self.global_config.openie_mode != 'offline'") # dev add
            
        # time.sleep(1000000) # dev add

        self.chunk_embedding_store.insert_strings(docs)
        
        print(f"[dev] [TAG] [topic_index] self.chunk_embedding_store.insert_strings(docs) OK") # dev add
        
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        print(f"[dev] [TAG] [topic_index] chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows() OK") # dev add        
        print(f"[dev] [TAG] [topic_index] chunk_to_rows (type={type(chunk_to_rows)})") # dev add        

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            print(f"[dev] [TAG] [topic_index] len(chunk_keys_to_process) > 0") # dev add

            new_ner_results_dict, new_triple_results_dict, new_topic_results_dict = self.openie.batch_openie(new_openie_rows)

            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict, new_topic_results_dict)

        if self.global_config.save_openie:
            print(f"[dev] [TAG] [topic_index] self.global_config.save_openie")
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict)

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())
        
        print(f"[dev] chunk_ids (type={type(chunk_ids)}): \n{chunk_ids[:2]}") # dev add

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        
        print(f"[dev] chunk_triples (type={type(chunk_triples)}): \n{chunk_triples[:2]}") # dev add
        
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        
        print(f"[dev] entity_nodes (type={type(entity_nodes)}): \n{entity_nodes[:2]}") # dev add
        print(f"[dev] chunk_triple_entities (type={type(chunk_triple_entities)}): \n{chunk_triple_entities[:2]}") # dev add
        
        facts = flatten_facts(chunk_triples)
        
        print(f"[dev] facts (type={type(facts)}): \n{facts[:2]}") # dev add

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])
        
        # time.sleep(100000) # dev add

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()
        
        # print(f"[dev] [TAG] [topic_index] self.ent_node_to_chunk_ids: \n{self.ent_node_to_chunk_ids}")
        # print(f"[dev] [TAG] [topic_index] len(self.ent_node_to_chunk_ids): \n{len(self.ent_node_to_chunk_ids)}")

        # # print(f"[dev] [TAG] [topic_index] self.node_to_node_stats: \n{self.node_to_node_stats}")
        # print(f"[dev] [TAG] [topic_index] len(self.node_to_node_stats): \n{len(self.node_to_node_stats)}")
        # print(f"[dev] [TAG] [topic_index] sum(self.node_to_node_stats.values()): \n{sum(self.node_to_node_stats.values())}")
        
        self.entity_to_entity_list = {} # dict{str: list(str)}
        for (ent1, ent2) in self.node_to_node_stats.keys():
            if ent1 not in self.entity_to_entity_list:
                self.entity_to_entity_list[ent1] = []
            if ent2 not in self.entity_to_entity_list:
                self.entity_to_entity_list[ent2] = []
            
            self.entity_to_entity_list[ent1].append(ent2)
            self.entity_to_entity_list[ent2].append(ent1)

        self.entity_to_triple_list = {} # dict{str: list(str)}
        # for (ent1, ent2) in self.node_to_node_stats.keys():
        #     if ent1 not in self.entity_to_triple_list:
        #         self.entity_to_triple_list[ent1] = []
        #     if ent2 not in self.entity_to_triple_list:
        #         self.entity_to_triple_list[ent2] = []
            
        #     self.entity_to_triple_list[ent1].append(self.node_to_node_stats[(ent1, ent2)])
        #     self.entity_to_triple_list[ent2].append(self.node_to_node_stats[(ent1, ent2)])
        
        for triple in facts:
            ent1, relation, ent2 = triple
            if ent1 not in self.entity_to_triple_list:
                self.entity_to_triple_list[ent1] = []
            if ent2 not in self.entity_to_triple_list:
                self.entity_to_triple_list[ent2] = []
            
            self.entity_to_triple_list[ent1].append(triple)
            self.entity_to_triple_list[ent2].append(triple)
        
        print(f"[dev] (len={len(self.entity_to_triple_list)}) self.entity_to_triple_list[:10]: ") # dev add
        for k, v in list(self.entity_to_triple_list.items())[:10]:
            print(f"{k}: {v}")
        
        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        self.fact_id_to_row = self.fact_embedding_store.get_all_id_to_rows()

        index_total_time = time.time() - index_start_time
        openie_stats = getattr(self.openie, "last_openie_stats", {}) or {}
        topic_stats = openie_stats.get("topic", {}) or {}
        ner_stats = openie_stats.get("ner", {}) or {}
        triple_stats = openie_stats.get("triple", {}) or {}
        total_stats = openie_stats.get("total", {}) or {}

        memory_time = float(topic_stats.get("time_s", 0.0) or 0.0)
        post_memory_time = max(index_total_time - memory_time, 0.0)
        post_memory_prompt_tokens = (
            int(ner_stats.get("prompt_tokens", 0) or 0)
            + int(triple_stats.get("prompt_tokens", 0) or 0)
        )
        post_memory_completion_tokens = (
            int(ner_stats.get("completion_tokens", 0) or 0)
            + int(triple_stats.get("completion_tokens", 0) or 0)
        )

        self.last_index_stats = {
            "index_time_s": index_total_time,
            "index_prompt_tokens": int(total_stats.get("prompt_tokens", 0) or 0),
            "index_completion_tokens": int(total_stats.get("completion_tokens", 0) or 0),
            "memory_extraction": {
                "time_s": memory_time,
                "prompt_tokens": int(topic_stats.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(topic_stats.get("completion_tokens", 0) or 0),
            },
            "post_memory_graph": {
                "time_s": post_memory_time,
                "prompt_tokens": post_memory_prompt_tokens,
                "completion_tokens": post_memory_completion_tokens,
            },
            "openie_breakdown": {
                "topic": topic_stats,
                "ner": ner_stats,
                "triple": triple_stats,
            },
        }
        
        
    def index(self, docs: List[str]):
        """
        Indexes the given documents using the CogitoRAG graph-indexing pipeline, generating an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """
        
        print(f"[dev] [TAG] [index] len(docs): {len(docs)}")

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict)

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the CogitoRAG pipeline.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        #Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        #Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        #Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        #Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False
        
    def rerank_supports(self, query, query_fact_scores): # dev add
        print(f"[dev] [TAG] [rerank_supports]") # dev add
        
        """
        Step 1: filter facts/entities
        
        
        Step 2: find original texts
        
        
        Step 3: rerank texts using vector similarity
        
        
        Step 4: return texts in format
        """
        
        # load args
        # link_top_k: int = self.global_config.linking_top_k # 5
        link_top_k: int = 5
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            
            print(f"[dev] len(fact_row_dict): {len(fact_row_dict)}")
            print(f"[dev] fact_row_dict: {fact_row_dict}")
            
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            # Rerank the facts
            rerank_filter = False
            if rerank_filter:
                top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                    candidate_facts,
                                                                                    candidate_fact_indices,
                                                                                    len_after_rerank=link_top_k)
            else:    
                top_k_fact_indices = candidate_fact_indices
                top_k_facts = candidate_facts
            
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}


    def rerank_entities(self, query, query_entity_scores): # dev add
        # print(f"[dev] [TAG] [rerank_entities]") # dev add
        
        """
        Step 1: filter facts/entities
        
        
        Step 2: find original texts
        
        
        Step 3: rerank texts using vector similarity
        
        
        Step 4: return texts in format
        """
        
        # load args
        link_top_k: int = self.global_config.linking_top_k # 5
        # link_top_k: int = 5
        
        # Check if there are any entities to rerank
        if len(query_entity_scores) == 0 or len(self.entity_node_keys) == 0:
            logger.warning("No entities available for reranking. Returning empty lists.")
            return [], [], {'entities_before_rerank': [], 'entities_after_rerank': []}
            
        try:
            # Get the top k entitie by score
            if len(query_entity_scores) <= link_top_k:
                # If we have fewer entitie than requested, use all of them
                candidate_entity_indices = np.argsort(query_entity_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_entity_indices = np.argsort(query_entity_scores)[-link_top_k:][::-1].tolist()
            
            # print(f"[dev] len(self.entity_node_keys): {len(self.entity_node_keys)}") # dev add
            
            # Get the actual fact IDs
            real_candidate_entity_ids = [self.entity_node_keys[idx] for idx in candidate_entity_indices]
            
            # print(f"[dev] len(real_candidate_entity_ids): {len(real_candidate_entity_ids)}")
            # print(f"[dev] real_candidate_entity_ids: {real_candidate_entity_ids}")
            
            entity_row_dict = self.entity_embedding_store.get_rows(real_candidate_entity_ids)
            
            # print(f"[dev] len(entity_row_dict): {len(entity_row_dict)}")
            # print(f"[dev] entity_row_dict: {entity_row_dict}")

            # candidate_entities = [eval(entity_row_dict[id]['content']) for id in real_candidate_entity_ids]
            candidate_entities = [entity_row_dict[id]['content'] for id in real_candidate_entity_ids]
            
            # print(f"[dev] len(candidate_entities): {len(candidate_entities)}")
            
            # Rerank the entitie
            rerank_filter = False
            if rerank_filter:
                top_k_entity_indices, top_k_entities, reranker_dict = self.rerank_filter(query,
                                                                                    candidate_entities,
                                                                                    candidate_entity_indices,
                                                                                    len_after_rerank=link_top_k)
            else:    
                top_k_entity_indices = candidate_entity_indices
                top_k_entities = candidate_entities
            
            # print(f"[dev] len(candidate_entities) = {len(candidate_entities)}")
            # print(f"[dev] len(candidate_entity_indices) = {len(candidate_entity_indices)}")
            
            rerank_log = {'entities_before_rerank': candidate_entities, 'entities_after_rerank': top_k_entities}
            
            return top_k_entity_indices, top_k_entities, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_entitie: {str(e)}")
            return [], [], {'entities_before_rerank': [], 'entities_after_rerank': [], 'error': str(e)}



    def topic_retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the CogitoRAG pipeline, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k
        
        print(f"[dev] [TAG] [topic_retrieve] num_to_retrieve: {num_to_retrieve}") # dev add

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            rerank_start = time.time()
            self.global_config.linking_top_k = 20 # dev add

            # 1\2\5\10\15\20\30

            query_entity_scores = self.get_entity_scores(query)
            top_k_entity_indices, top_k_entities, rerank_log = self.rerank_entities(query, query_entity_scores)
            # print(f"[dev] [TAG] [retrieve] query_entity_scores (len={len(query_entity_scores)}: \n{query_entity_scores}") # dev add
            # print(f"[dev] [TAG] [retrieve] rerank_log['entities_before_rerank'] (len={len(rerank_log['entities_before_rerank'])}): \n{rerank_log['entities_before_rerank']}") # dev add
            # print(f"[dev] [TAG] [retrieve] rerank_log['entities_after_rerank'] (len={len(rerank_log['entities_after_rerank'])}): \n{rerank_log['entities_after_rerank']}") # dev add
            
            # self.global_config.linking_top_k = 5 # dev add
            # query_fact_scores = self.get_fact_scores(query)
            # # top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            # # top_k_fact_indices, top_k_facts, rerank_log = self.rerank_supports(query, query_fact_scores)
            # top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts_and_rerank(query, query_fact_scores)
            # # print(f"[dev] [TAG] [retrieve] query_fact_scores (len={len(query_fact_scores)}: \n{query_fact_scores}") # dev add
            # # print(f"[dev] [TAG] [retrieve] rerank_log['facts_before_rerank'] (len={len(rerank_log['facts_before_rerank'])}): \n{rerank_log['facts_before_rerank']}") # dev add
            # # print(f"[dev] [TAG] [retrieve] rerank_log['facts_after_rerank'] (len={len(rerank_log['facts_after_rerank'])}): \n{rerank_log['facts_after_rerank']}") # dev add
            
            # top_k_facts = []
            
            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            # if len(top_k_facts) == 0: # dev delete
            #     logger.info('No facts found after reranking, return DPR results')
            #     sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            # else:
            #     sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities_and_rerank(query=query,
            #                                                                              link_top_k=self.global_config.linking_top_k,
            #                                                                              query_fact_scores=query_fact_scores,
            #                                                                              top_k_facts=top_k_facts,
            #                                                                              top_k_fact_indices=top_k_fact_indices,
            #                                                                              passage_node_weight=self.global_config.passage_node_weight)
            
            if len(top_k_entities) == 0: # dev add
                logger.info('No entities found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_entity_scores=query_entity_scores,
                                                                                         top_k_entities=top_k_entities,
                                                                                         top_k_entity_indices=top_k_entity_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight)
                
            # print(f"[dev] [TAG] [topic_retrieve] sorted_doc_scores: {sorted_doc_scores}")
            
            # print(f"[dev] num_to_retrieve: \n{num_to_retrieve}") # 200
            
            # print(f"[dev] self.passage_node_keys: \n{self.passage_node_keys}")
            if isinstance(sorted_doc_ids[0], str):
                top_k_docs = [self.chunk_embedding_store.get_row(idx)["content"] for idx in sorted_doc_ids[:num_to_retrieve]]
            else:            
                top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results
        
    def retrieve_new(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the CogitoRAG pipeline, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            
            self.global_config.linking_top_k = 5 # dev add
            
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            
            print(f"[dev] [TAG] [retrieve] query_fact_scores (len={len(query_fact_scores)}: \n{query_fact_scores}") # dev add
            print(f"[dev] [TAG] [retrieve] rerank_log['facts_before_rerank'] (len={len(rerank_log['facts_before_rerank'])}): \n{rerank_log['facts_before_rerank']}") # dev add
            print(f"[dev] [TAG] [retrieve] rerank_log['facts_after_rerank'] (len={len(rerank_log['facts_after_rerank'])}): \n{rerank_log['facts_after_rerank']}") # dev add
            
            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(
                    query=query,
                    link_top_k=self.global_config.linking_top_k,
                    query_fact_scores=query_fact_scores,
                    top_k_facts=top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    passage_node_weight=self.global_config.passage_node_weight
                )

            rerank_k = min(200, len(sorted_doc_ids))
            candidate_local_ids = np.asarray(sorted_doc_ids[:rerank_k], dtype=int)
            ppr_scores_topk = np.asarray(sorted_doc_scores[:rerank_k], dtype=float)

            candidate_keys = { self.passage_node_keys[i] for i in candidate_local_ids }
            dense_all_ids, dense_all_scores = self.dense_passage_rerank(query, candidate_keys)

            dense_score_map = { int(doc_id): float(score) for doc_id, score in zip(dense_all_ids.tolist(),
                                                                                dense_all_scores.tolist()) }
            dense_scores_topk = np.array([dense_score_map.get(int(i), 0.0) for i in candidate_local_ids], dtype=float)

            gamma = float(getattr(self.global_config, "dense_fuse_gamma", 0.95))
            gamma = max(0.0, min(1.0, gamma))

            def _minmax(x):
                mn, mx = float(np.min(x)), float(np.max(x))
                if mx - mn < 1e-12:
                    return np.zeros_like(x)
                return (x - mn) / (mx - mn)

            dense_n = _minmax(dense_scores_topk)
            ppr_n   = _minmax(ppr_scores_topk)

            fused = gamma * ppr_n + (1.0 - gamma) * dense_n

            order_in_topk = np.argsort(fused)[::-1]
            final_local_ids_in_topk = candidate_local_ids[order_in_topk]
            final_scores_in_topk    = fused[order_in_topk]

            take = min(num_to_retrieve, len(final_local_ids_in_topk))
            final_local_ids = final_local_ids_in_topk[:take]
            final_scores    = final_scores_in_topk[:take]

            top_k_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                for idx in final_local_ids
            ]
            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=final_scores)
            )


        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results


    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the CogitoRAG pipeline, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            
            self.global_config.linking_top_k = 5 # dev add
            
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            
            print(f"[dev] [TAG] [retrieve] query_fact_scores (len={len(query_fact_scores)}: \n{query_fact_scores}") # dev add
            print(f"[dev] [TAG] [retrieve] rerank_log['facts_before_rerank'] (len={len(rerank_log['facts_before_rerank'])}): \n{rerank_log['facts_before_rerank']}") # dev add
            print(f"[dev] [TAG] [retrieve] rerank_log['facts_after_rerank'] (len={len(rerank_log['facts_after_rerank'])}): \n{rerank_log['facts_after_rerank']}") # dev add
            
            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def TAG_rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None,
               time_dir: str | None = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        
        TAG_data_dir = os.path.join(self.global_config.save_dir, "cognitive_graph_tag_data")
        if not os.path.exists(TAG_data_dir):
            os.makedirs(TAG_data_dir)
        settings_index_path = os.path.join(TAG_data_dir, "settings_query.yaml")
        settings_path = os.path.join(time_dir, "settings.yaml")
        shutil.copyfile(settings_index_path, settings_path)
        print(f"Copied {settings_index_path} to {settings_path}")
        
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            # print(f"[dev] [not isinstance(queries[0], QuerySolution)] queries[0]: \n{queries[0]}")
            
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)
                
            # print(f"[dev] [not isinstance(queries[0], QuerySolution)] queries[0]: \n{queries[0]}")

        # Performing QA
        # queries_solutions, all_response_message, all_metadata = self.qa(queries) # dev delete
        
        queries_solutions, all_response_message, all_metadata = self.TAG_qa(queries, time_dir) # dev add
        
        # print(f"[dev] [TAG] [TAG_rag_qa] queries_solutions: \n{queries_solutions}")
        # print(f"[dev] [TAG] [TAG_rag_qa] all_response_message: \n{all_response_message}")
        # print(f"[dev] [TAG] [TAG_rag_qa] all_metadata: \n{all_metadata}")

        # Evaluating QA
        if gold_answers is not None:
            print(f"[dev] gold_answers is not None:")
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            print(f"[dev] gold_answers is None:")
            
            return queries_solutions, all_response_message, all_metadata
        

    def split_queries_with_llm(self, queries: List[str], max_splits: int = 2, raw_log_path: str | None = None) -> List[List[str]]:
        """
        Use the current LLM to decide whether to split each query into up to `max_splits` sub-questions.
        Returns a list of sub-question lists aligned with the input queries.
        """
        split_results: List[List[str]] = []
        split_prompt_tokens = 0
        split_completion_tokens = 0
        split_cache_hits = 0
        split_start_time = time.time()

        # system_prompt = f"""
        #                 You are a query decomposition assistant for document retrieval.

        #                 Your task:
        #                 - Decide whether the question should be split into simpler sub-questions.
        #                 - If splitting is necessary, set "split" to true and provide up to {max_splits} short sub-questions.
        #                 - If no split is needed, set "split" to false and leave "sub_questions" as an empty list.

        #                 Rules:
        #                 - Output ONLY valid JSON.
        #                 - Do NOT include explanations, markdown, or extra text.
        #                 - Use exactly this format:

        #                 {{"split": true|false, "sub_questions": ["..."]}}
        #                 """


        system_prompt = f"""
                        You are a query decomposition assistant for document retrieval.

                        Your task:
                        - Decide whether the question should be split into simpler sub-questions.
                        - If splitting is necessary, set "split" to true and provide up to {max_splits} short sub-questions.
                        - If no split is needed, set "split" to false and leave "sub_questions" as an empty list.

                        Split the question ONLY when:
                        - answering requires retrieving information about two or more independent entities.
                        - the question involves comparison (e.g., earlier, later, same, different, both).
                        - the question can be naturally decomposed into parallel fact-finding queries.

                        Do NOT split when:
                        - the question asks about a single entity through a chain of relations.
                        - the question involves family or social relations.
                        - splitting would introduce ambiguous references (e.g., "the director").

                        Rules:
                        - Output ONLY valid JSON.
                        - Do NOT include explanations, markdown, or extra text.
                        - Use exactly this format:
                        {{"split": true|false, "sub_questions": ["..."]}}
                        """

        if raw_log_path is not None:
            os.makedirs(os.path.dirname(raw_log_path), exist_ok=True)
            with open(raw_log_path, "w", encoding="utf-8") as f:
                f.write("")

        for query in tqdm(queries, desc="Splitting queries"):
            # messages = [
            #     {"role": "system", "content": system_prompt},
            #     {"role": "user", "content": f"Question: {query}"}
            # ]
            messages  = [
                {"role": "system", "content": system_prompt},

                # ===== positive example =====
                {
                    "role": "user",
                    "content": (
                        "Question: Which film has the director born later, Arr\u00eate Ton Cin\u00e9ma or Agni (2004 Film)?"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        '{"split": true, "sub_questions": '
                        '["Who directed Arr\u00eate Ton Cin\u00e9ma and when was the director born?", '
                        '"Who directed Agni (2004 Film) and when was the director born?"]}'
                    )
                },

                # ===== negative example =====
                {
                    "role": "user",
                    "content": "Question: Who is the father of the director of film Power Of Women (Film)?"
                },
                {
                    "role": "assistant",
                    "content": '{"split": false, "sub_questions": []}'
                },

             # ===== real query =====
                {"role": "user", "content": f"Question: {query}"}
            ]
            
            try:
                llm_result = self.llm_model.infer(messages)
                response_content = llm_result[0] if isinstance(llm_result, tuple) else llm_result
                metadata = llm_result[1] if isinstance(llm_result, tuple) and len(llm_result) > 1 else {}
                cache_hit = llm_result[2] if isinstance(llm_result, tuple) and len(llm_result) > 2 else False
                split_prompt_tokens += metadata.get("prompt_tokens", 0)
                split_completion_tokens += metadata.get("completion_tokens", 0)
                if cache_hit:
                    split_cache_hits += 1
            except Exception as e:
                logger.warning(f"Query split failed, fallback to original. Error: {str(e)}")
                split_results.append([query])
                continue
            if raw_log_path is not None:
                try:
                    with open(raw_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"question": query, "raw_output": str(response_content)}, ensure_ascii=True) + "\n")
                except Exception:
                    pass

            sub_questions: List[str] = []
            try:
                match = re.search(r"\{.*\}", str(response_content), flags=re.DOTALL)
                payload = json.loads(match.group(0)) if match else {}
                if payload.get("split") is True:
                    sub_questions = [str(q).strip() for q in payload.get("sub_questions", []) if str(q).strip()]
            except Exception:
                sub_questions = []

            if not sub_questions:
                split_results.append([query])
            else:
                split_results.append(sub_questions[:max_splits])

        split_time = time.time() - split_start_time
        self.last_split_stats = {
            "time_s": split_time,
            "prompt_tokens": split_prompt_tokens,
            "completion_tokens": split_completion_tokens,
            "cache_hits": split_cache_hits,
        }

        return split_results

    def merge_docs_topk_plus_one(self, sub_solutions: List[QuerySolution], per_sub_top_k: int, extra_global: int = 1) -> Tuple[List[str], List[float]]:
        """
        Merge documents by taking top-k from each sub-question, then add extra_global highest-scoring docs from the remaining pool.
        """
        merged_docs: List[str] = []
        merged_scores: List[float] = []
        seen: set[str] = set()

        # take top-k per sub-question
        for sol in sub_solutions:
            if sol is None or not sol.docs:
                continue
            docs = sol.docs
            scores = list(sol.doc_scores) if sol.doc_scores is not None else [0.0] * len(docs)
            for doc, score in list(zip(docs, scores))[:per_sub_top_k]:
                if doc in seen:
                    continue
                seen.add(doc)
                merged_docs.append(doc)
                merged_scores.append(float(score))

        # collect remaining candidates
        candidates: List[Tuple[str, float]] = []
        for sol in sub_solutions:
            if sol is None or not sol.docs:
                continue
            docs = sol.docs
            scores = list(sol.doc_scores) if sol.doc_scores is not None else [0.0] * len(docs)
            for doc, score in zip(docs, scores):
                if doc in seen:
                    continue
                candidates.append((doc, float(score)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        for doc, score in candidates[:extra_global]:
            seen.add(doc)
            merged_docs.append(doc)
            merged_scores.append(score)

        return merged_docs, merged_scores

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None,
               enable_split: bool = True) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the CogitoRAG pipeline.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """

        print("========================================================")
        retrieval_stage_start = time.time()
        self._rerank_token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "cache_hits": 0}
        self.last_split_stats = {}
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None
        
        topic_method = False # dev add

        split_meta: List[Dict[str, Any]] = []
        original_questions: List[str] | None = None

        if not isinstance(queries[0], QuerySolution):
            # print(f"[dev] [not isinstance(queries[0], QuerySolution)] queries[0]: \n{queries[0]}")

            original_questions = list(queries)
            raw_log_path = None
            if enable_split:
                split_groups = self.split_queries_with_llm(queries, max_splits=2, raw_log_path=raw_log_path)
            else:
                split_groups = [[q] for q in queries]
            # Disable writing split results to disk.
            flattened_queries: List[str] = []
            group_sizes: List[int] = []
            group_is_split: List[bool] = []

            for original_query, sub_queries in zip(queries, split_groups):
                cleaned = [q.strip() for q in (sub_queries or []) if q and str(q).strip()]
                if not cleaned:
                    cleaned = [original_query]
                is_split = len(cleaned) > 1 or (len(cleaned) == 1 and cleaned[0] != original_query)
                group_is_split.append(is_split)
                group_sizes.append(len(cleaned))
                flattened_queries.extend(cleaned)
                split_meta.append(
                    {
                        "split": bool(is_split),
                        "sub_questions": cleaned[:2] if is_split else []
                    }
                )

            # Retrieve all sub-queries in one batch
            if topic_method:
                sub_solutions = self.topic_retrieve(queries=flattened_queries)
            else:
                sub_solutions = self.retrieve(queries=flattened_queries)

            # Merge back to original queries
            merged_queries: List[QuerySolution] = []
            cursor = 0
            for original_query, size, is_split in zip(queries, group_sizes, group_is_split):
                sub_slice = sub_solutions[cursor:cursor + size]
                cursor += size
                if not is_split:
                    qsol = sub_slice[0]
                    qsol.question = original_query
                    merged_queries.append(qsol)
                else:
                    merged_docs, merged_scores = self.merge_docs_topk_plus_one(sub_slice, per_sub_top_k=2, extra_global=1)
                    merged_queries.append(
                        QuerySolution(
                            question=original_query,
                            docs=merged_docs,
                            doc_scores=np.array(merged_scores) if merged_scores is not None else None
                        )
                    )

            queries = merged_queries

            # Evaluate retrieval on merged docs if gold_docs provided
            if gold_docs is not None:
                retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)
                k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
                overall_retrieval_result, _example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                    gold_docs=gold_docs,
                    retrieved_docs=[retrieval_result.docs for retrieval_result in queries],
                    k_list=k_list
                )
                logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")
        else:
            original_questions = [q.question for q in queries]
            split_meta = [{"split": False, "sub_questions": []} for _ in queries]
            
            # print(f"[dev] [not isinstance(queries[0], QuerySolution)] queries[0].question: \n{queries[0].question}")
            # print(f"[dev] [not isinstance(queries[0], QuerySolution)] queries[0].answer: \n{queries[0].answer}")

        retrieval_stage_time = time.time() - retrieval_stage_start
        split_stats = self.last_split_stats or {"prompt_tokens": 0, "completion_tokens": 0, "cache_hits": 0, "time_s": 0.0}
        rerank_stats = self._rerank_token_stats or {"prompt_tokens": 0, "completion_tokens": 0, "cache_hits": 0}
        self.last_retrieval_stats = {
            "time_s": retrieval_stage_time,
            "prompt_tokens": split_stats.get("prompt_tokens", 0) + rerank_stats.get("prompt_tokens", 0),
            "completion_tokens": split_stats.get("completion_tokens", 0) + rerank_stats.get("completion_tokens", 0),
            "split": split_stats,
            "rerank": rerank_stats,
        }

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)
        if original_questions is not None:
            os.makedirs(self.global_config.save_dir, exist_ok=True)
            records: List[Dict[str, Any]] = []
            for idx, (q, sol, meta) in enumerate(zip(original_questions, queries_solutions, split_meta), start=1):
                record: Dict[str, Any] = {
                    "params": {
                        "enable_split": bool(enable_split),
                        "total_questions": len(original_questions),
                    },
                    "question": q,
                    "split": bool(meta.get("split", False)),
                    "sub_questions": meta.get("sub_questions", []),
                    "output": sol.answer,
                    "golden_answer": None,
                }
                if gold_answers is not None and (idx - 1) < len(gold_answers):
                    ga = gold_answers[idx - 1]
                    if isinstance(ga, set):
                        ga = list(ga)
                    record["golden_answer"] = ga
                records.append(record)

            # Write a single result file per run: resultN.json
            existing = []
            for name in os.listdir(self.global_config.save_dir):
                if name.startswith("result") and name.endswith(".json") and "_with_meta" not in name:
                    num = name[len("result"):-len(".json")]
                    if num.isdigit():
                        existing.append(int(num))
            next_idx = max(existing) + 1 if existing else 1
            out_path = os.path.join(self.global_config.save_dir, f"result{next_idx}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=True, indent=2)

            self.last_result_json_path = out_path

            run_metadata = copy.deepcopy(getattr(self, "last_run_metadata", {}) or {})
            run_metadata.setdefault("qa_prompt_name_requested", self.last_qa_prompt_name_requested)
            run_metadata.setdefault("qa_prompt_name_resolved", self.last_qa_prompt_name)
            run_metadata.setdefault("qa_prompt_path", self.last_qa_prompt_path)
            run_metadata.setdefault("graph_pickle_path", getattr(self, "_graph_pickle_filename", None))
            run_metadata.setdefault("save_dir", self.global_config.save_dir)
            run_metadata.setdefault("graph_working_dir", getattr(self.global_config, "graph_working_dir", None))
            run_metadata.setdefault("memory_map_path", getattr(self.global_config, "memory_map_path", None))
            run_metadata.setdefault("qa_use_memory", os.environ.get("QA_USE_MEMORY", "").lower() in {"1", "true", "yes"})
            run_metadata.setdefault("llm_name", self.global_config.llm_name)
            run_metadata.setdefault("embedding_name", self.global_config.embedding_model_name)
            run_metadata.setdefault("qa_top_k", self.global_config.qa_top_k)
            run_metadata.setdefault("retrieval_top_k", self.global_config.retrieval_top_k)
            run_metadata.setdefault("max_qa_steps", self.global_config.max_qa_steps)
            run_metadata.setdefault("topic_prompt_name", None)
            run_metadata.setdefault("ner_prompt_name", None)
            run_metadata.setdefault("triple_prompt_name", None)
            run_metadata.setdefault("enable_split", None)
            run_metadata.setdefault("dataset", getattr(self.global_config, "dataset", None))

            if "args" not in run_metadata:
                run_metadata["args"] = {}

            if "command" not in run_metadata:
                safe_argv = []
                skip_next = False
                for token in sys.argv:
                    if skip_next:
                        skip_next = False
                        continue
                    if token in {"--api_key", "--topic_api_key"}:
                        safe_argv.extend([token, "***"])
                        skip_next = True
                        continue
                    if token.startswith("--api_key=") or token.startswith("--topic_api_key="):
                        safe_argv.append(token.split("=", 1)[0] + "=***")
                    else:
                        safe_argv.append(token)
                run_metadata["command"] = " ".join(safe_argv)

            run_metadata["stage_stats"] = {
                "index": copy.deepcopy(getattr(self, "last_index_stats", {}) or {}),
                "retrieval": copy.deepcopy(getattr(self, "last_retrieval_stats", {}) or {}),
                "qa": copy.deepcopy(getattr(self, "last_qa_stats", {}) or {}),
            }

            output_files = run_metadata.setdefault("output_files", {})
            output_files["result_json"] = out_path

            out_with_meta_path = os.path.join(self.global_config.save_dir, f"result{next_idx}_with_meta.json")
            output_files["result_with_meta_json"] = out_with_meta_path
            with open(out_with_meta_path, "w", encoding="utf-8") as f:
                json.dump({"run_metadata": run_metadata, "records": records}, f, ensure_ascii=False, indent=2)

            self.last_result_with_meta_json_path = out_with_meta_path
            self.last_run_metadata = run_metadata
        
        # print(f"[dev] [TAG] [TAG_rag_qa] len(queries): \n{len(queries)}")
        # print(f"[dev] [TAG] [TAG_rag_qa] queries: \n{queries}")
        # print(f"[dev] [TAG] [TAG_rag_qa] queries_solutions: \n{queries_solutions}")
        # print(f"[dev] [TAG] [TAG_rag_qa] all_response_message: \n{all_response_message}")
        # print(f"[dev] [TAG] [TAG_rag_qa] all_metadata: \n{all_metadata}")

        # Evaluating QA
        if gold_answers is not None:
            print(f"[dev] gold_answers is not None:")
            import re

            def _is_valid_answer(ans: str) -> bool:
                if ans is None:
                    return False
                s = str(ans).strip()
                if not s:
                    return False
                low = s.lower()
                bad_keys = [
                    "i'm sorry", "i am sorry", "cannot help", "can't help",
                    "i cannot assist", "i can't assist", "as an ai",
                    "policy", "违反政策", "不符合政策", "抱歉", "拒绝回答", "无法提供"
                ]
                return not any(k in low for k in bad_keys)

            _filtered_pairs = []
            drop_cnt = 0
            for q, g in zip(queries_solutions, gold_answers):
                ans = getattr(q, "answer", None) if q is not None else None
                if _is_valid_answer(ans):
                    _filtered_pairs.append((ans, g))
                else:
                    drop_cnt += 1

            if len(_filtered_pairs) == 0:
                logger.warning("[dev] All QA examples filtered out by policy/refusal check; skip EM/F1.")
                overall_qa_results = {"ExactMatch": 0.0, "F1": 0.0}
                for idx, q in enumerate(queries_solutions):
                    if q is None: 
                        continue
                    try:
                        q.gold_answers = list(gold_answers[idx])
                        if gold_docs is not None:
                            q.gold_docs = gold_docs[idx]
                    except Exception:
                        pass
                return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results

            logger.info(f"[dev] QA eval filtered {drop_cnt} invalid/policy samples.")

            predicted_answers = [a for a, _g in _filtered_pairs]
            filtered_gold_answers = [_g for _a, _g in _filtered_pairs]


            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=filtered_gold_answers,
                predicted_answers=predicted_answers,
                aggregation_fn=np.max
            )
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=filtered_gold_answers,
                predicted_answers=predicted_answers,
                aggregation_fn=np.max
            )




            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            print(f"[dev] gold_answers is None:")
            
            return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        #Running inference for QA
        qa_start_time = time.time()
        all_qa_messages = []

        use_memory = os.environ.get("QA_USE_MEMORY", "").lower() in {"1", "true", "yes"}
        passage_to_memory = getattr(self, "passage_to_memory", None) if use_memory else None

        dataset_name = self.global_config.dataset
        dataset_v4_1 = f'rag_qa_{dataset_name}_v4_1'
        dataset_default = f'rag_qa_{dataset_name}'
        override_prompt = getattr(self.global_config, "qa_prompt_name", None)
        self.last_qa_prompt_name_requested = override_prompt

        if override_prompt and self.prompt_template_manager.is_template_name_valid(name=override_prompt):
            prompt_name = override_prompt
        elif self.prompt_template_manager.is_template_name_valid(name=dataset_v4_1):
            prompt_name = dataset_v4_1
        elif self.prompt_template_manager.is_template_name_valid(name=dataset_default):
            # find the corresponding prompt for this dataset
            prompt_name = dataset_default
        elif self.prompt_template_manager.is_template_name_valid(name='rag_qa_musique_v4_1'):
            prompt_name = 'rag_qa_musique_v4_1'
        else:
            # the dataset does not have a customized prompt template yet
            logger.debug(
                f"rag_qa_{dataset_name} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
            prompt_name = 'rag_qa_musique'

        self.last_qa_prompt_name = prompt_name
        try:
            module = None
            for module_name in (
                f"src.cogitorag.prompts.templates.{prompt_name}",
                f"src.legacy.prompts.templates.{prompt_name}",
                f"cogitorag.prompts.templates.{prompt_name}",
                f"legacy.prompts.templates.{prompt_name}",
            ):
                try:
                    module = importlib.import_module(module_name)
                    break
                except Exception:
                    continue
            self.last_qa_prompt_path = getattr(module, "__file__", None)
        except Exception:
            self.last_qa_prompt_path = None

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):
            try:
                # obtain the retrieved docs
                retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

                prompt_user = ''
                for i, passage in enumerate(retrieved_passages):
                    if passage_to_memory is not None:
                        prompt_user += f'[Document {i+1} - Original]\n'
                    prompt_user += f'Wikipedia Title: {passage}\n\n'

                    if passage_to_memory is not None:
                        memory = passage_to_memory.get(passage, None)
                        if memory and memory != passage:
                            prompt_user += f'[Document {i+1} - Summary]\n'
                            prompt_user += f'{memory}\n\n'
                prompt_user += 'Question: ' + query_solution.question + '\nThought: '

                all_qa_messages.append(
                    self.prompt_template_manager.render(name=prompt_name, prompt_user=prompt_user))
            except Exception as e: # dev add
                print(f"[dev] Error: {e}")
                print(f"[dev] query_solution: {query_solution}")

        # all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")] # dev delete
        
        all_qa_results = [] # dev add
        for qa_messages in tqdm(all_qa_messages, desc="QA Reading"):
            """
            qa_result = self.llm_model.infer(qa_messages)
            all_qa_results.append(qa_result)
            """
            try:
                qa_result = self.llm_model.infer(qa_messages)
                all_qa_results.append(qa_result)
            except Exception as e:  # dev add
                print(f"[dev] Error: {e}")
                print(f"[dev] qa_messages: {qa_messages}")
                all_qa_results.append(("", {}, False))

        
        # dev add: test print
        qa_message = all_qa_messages[0]
        qa_result = all_qa_results[0]
        print(f"[dev] [TAG] [qa] qa_message: \n{qa_message}")
        print(f"[dev] [TAG] [qa] qa_result: \n{qa_result}")
        
        temp = None
        for i in range(len(all_qa_results)):
            if all_qa_results[i] != None:
                temp = all_qa_results[i]
                break
        if temp == None:
            print("[dev] Maybe wrong in llm generating!")
        for i in range(len(all_qa_results)):
            if all_qa_results[i] == None:
                all_qa_results[i] = temp
                
        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans, _matched = extract_final_answer(response_content)
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        qa_time = time.time() - qa_start_time
        qa_prompt_tokens = sum(m.get("prompt_tokens", 0) for m in all_metadata if isinstance(m, dict))
        qa_completion_tokens = sum(m.get("completion_tokens", 0) for m in all_metadata if isinstance(m, dict))
        qa_cache_hits = sum(1 for hit in all_cache_hit if hit)
        self.last_qa_stats = {
            "time_s": qa_time,
            "prompt_tokens": qa_prompt_tokens,
            "completion_tokens": qa_completion_tokens,
            "cache_hits": qa_cache_hits,
            "prompt_name_requested": self.last_qa_prompt_name_requested,
            "prompt_name_resolved": self.last_qa_prompt_name,
            "prompt_path": self.last_qa_prompt_path,
        }

        return queries_solutions, all_response_message, all_metadata
    
    def TAG_qa(self, queries: List[QuerySolution],
               time_dir: str | None = None) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        
        import graphrag.api as api
        print(f"[dev] import graphrag.api as api")
        
        from graphrag.cli.main import get_search_engine
        print(f"[dev] from graphrag.cli.main import get_search_engine")
        
        # Legacy note: older experiments referenced a private settings.yaml path here.
        from pathlib import Path
        root = Path(time_dir)
        local_search_engine = get_search_engine(method="local", query="None", root=root)
        
        # /data2-HDD-SATA-20T/nzq/jmf/new_rag_2/graphrag_test/graphrag/graphrag/api/query.py
        
        search_prompt = local_search_engine.get_system_prompt(query=queries[0].question, rerank=True)
        
        print(f"========== [dev] ==========")
        print(f"[dev] graphrag.query.structured_search.local_search.search.py:search()")
        print(f"search_prompt: \n{search_prompt}")
        print(f"========== [dev] ==========")
        

        print(f"[dev] local_search_engine: {local_search_engine}")
        
        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):
            try:
                # obtain the retrieved docs
                retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]
                
                print(f"[dev] retrieved_passages: \n{retrieved_passages}")

                # prompt_user = ''
                # for passage in retrieved_passages:
                #     prompt_user += f'Wikipedia Title: {passage}\n\n'
                # prompt_user += 'Question: ' + query_solution.question + '\nThought: '
                prompt_user = local_search_engine.get_system_prompt(query=query_solution.question, rerank=True)

                if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                    # find the corresponding prompt for this dataset
                    prompt_dataset_name = self.global_config.dataset
                else:
                    # the dataset does not have a customized prompt template yet
                    logger.debug(
                        f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                    prompt_dataset_name = 'musique'
                # all_qa_messages.append( # dev delete
                #     self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))
                all_qa_messages.append( # dev add
                    self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))
            except Exception as e: # dev add
                print(f"[dev] Error: {e}")
                print(f"[dev] query_solution: {query_solution}")

        # all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")] # dev delete
        
        print(f"[dev] [TAG] [TAG_qa] all_qa_messages[0]: \n{all_qa_messages[0]}")
        
        all_qa_results = [] # dev add
        for qa_messages in tqdm(all_qa_messages, desc="QA Reading"):
            try:
                qa_result = self.llm_model.infer(qa_messages)
                import inspect
                print(f"inspect.getmodule(self.llm_model.infer): {inspect.getmodule(self.llm_model.infer)}")
                
                print(f"[dev] qa_result: \n{qa_result}")
                
                all_qa_results.append(qa_result)
            except Exception as e: # dev add
                print(f"[dev] Error: {e}")
                print(f"[dev] qa_messages: {qa_messages}")
                all_qa_results.append(all_qa_results[-1])
        
        print(f"[dev] [TAG] [TAG_qa] all_qa_results[0]: \n{all_qa_results[0]}")
        
        # dev add: test print
        qa_message = all_qa_messages[0]
        qa_result = all_qa_results[0]

        
        print(f"[dev] [TAG] [qa] qa_message: \n{qa_message}")
        print(f"[dev] [TAG] [qa] qa_result: \n{qa_result}")
        
        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput],
                             topic_results_dict: Dict[str, TopicRawOutput] = None) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}

            if topic_results_dict and chunk_key in topic_results_dict:
                topic_result = topic_results_dict[chunk_key]
                chunk_openie_info['think'] = topic_result.think
                chunk_openie_info['memory'] = topic_result.memory
                chunk_openie_info['metadata'] = topic_result.metadata

            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Additionally, saves think content to separate files in the think_storage directory.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities, think, and memory.
        """

        def _safe_len(value) -> int:
            if value is None:
                return 0
            return len(str(value))

        def _safe_word_len(value) -> int:
            if value is None:
                return 0
            return len(str(value).split())

        sum_phrase_chars = sum([
            _safe_len(e) for chunk in all_openie_info for e in chunk.get('extracted_entities', [])
        ])
        sum_phrase_words = sum([
            _safe_word_len(e) for chunk in all_openie_info for e in chunk.get('extracted_entities', [])
        ])
        num_phrases = sum([len(chunk.get('extracted_entities', [])) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0

            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }

            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

            think_count = 0
            for chunk in all_openie_info:
                if 'think' in chunk and chunk['think']:
                    chunk_id = chunk['idx']
                    think_content = chunk['think']

                    think_file_path = os.path.join(self.think_storage_dir, f'{chunk_id}.txt')

                    with open(think_file_path, 'w', encoding='utf-8') as f:
                        f.write(think_content)

                    think_count += 1

            if think_count > 0:
                logger.info(f"Saved {think_count} think files to {self.think_storage_dir}")

            failed_count = 0
            failed_records = []

            for chunk in all_openie_info:
                if 'metadata' in chunk and chunk.get('metadata', {}).get('extraction_failed', False):
                    chunk_id = chunk['idx']
                    passage = chunk['passage']
                    failure_reason = chunk.get('metadata', {}).get('failure_reason', 'unknown')
                    retry_count = chunk.get('metadata', {}).get('retry_count', 0)

                    failed_record = {
                        'chunk_id': chunk_id,
                        'passage': passage,
                        'failure_reason': failure_reason,
                        'retry_count': retry_count,
                        'timestamp': str(datetime.now())
                    }
                    failed_records.append(failed_record)
                    failed_count += 1

            if failed_count > 0:
                failed_file_path = os.path.join(self.failed_extraction_dir,
                                               f'failed_extractions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

                with open(failed_file_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'total_failed': failed_count,
                        'failed_records': failed_records
                    }, f, indent=2, ensure_ascii=False)

                logger.warning(f"Saved {failed_count} failed extraction records to {failed_file_path}")
                logger.warning(f"These texts need manual review and processing.")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            # If the graph is empty but we have nodes, we need to add them
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # Check if all entity and passage nodes are in the graph
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                # If nodes are missing, rebuild the graph
                self.add_new_nodes()
                self.save_igraph()
                # Update the mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # Check if the lengths match
            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")
                
                # If there are missing keys, create empty entries for them
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )

            # prepare data_store
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])
    
    def get_entity_scores(self, query: str) -> np.ndarray: # dev add
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        # Check if there are any facts
        if len(self.entity_embeddings) == 0:
            logger.warning("No entities available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_entity_scores = np.dot(self.entity_embeddings, query_embedding.T) # shape: (#facts, )
            query_entity_scores = np.squeeze(query_entity_scores) if query_entity_scores.ndim == 2 else query_entity_scores
            query_entity_scores = min_max_normalize(query_entity_scores)
            return query_entity_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def dense_passage_rerank(self, query: str, chunk_ids_related) -> Tuple[np.ndarray, np.ndarray]: # dev add
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        for query_doc_scores_index in range(len(query_doc_scores)):
            if self.passage_node_keys[query_doc_scores_index] not in chunk_ids_related:
                query_doc_scores[query_doc_scores_index] = 0

        # print(f"[dev] query_doc_scores: {query_doc_scores}")

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        
        # print(f"[dev] sorted_doc_ids: \n{sorted_doc_ids}") # int
        
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """
        
        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))

        dataset_name = (self.global_config.dataset or "").lower()
        alpha_beta_map = {
            "nq": (1.0, 0.5),
            "popqa": (6.0, 6.0),
            "musique": (4.0, 3.0),
            "2wikimultihopqa": (2.0, 1.2),
            "2wiki": (2.0, 1.2),
            "hotpotqa": (2.0, 1.8),
        }
        default_alpha, default_beta = alpha_beta_map.get(dataset_name, (4.0, 3.0))
        config_alpha = getattr(self.global_config, "entity_alpha", None)
        config_beta = getattr(self.global_config, "entity_beta", None)
        alpha = default_alpha if config_alpha is None else float(config_alpha)
        beta = default_beta if config_beta is None else float(config_beta)
        # ------------------------

        from collections import defaultdict
        entity_hit_count = defaultdict(int)

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                entity_hit_count[phrase] += 1

                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    bonus = 1.0 + alpha * (1 - np.exp(-beta * entity_hit_count[phrase]))
                    weight = fact_score * bonus
                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weight /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] = weight

                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)


        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        #Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        #Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def run_ppr_new(
        self,
        node_personalization: np.ndarray,
        damping: float = 0.5,
        edge_weight_attr: str = "weight",
        **kwargs,
    ):
        """
        实体子图 PPR → 段落聚合采用：取段落内实体PPR前K之和 / sqrt(实体数) → Min-Max归一化
        返回：(sorted_ids, sorted_scores_in_[0,1])
        - 忽略段落位置的个性化权重（即使有人误传了非零值）
        - 仅使用实体节点的个性化权重作为 reset 分布
        """
        import numpy as np

        if not hasattr(self, "entity_node_idxs") or not hasattr(self, "passage_node_idxs"):
            raise RuntimeError("缺少 entity_node_idxs 或 passage_node_idxs。")

        entity_indices  = self.entity_node_idxs
        passage_indices = self.passage_node_idxs
        if len(entity_indices) == 0:
            raise RuntimeError("entity_node_idxs 为空，无法执行实体子图 PPR。")

        K = int(getattr(self.global_config, "ppr_topk", 8))




        node_names = self.graph.vs["name"]

        subgraph = self.graph.induced_subgraph(entity_indices)
        full_to_sub = {full_i: sub_i for sub_i, full_i in enumerate(entity_indices)}

        reset_sub = np.array([float(node_personalization[full_i]) for full_i in entity_indices], dtype=float)
        s = reset_sub.sum()
        if s <= 0:
            reset_sub = np.ones(len(entity_indices), dtype=float) / len(entity_indices)
        else:
            reset_sub = reset_sub / s

        pr_sub = subgraph.personalized_pagerank(
            vertices=range(len(entity_indices)),
            damping=damping,
            directed=False,
            weights=edge_weight_attr,
            reset=reset_sub,
            implementation="prpack",
        )

        if not hasattr(self, "chunk_id_to_entity_full_indices"):
            if not hasattr(self, "ent_node_to_chunk_ids"):
                raise RuntimeError("缺少 ent_node_to_chunk_ids 或 chunk_id_to_entity_full_indices 映射。")
            chunk_to_ent_full = {}
            for ent_full_i in entity_indices:
                ent_key = node_names[ent_full_i]
                for chunk_id in self.ent_node_to_chunk_ids.get(ent_key, set()):
                    chunk_to_ent_full.setdefault(chunk_id, []).append(ent_full_i)
            self.chunk_id_to_entity_full_indices = chunk_to_ent_full

        doc_scores = np.zeros(len(passage_indices), dtype=float)

        for idx_in_list, pass_full_i in enumerate(passage_indices):
            chunk_key = node_names[pass_full_i]
            ent_full_list = self.chunk_id_to_entity_full_indices.get(chunk_key, [])

            if not ent_full_list:
                doc_scores[idx_in_list] = 0.0
                continue

            ent_full_set = set(ent_full_list)

            scores_e = []
            for ent_full_i in ent_full_set:
                sub_i = full_to_sub.get(ent_full_i)
                if sub_i is not None:
                    scores_e.append(pr_sub[sub_i])

            m = len(scores_e)
            if m == 0:
                doc_scores[idx_in_list] = 0.0
                continue

            se = np.array(scores_e, dtype=float)
            k = min(K, m)
            topk_vals = np.partition(se, -k)[-k:]
            base = float(np.sum(topk_vals))

            doc_scores[idx_in_list] = base / (m ** 0.5)

        mn, mx = float(np.min(doc_scores)), float(np.max(doc_scores))
        if mx - mn < 1e-12:
            doc_scores_norm = np.zeros_like(doc_scores)
        else:
            doc_scores_norm = (doc_scores - mn) / (mx - mn)

        sorted_ids = np.argsort(doc_scores_norm)[::-1]
        return sorted_ids, doc_scores_norm[sorted_ids]




    def graph_search_with_fact_entities_and_rerank(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """
        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))

        dataset_name = (self.global_config.dataset or "").lower()
        alpha_beta_map = {
            "nq": (1.0, 0.5),
            "popqa": (6.0, 6.0),
            "musique": (4.0, 3.0),
            "2wikimultihopqa": (2.0, 1.2),
            "2wiki": (2.0, 1.2),
            "hotpotqa": (2.0, 1.8),
        }
        default_alpha, default_beta = alpha_beta_map.get(dataset_name, (4.0, 3.0))
        config_alpha = getattr(self.global_config, "entity_alpha", None)
        config_beta = getattr(self.global_config, "entity_beta", None)
        alpha = default_alpha if config_alpha is None else float(config_alpha)
        beta = default_beta if config_beta is None else float(config_beta)
        # ------------------------

        entity_hit_count = defaultdict(int)

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                entity_hit_count[phrase] += 1

                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    bonus = 1.0 + alpha * (1 - np.exp(-beta * entity_hit_count[phrase]))
                    weight = fact_score * bonus
                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weight /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] = weight

                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)


        

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        #Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        #Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        chunk_int_ids_related = ppr_sorted_doc_ids[:200]
        chunk_ids_related = [self.passage_node_keys[chunk_int_id] for chunk_int_id in chunk_int_ids_related]
        
        # print(f"[dev] chunk_ids_related: \n{chunk_ids_related}")

        chunk_ids_related_rerank, chunk_ids_related_rerank_scores = self.dense_passage_rerank(query, chunk_ids_related)
        
        return chunk_ids_related_rerank, chunk_ids_related_rerank_scores

    def graph_search_with_entities(self, query: str,
                                        link_top_k: int,
                                        query_entity_scores: np.ndarray,
                                        # top_k_entities: List[Tuple],
                                        top_k_entities: List[str],
                                        top_k_entity_indices: List[str],
                                        passage_node_weight: float = 0.05,
                                        use_ppr: bool = False
                                        ) -> Tuple[np.ndarray, np.ndarray]:
        # print(f"[dev] [TAG] [graph_search_with_entities]")
        
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """
        
        chunk_ids_related = []
        
        multi_hop = 7

        # entities_related = [] # "entity-"
        # for rank, e in enumerate(top_k_entities):
        #     entitiy_phrase = e.lower()
        #     phrase = entitiy_phrase
        #     phrase_key = compute_mdhash_id(
        #         content=phrase,
        #         prefix="entity-"
        #     )
        
        # entities_related = list(set(entities_related))
        
        entities_related = set()
        triples_related = set()
        visited = set()
        
        # self.entity_to_triple_list

        for e in top_k_entities:
            entity = e.lower()
            entity_key = compute_mdhash_id(
                content=entity,
                prefix="entity-"
            )
            visited.add(entity_key)
            frontier = {entity_key}

            for _ in range(multi_hop):
                next_frontier = set()
                for ent in frontier:
                    for neighbor in self.entity_to_entity_list.get(ent, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
                frontier = next_frontier

            entities_related.update(visited)
        
        entities_related = list(entities_related)
        
        entities_related = [self.entity_id_to_row[entity_key]["content"] for entity_key in entities_related] # NOTE: entity_key → entity_name
        
        for entity in entities_related:
            triple_list = set(self.entity_to_triple_list[entity])
            triples_related.update(triple_list)
        
        triples_related = list(triples_related)
        facts_related = [str(triple) for triple in triples_related]
        # facts_key_related = [compute_mdhash_id(content=fact, prefix='fact-') for fact in facts_related]
        # facts_embedding_related = self.fact_embedding_store.get_embeddings(hash_ids=facts_key_related)
        # print(f"[dev] len(facts_key_related): {len(facts_key_related)}")
        # print(f"[dev] len(facts_embedding_related): {len(facts_embedding_related)}")
        
        query_fact_scores = self.get_fact_scores(query)
        top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts_and_rerank(query, query_fact_scores)

        # # print(f"[dev] top_k_facts[:10]: \n{top_k_facts[:10]}")
        # print(f"[dev] query: \n{query}")
        # print(f"[dev] [Before Rerank] facts_related: \n{facts_related}")

        facts_related_rank = []
        see_num = 1000
        for fact in facts_related:
            if fact in top_k_facts[:see_num]:
                facts_related_rank.append(top_k_facts.index(fact))
            else:
                facts_related_rank.append(len(top_k_facts))
        # print(f"[dev] facts_related_rank: \n{facts_related_rank}")
        
        facts_related = [fact for _, fact in sorted(zip(facts_related_rank, facts_related), key=lambda x: x[0])]
        
        # print(f"[dev] [After Rerank] facts_related: \n{facts_related}")
        
        fact_number = 10
        facts_related = facts_related[:fact_number]
        triples_related = [eval(fact) for fact in facts_related]
        
        entities_related = []
        for rank, f in enumerate(triples_related):
            ent1 = f[0].lower()
            relation = f[1].lower()
            ent2 = f[2].lower()
            entities_related.append(ent1)
            entities_related.append(ent2)
        
        entities_related = list(set(entities_related))
        
        # print(f"[dev] entities_related: \n{entities_related}")
        
        # # print(f"[dev] top_k_entities: \n{top_k_entities}") # dev add
        # print(f"==========") # dev add
        # # print(f"[dev] top_k_entities: \n{[compute_mdhash_id(content=entity, prefix='entity-') for entity in top_k_entities]}") # dev add
        # print(f"[dev] entities_related: \n{entities_related}") # dev add
        # print(f"[dev] triples_related: \n{triples_related}") # dev add
        # print(f"==========") # dev add
        
        
        
        
        """
        for rank, e in enumerate(top_k_entities):
            entitiy_phrase = e.lower()
            # entity_score = query_entity_scores[
            #     top_k_entity_indices[rank]] if query_entity_scores.ndim > 0 else query_entity_scores
            phrase = entitiy_phrase
            phrase_key = compute_mdhash_id(
                content=phrase,
                prefix="entity-"
            )
            
            phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
            
            chunk_ids_str: set = self.ent_node_to_chunk_ids[phrase_key]
            
            chunk_ids_related.extend(list(chunk_ids_str))
        
        chunk_ids_related = list(set(chunk_ids_related))
        
        print(f"[dev] [Before Multi-Hop] chunk_ids_related (len={len(chunk_ids_related)}): {chunk_ids_related}")
        """

        # print(f"[dev] entities_related: \n{entities_related}")
        for rank, e in enumerate(entities_related):
            entitiy_phrase = e.lower()
            phrase = entitiy_phrase
            phrase_key = compute_mdhash_id(
                content=phrase,
                prefix="entity-"
            )
            phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
            
            chunk_ids_str: set = self.ent_node_to_chunk_ids[phrase_key]
            
            chunk_ids_related.extend(list(chunk_ids_str))
        
        chunk_ids_related = list(set(chunk_ids_related))
        
        # print(f"[dev] [After Multi-Hop] chunk_ids_related (len={len(chunk_ids_related)}): {chunk_ids_related}")
        
        # chunk_ids_related_rerank, chunk_ids_related_rerank_scores = self.dense_passage_retrieval(query)
        chunk_ids_related_rerank, chunk_ids_related_rerank_scores = self.dense_passage_rerank(query, chunk_ids_related)
        dense_ids, dense_scores = chunk_ids_related_rerank, chunk_ids_related_rerank_scores
        
        # return chunk_ids_related_rerank, chunk_ids_related_rerank_scores
        

        if use_ppr:

            #Assigning phrase weights based on selected facts from previous steps.
            linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
            phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
            phrase_weights = np.zeros(len(self.graph.vs['name']))
            passage_weights = np.zeros(len(self.graph.vs['name']))
            
            for rank, e in enumerate(top_k_entities):
                # subject_phrase = f[0].lower()
                # predicate_phrase = f[1].lower()
                # object_phrase = f[2].lower()
                entitiy_phrase = e.lower()
                entity_score = query_entity_scores[
                    top_k_entity_indices[rank]] if query_entity_scores.ndim > 0 else query_entity_scores
                for phrase in [entitiy_phrase]:
                    phrase_key = compute_mdhash_id(
                        content=phrase,
                        prefix="entity-"
                    )
                    phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                    if phrase_id is not None:
                        phrase_weights[phrase_id] = entity_score

                        if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                            # print(f"[dev] len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0") # dev add
                            phrase_weights[phrase_id] /= len(self.ent_node_to_chunk_ids[phrase_key])

                    if phrase not in phrase_scores:
                        phrase_scores[phrase] = []
                    phrase_scores[phrase].append(entity_score)

            # calculate average fact score for each phrase
            for phrase, scores in phrase_scores.items():
                linking_score_map[phrase] = float(np.mean(scores))

            if link_top_k:
                phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                            phrase_weights,
                                                                            linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

            #Get passage scores according to chosen dense retrieval model
            dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
            normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)
            
            passage_node_weight = 0 # dev add

            for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
                passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
                passage_dpr_score = normalized_dpr_sorted_scores[i]
                passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
                passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
                passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
                linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

            #Combining phrase and passage scores into one array for PPR
            node_weights = phrase_weights + passage_weights
            
            print(f"[dev] node_weights (len={len(node_weights)}) (sum={sum(node_weights)}): \n{node_weights}") # dev add

            #Recording top 30 facts in linking_score_map
            if len(linking_score_map) > 30:
                linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

            assert sum(node_weights) > 0, f'No phrases found in the graph for the given entities: {top_k_entities}'

            
            # #Running PPR algorithm based on the passage and phrase weights previously assigned
            # ppr_start = time.time()
            # ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
            # ppr_end = time.time()

            #Running PPR algorithm based on the passage and phrase weights previously assigned
            ppr_start = time.time()
            ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_rerank_supports(node_weights, damping=self.global_config.damping)
            ppr_end = time.time()


            self.ppr_time += (ppr_end - ppr_start)

            assert len(ppr_sorted_doc_ids) == len(
                self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

            # # (len=600) (type=<class 'numpy.ndarray'>)
            # print(f"[dev] ppr_sorted_doc_scores (len={len(ppr_sorted_doc_scores)}) (type={type(ppr_sorted_doc_scores)}): {ppr_sorted_doc_scores}")

            return ppr_sorted_doc_ids, ppr_sorted_doc_scores

        else:
            return dense_ids, dense_scores

        """
        # Step 1: Prepare doc_id -> index map
        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.passage_node_idxs)}
        binary_doc_scores = np.zeros(len(self.passage_node_idxs))

        # Step 2: Find docs containing the top_k_entities
        for e in top_k_entities:
            ent_phrase = e.lower()
            phrase_key = compute_mdhash_id(content=ent_phrase, prefix="entity-")
            doc_ids = self.ent_node_to_chunk_ids.get(phrase_key, set())

            for doc_id in doc_ids:
                if doc_id in doc_id_to_idx:
                    idx = doc_id_to_idx[doc_id]
                    binary_doc_scores[idx] = 1.0

        # Step 3: Sort the results — entities-matched docs first
        sorted_indices = np.argsort(-binary_doc_scores)  # 1.0 first
        ppr_sorted_doc_ids = np.array([self.passage_node_idxs[i] for i in sorted_indices])
        ppr_sorted_doc_scores = binary_doc_scores[sorted_indices]

        print(f"[dev] sum(ppr_sorted_doc_scores)/len(ppr_sorted_doc_scores): {sum(ppr_sorted_doc_scores)}/{len(ppr_sorted_doc_scores)}")
        return ppr_sorted_doc_ids, ppr_sorted_doc_scores
        """

    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        
        # load args
        link_top_k: int = self.global_config.linking_top_k
        
        # print(f"[dev] [TAG] [rerank_facts] link_top_k: {link_top_k}")
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            # Rerank the facts
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)

            rerank_metadata = getattr(self.rerank_filter, "last_metadata", {}) or {}
            rerank_cache_hit = bool(getattr(self.rerank_filter, "last_cache_hit", False))
            self._rerank_token_stats["prompt_tokens"] += rerank_metadata.get("prompt_tokens", 0)
            self._rerank_token_stats["completion_tokens"] += rerank_metadata.get("completion_tokens", 0)
            if rerank_cache_hit:
                self._rerank_token_stats["cache_hits"] += 1
            
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    

    def rerank_facts_and_rerank(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        
        # load args
        # link_top_k: int = self.global_config.linking_top_k
        link_top_k: int = np.inf
        
        # print(f"[dev] [TAG] [rerank_facts] link_top_k: {link_top_k}")
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [fact_row_dict[id]['content'] for id in real_candidate_fact_ids]
            
            # Rerank the facts
            rerank_filter = False
            if rerank_filter:
                top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                    candidate_facts,
                                                                                    candidate_fact_indices,
                                                                                    len_after_rerank=link_top_k)
            else:    
                top_k_fact_indices = candidate_fact_indices
                top_k_facts = candidate_facts
            
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """
        print(f"[Debug] [run_ppr]")
        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
    
    def run_rerank_supports(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        
        # import inspect
        # print(f"[dev] inspect.getmodule(self.graph.personalized_pagerank): {inspect.getmodule(self.graph.personalized_pagerank)}")
        # print(f"[dev] inspect.getfile(self.graph.personalized_pagerank): {inspect.getfile(self.graph.personalized_pagerank)}")
        
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
