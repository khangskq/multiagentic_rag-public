from dataclasses import field
from importlib import resources
from multirag import prompts
from pydantic.dataclasses import dataclass
from datetime import timedelta


@dataclass
class RetrievalCfg:
    chunk_size: int = 512
    chunk_overlap: int = 256
    n_passages: int = 5
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ".", ",", " ", ""])
    max_iterations: int = 3
    max_days_back: int = 7


@dataclass
class SearchCfg:
    n_articles: int = 50
    max_summary_chars: int = 1000
    languages: list[str] = field(default_factory=lambda: ["en"])
    sort_by: str = "relevancy"


@dataclass
class EmbeddingsCfg:
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_model_kwargs: dict = field(default_factory=dict)
    encode_kwargs: dict = field(default_factory=dict)


@dataclass
class EngineCfg:
    summarize_past: bool = True
    max_new_tokens: int = 1024
    temperature: float|None = 0.7
    top_k: int|None = 50
    top_p: float|None = 0.9
    do_sample = True


@dataclass
class SystemPrompts:
    manager_system_prompt: str = resources.read_text(prompts, 'manager_system_prompt.txt')
    news_search_system_prompt: str = resources.read_text(prompts, 'news_retriever_system_prompt.txt')
    article_retriever_system_prompt: str = resources.read_text(prompts, 'article_retriever_system_prompt.txt')
    chat_system_prompt: str = resources.read_text(prompts, 'chat_system_prompt.txt')
