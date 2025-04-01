import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from multirag.agents import NewsRetriever, ArticleRetriever
from multirag.engine import ModelEngine
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

@pytest.fixture
def model_engine():
    # Initialize Qwen model with lower precision for Colab
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval()
    return ModelEngine(model=model, tokenizer=tokenizer)

@pytest.fixture
def embeddings():
    # Use a smaller embedding model suitable for Colab
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@pytest.fixture
def article_retriever(model_engine, embeddings):
    return ArticleRetriever(
        llm_engine=model_engine,
        embeddings=embeddings,
    )

@pytest.fixture
def news_retriever(model_engine, article_retriever):
    return NewsRetriever(
        llm_engine=model_engine,
        article_retriever=article_retriever,
    )

def test_article_retriever_initialization(article_retriever):
    assert article_retriever.vector_db is None
    assert article_retriever.logs == []

def test_news_retriever_initialization(news_retriever):
    assert news_retriever.logs == []

def test_model_output_format(model_engine):
    test_prompt = "What is the latest news about AI?"
    response = model_engine.generate(test_prompt)
    assert isinstance(response, str)
    assert len(response) > 0