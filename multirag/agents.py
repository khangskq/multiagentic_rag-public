from typing import List, Optional
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

from multirag.cfg import RetrievalCfg, SearchCfg

load_dotenv()

class ArticleRetriever:
    def __init__(
        self,
        llm_engine,
        embeddings: HuggingFaceEmbeddings,
        retrieval_cfg: RetrievalCfg = RetrievalCfg()
    ):
        self.llm_engine = llm_engine
        self.embeddings = embeddings
        self.retrieval_cfg = retrieval_cfg
        self.vector_db = None
        self.logs = []

    def compute_article_embeddings(self, content: str) -> None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.retrieval_cfg.chunk_size,
            chunk_overlap=self.retrieval_cfg.chunk_overlap,
            separators=self.retrieval_cfg.separators
        )
        article_splits = splitter.create_documents(texts=[content])
        self.vector_db = FAISS.from_documents(article_splits, self.embeddings)

    def retrieve_passages(self, query: str) -> List[str]:
        if not self.vector_db:
            return []
        passages = self.vector_db.similarity_search(
            query, 
            k=self.retrieval_cfg.n_passages
        )
        return [p.page_content for p in passages]

class NewsRetriever:
    def __init__(
        self,
        llm_engine,
        article_retriever: ArticleRetriever,
        search_cfg: SearchCfg = SearchCfg()
    ):
        self.newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        self.llm_engine = llm_engine
        self.article_retriever = article_retriever
        self.search_cfg = search_cfg
        self.logs = []

    def search_news(self, query: str) -> List[dict]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        response = self.newsapi.get_everything(
            q=query,
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by=self.search_cfg.sort_by,
            page_size=self.search_cfg.n_articles
        )
        print(response['articles'])
        return response['articles']

    def chat(self, query: str) -> str:
        # Search for relevant articles
        articles = self.search_news(query)
        
        if not articles:
            return "No relevant news articles found."
            
        # Process articles and create embeddings
        all_content = []
        for article in articles:
            content = f"Title: {article['title']}\n{article['description']}\n{article['content']}"
            all_content.append(content)
            self.article_retriever.compute_article_embeddings(content)
            
        # Retrieve relevant passages
        relevant_passages = self.article_retriever.retrieve_passages(query)
        
        # Generate response using the language model
        context = "\n".join(relevant_passages)
        prompt = f"""Based on the following news articles, answer this question: {query}

Context:
{context}

Answer:"""
        
        return self.llm_engine.generate(prompt)
