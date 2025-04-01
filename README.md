# News Chat AI System with Qwen2.5

An AI-powered news chat system using Qwen2.5-7B-Instruct model and NewsAPI.

## Features

- Real-time news search and retrieval with NewsAPI
- Qwen2.5-7B-Instruct model integration
- Semantic search within articles
- Google Colab compatibility
- Memory-efficient processing

## Prerequisites

- Python 3.12 or higher
- NewsAPI key (get one at https://newsapi.org)
- GPU with 16GB+ VRAM (or Google Colab)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multiagentic_rag-public
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Create .env file:
```bash
cat << EOF > .env
NEWS_API_KEY=your_api_key_here
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
MAX_TOKENS=500
TEMPERATURE=0.7
LOG_LEVEL=INFO
EOF
```

## Project Structure

```
multirag/
├── agents.py           # News retrieval agents
├── engine.py          # Model engine implementation
├── cfg.py             # Configuration classes
├── prompts/           # System prompts
└── utils/
    └── cache.py       # Caching utilities

tests/
└── test_agents.py     # Agent tests
```

## Usage

### Local Development
```python
from multirag.agents import NewsRetriever, ArticleRetriever
from multirag.engine import ModelEngine
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize components
model_engine = ModelEngine()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
article_retriever = ArticleRetriever(
    llm_engine=model_engine, 
    embeddings=embeddings
)
news_retriever = NewsRetriever(
    llm_engine=model_engine, 
    article_retriever=article_retriever
)

# Query news
response = news_retriever.chat("What are the latest AI developments?")
print(response)
```

### Google Colab
Open `News_Chat_Demo.ipynb` in Google Colab and follow the setup instructions.

## Testing

Run tests:
```bash
pytest tests/
```

## Troubleshooting

1. Memory issues:
   - Use Google Colab with GPU runtime
   - Reduce model precision to float16
   - Use smaller embedding model

2. API errors:
   - Verify NewsAPI key in .env
   - Check NewsAPI rate limits
   - Review logs for error messages

## License

MIT License - see LICENSE file for details
