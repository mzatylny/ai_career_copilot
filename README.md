# AI Career Copilot API

A polished FastAPI backend for an AI-engineering portfolio project. It analyzes a CV against a job description, creates a learning roadmap, indexes uploaded PDFs into a vector database, answers questions with source chunks, and simulates technical interviews.

## Features

This is not a basic chatbot. It demonstrates:

- API design with FastAPI
- Pydantic validation and typed response models
- OpenAI structured JSON outputs
- RAG over uploaded PDFs
- ChromaDB vector search with session-level isolation
- Resume/job matching
- Interview feedback
- Docker-ready deployment
- Mock mode for demos without an API key

## Project structure

```text
ai_career_copilot_wow/
├── app/
│   ├── main.py          # FastAPI endpoints
│   ├── models.py        # Request/response schemas
│   ├── ai_services.py   # LLM prompts + structured outputs
│   ├── rag_engine.py    # PDF extraction, embeddings, ChromaDB
│   ├── config.py        # Environment settings
│   └── utils.py         # Helpers
├── tests/
│   └── test_api.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your key:

```bash
OPENAI_API_KEY=sk-your-key-here
```

Run:

```bash
./run_dev.sh
```

Open:

```text
http://localhost:8000/docs
```

## Free mock mode

For screenshots, tests, or demos without spending money:

```bash
AI_COPILOT_MOCK_LLM=true
AI_COPILOT_MOCK_EMBEDDINGS=true
uvicorn app.main:app --reload
```

## Main endpoints

### Health check

```bash
curl http://localhost:8000/api/health
```

### Analyze CV/job gap

```bash
curl -X POST http://localhost:8000/api/analyze-gap \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Python developer with FastAPI and data projects...",
    "job_description_text": "We need an AI Engineer with Python, RAG, APIs, Docker and SQL...",
    "target_seniority": "junior"
  }'
```

### Generate roadmap

```bash
curl -X POST http://localhost:8000/api/generate-roadmap \
  -H "Content-Type: application/json" \
  -d '{
    "missing_skills": ["RAG", "Docker", "Vector databases"],
    "timeframe_days": 90,
    "hours_per_week": 10,
    "target_role": "AI Engineer"
  }'
```

### Upload a PDF

```bash
curl -X POST http://localhost:8000/api/upload-document \
  -F "session_id=demo123" \
  -F "file=@resume.pdf"
```

### Chat with uploaded documents

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo123",
    "user_query": "What are the strongest AI skills in this CV?"
  }'
```

### Generate interview question

```bash
curl -X POST http://localhost:8000/api/interview/question \
  -H "Content-Type: application/json" \
  -d '{
    "target_role": "AI Engineer",
    "seniority": "junior",
    "focus_skills": ["Python", "FastAPI", "RAG"]
  }'
```

### Evaluate interview answer

```bash
curl -X POST http://localhost:8000/api/interview/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How would you build a RAG API?",
    "user_answer": "I would upload PDFs, embed chunks, retrieve relevant text and answer with sources.",
    "target_role": "AI Engineer"
  }'
```

## Run tests

```bash
AI_COPILOT_MOCK_LLM=true AI_COPILOT_MOCK_EMBEDDINGS=true pytest
```

## Docker

```bash
cp .env.example .env
# edit .env

docker compose up --build
```

## Demo script for GitHub / university presentation

1. Open `/docs`.
2. Show `/api/health` returning `ok`.
3. Paste a CV and AI Engineer job description into `/api/analyze-gap`.
4. Generate a roadmap from the missing skills.
5. Upload a PDF under `session_id=demo123`.
6. Ask a question with `/api/chat` and show sources.
7. Generate and evaluate an interview answer.
8. Explain that mock mode allows safe demos and real OpenAI mode enables production behavior.

## What to add later

- Frontend dashboard in Next.js
- User authentication
- PostgreSQL for user history
- Stripe/pricing if you want SaaS style
- Deployment to Render/Fly.io/Railway
- Analytics dashboard for job match progress
