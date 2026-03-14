# Resume RAG Job Matching

Production-ready local RAG pipeline for resume ingestion and job matching.

## Stack

- Vector DB: ChromaDB (local persistent store)
- Embeddings: Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
- LLM: Optional local Ollama model for richer reasoning (scoring remains deterministic)

## Project Structure

```
RAG-based-resume-scanner/
├── resumes/
├── chroma_db/
├── config.py
├── resume_rag.py
├── job_matcher.py
├── requirements.txt
└── README.md
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ingest Resumes

Place PDF/DOCX resumes in `resumes/`, then run:

```bash
python3 resume_rag.py --resumes-dir resumes --rebuild
```

## Run Job Matching

```bash
python3 job_matcher.py \
  --job-description "Looking for an ML Engineer with 5+ years Python, NLP, and AWS." \
  --top-k 10
```

Interactive CLI mode:

```bash
python3 job_matcher.py --interactive --top-k 10
```

You can then type job descriptions repeatedly, apply optional filter prompts, and get JSON output each time.

Optional local LLM reasoning (Ollama server):

```bash
# one-time: install/pull model
ollama pull llama3.2:3b

python3 job_matcher.py \
  --job-description "Looking for an ML Engineer with 5+ years Python, NLP, and AWS." \
  --top-k 10 \
  --use-llm-reasoning \
  --llm-provider ollama \
  --llm-model llama3.2:3b \
  --llm-base-url http://localhost:11434 \
  --llm-reasoning-candidates 5
```

Optional local LLM reasoning (in-process HuggingFace model):

```bash
python3 job_matcher.py \
  --job-description "Looking for an ML Engineer with 5+ years Python, NLP, and AWS." \
  --top-k 10 \
  --use-llm-reasoning \
  --llm-provider hf-local \
  --llm-model google/flan-t5-small \
  --llm-reasoning-candidates 3
```

Optional hard filters:

```bash
python3 job_matcher.py \
  --job-file jd.txt \
  --required-skills "Python,Machine Learning,AWS" \
  --min-years 5
```

## Output

`job_matcher.py` returns JSON in this shape:

```json
{
  "job_description": "...",
  "top_matches": [
    {
      "candidate_name": "John Doe",
      "resume_path": "resumes/john_doe.pdf",
      "match_score": 92,
      "matched_skills": ["Python", "Machine Learning"],
      "relevant_excerpts": ["..."],
      "reasoning": "Strong match for ML experience..."
    }
  ]
}
```

Notes:
- The `match_score` is always computed by local ranking logic.
- LLM updates only the `reasoning` field for top matches.
- If the chosen local LLM is unavailable, matcher automatically falls back to rule-based reasoning.
