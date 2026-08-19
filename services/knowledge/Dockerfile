FROM python:3.12-slim

WORKDIR /app

COPY services/knowledge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

ENV HF_HUB_OFFLINE=1

COPY src/knowledge.py src/
COPY knowledge/ knowledge/
COPY services/knowledge/app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
