# Node is needed to build the frontend but not to run anything, so it stays in
# a stage of its own and only the built files cross over.
FROM node:24-slim AS web

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build


FROM python:3.12-slim

WORKDIR /app

# Dependencies first: this layer is cached and only rebuilds when
# requirements.txt changes, so editing source code stays a fast rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake both models into the image rather than fetching them on first run.
# Costs image size, but the container then starts fast and never depends on
# the Hugging Face Hub being reachable — which is not something to discover
# during a demo.
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
AutoTokenizer.from_pretrained('KOTAYE/xlm-roberta-base-ag-news'); \
AutoModelForSequenceClassification.from_pretrained('KOTAYE/xlm-roberta-base-ag-news')" \
 && python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Without this transformers still calls the Hub on every load to check for
# updates, so the "works offline" claim above only holds with it set. Both
# models are already in the image, so a cache miss here should be a loud
# failure rather than a silent download.
ENV HF_HUB_OFFLINE=1

COPY src/ src/
COPY knowledge/ knowledge/
COPY --from=web /web/dist web/dist

# Three front-ends over one agent. Terminal by default; the API key is never
# baked in, so credentials come from --env-file at run time.
#
#   terminal:  docker run -it --rm --env-file src/.env news-agent
#
#   web:       docker run --rm -p 8000:8000 --env-file src/.env news-agent \
#                python -m uvicorn api:app --host 0.0.0.0 --port 8000 --app-dir src
#
#   telegram:  docker run --rm --env-file src/.env news-agent \
#                python src/telegram_bot.py
CMD ["python", "src/agent.py"]
