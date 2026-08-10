FROM python:3.12-slim

WORKDIR /app

# Dependencies first: this layer is cached and only rebuilds when
# requirements.txt changes, so editing source code stays a fast rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake both models into the image rather than fetching them on first run.
# Costs image size, but the container then starts fast and works with no
# network at all — which is what you want when demoing on someone else's wifi.
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
AutoTokenizer.from_pretrained('KOTAYE/xlm-roberta-base-ag-news'); \
AutoModelForSequenceClassification.from_pretrained('KOTAYE/xlm-roberta-base-ag-news')" \
 && python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Without this transformers still calls the Hub on every load to check for
# updates, so the "works offline" claim above only holds with it set. Both
# models are already in the image, so a cache miss here should be a loud
# failure rather than a silent download.
ENV HF_HUB_OFFLINE=1

COPY src/ src/
COPY knowledge/ knowledge/

# The API key is never baked in. Pass it at run time:
#   docker run -it --env-file src/.env news-agent
CMD ["python", "src/agent.py"]
