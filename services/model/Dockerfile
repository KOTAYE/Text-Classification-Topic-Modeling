FROM python:3.12-slim

WORKDIR /app

COPY services/model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The weights live in the image so the container starts fast and never depends
# on the Hugging Face Hub being reachable.
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
AutoTokenizer.from_pretrained('KOTAYE/xlm-roberta-base-ag-news'); \
AutoModelForSequenceClassification.from_pretrained('KOTAYE/xlm-roberta-base-ag-news')"

# Without this transformers calls the Hub on every load to check for updates.
ENV HF_HUB_OFFLINE=1

COPY src/classifier.py src/
COPY services/model/app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
