# Node builds the frontend and stays behind: only the built files cross over.
FROM node:24-slim AS web

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build


FROM python:3.12-slim

WORKDIR /app

COPY services/agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY --from=web /web/dist web/dist

# Web interface by default. The terminal app and the Telegram bot are the same
# image with a different command — see docker-compose.yml.
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
