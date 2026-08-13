# Text Classification / Topic Modeling

News topic classification with a fine-tuned transformer, wrapped in an agent
with tool calling and retrieval.

UnidataLab ML Summer Camp 2026 — topic 8.

**Model:** [KOTAYE/xlm-roberta-base-ag-news](https://huggingface.co/KOTAYE/xlm-roberta-base-ag-news)

---

## Results

`xlm-roberta-base` fine-tuned on [AG News](https://huggingface.co/datasets/fancyzhx/ag_news)
to classify English news into four topics: **World**, **Sports**, **Business**, **Sci/Tech**.

All numbers below are measured on the held-out test split (7,600 examples),
which was used exactly once, after every design decision was final.

| Method | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 91.72% | 0.9171 |
| **xlm-roberta-base (fine-tuned)** | **94.54%** | **0.9453** |
| Camp requirement | 80% | — |

The fine-tuned model cuts the baseline's error rate from 8.28% to 5.46% —
roughly a third of its mistakes removed.

### Model selection

Both candidates were trained under identical hyperparameters and compared on
the validation split:

| Model | Accuracy (val) |
|---|---|
| `roberta-base` | 94.64% |
| `xlm-roberta-base` | 94.14% |

`xlm-roberta-base` was selected despite being ~0.5 points behind. It covers
100 languages, which keeps a single pipeline usable if the project is extended
to Ukrainian, instead of maintaining two.

### Hyperparameters

A learning rate sweep was run at fixed everything else:

| Learning rate | Accuracy (val) |
|---|---|
| 1e-5 | 93.93% |
| **2e-5** | **94.35%** |
| 3e-5 | 94.29% |

Final configuration: 2 epochs, lr 2e-5, batch size 32, max sequence length 128,
fp16, warmup 400 steps, weight decay 0.01. Trained on a single Tesla T4 in
about 26 minutes.

`max_length=128` was chosen by measuring the token-length distribution: it
holds 99.2% of texts in full, while 64 would truncate one text in six and 256
would cost roughly four times the compute for another 0.8% of coverage.

Two epochs rather than three: training loss kept falling while validation loss
had flattened, which is where overfitting starts.

## Error analysis

703 of 12,000 validation predictions are wrong. **369 of them (52%) are
Business ↔ Sci/Tech confusion** — one pair of classes accounts for more errors
than the other five pairs combined.

| Class | F1 (val) |
|---|---|
| Sports | 0.986 |
| World | 0.951 |
| Sci/Tech | 0.918 |
| Business | 0.910 |

Reading the failures by hand suggests most are genuinely ambiguous rather than
model errors:

```
[Business → Sci/Tech]  AT&T to Cut About 7,000 Jobs
[Business → Sci/Tech]  CA Taps IBM Vet John Swainson As CEO
[Sci/Tech → Business]  Monti: Courts must rule on MS anti-trust
```

AG News also carries label noise. A story about Tiger Woods needing to win a
major is labelled `World`; the model predicted `Sports` and is arguably right.
Around 94-95% looks like a practical ceiling for this dataset.

The raw text contains leaked HTML entities (`&lt;b&gt;`), stripped ampersands
("AT T"), and stray backslashes. These were left untouched — the model handles
them, and cleaning risked discarding signal.

## Ukrainian, without retraining

`xlm-roberta-base` was pretrained on 100 languages, so a model fine-tuned only
on English AG News can be pointed at Ukrainian text as-is. This is what the
0.5-point sacrifice against `roberta-base` was for.

**Translated AG News.** 500 test items were machine-translated into Ukrainian
(`scripts/build_ukrainian_eval.py`). Translation does not change what a story
is about, so the labels carry over, and scoring the same items in both
languages isolates the language from the sample:

| | Accuracy | Macro F1 |
|---|---|---|
| English | 92.20% | 0.9233 |
| Ukrainian | 89.80% | 0.8996 |

**2.40 points** lost, and the model returns the same verdict in both languages
for **95.6%** of stories. Note that this 500-item subsample scores 92.20% in
English against 94.54% on the full test set — which is exactly why both
languages are measured on the same items rather than against the headline
figure.

**Genuine Ukrainian news.** Machine translation reads like English wearing
Ukrainian grammar, so the figure above is optimistic. As a cross-check,
`scripts/eval_ukrainian_real.py` runs the classifier over
`Zarakun/ukrainian_news`, mapping its rubrics onto these four classes:

| Ukrainian rubric | Mapped to | Model agrees |
|---|---|---|
| `sport` | Sports | 92% |
| `tech` | Sci/Tech | 93% |
| `economy` | Business | 82% |
| `financy` | Business | 67% |
| `business` | Business | 55% |
| `zakordon` | World | 28% |

Overall accuracy is 69.0%, or **79.3% excluding `zakordon`** — and that
exclusion is justified rather than convenient. "Zakordon" means "abroad", and
the rubric holds fuel prices in Germany, labour shortages in the Netherlands
and advice on learning languages. AG News' "World" means international
politics and conflict. Mapping one to the other was a mistake in the
evaluation, not a failure in the model.

The `business` rubric splitting 55/40 between Business and Sci/Tech is the
same confusion documented on English AG News above, showing up unchanged in
Ukrainian: a property of the label scheme rather than of the language.

## Data

| Split | Size | Purpose |
|---|---|---|
| train | 108,000 | fine-tuning |
| validation | 12,000 | epoch, model and hyperparameter selection |
| test | 7,600 | final measurement, used once |

AG News ships 120,000 train and 7,600 test examples, perfectly balanced at
30,000 per class. The validation split was carved out of train at 10%,
stratified, seed 42. The test split was never used for any decision.

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="KOTAYE/xlm-roberta-base-ag-news")
clf("Manchester United signs new striker ahead of the season")
# [{'label': 'Sports', 'score': 0.99}]
```

## Repository

```
notebooks/    training and evaluation notebook
src/          agent, tools and retrieval (step 2)
data/         datasets — not tracked
models/       weights — not tracked
```

Training runs on Google Colab with a T4 GPU. Weights are published to the
Hugging Face Hub rather than committed here.

## The agent

A terminal agent with three tools:

| Tool | What it does |
|---|---|
| `classify_news` | the fine-tuned classifier above, running locally |
| `about_student` | retrieval over `knowledge/about_me.md` |
| `send_telegram_message` | delivers a message to Telegram, over MCP |

```bash
python src/agent.py
```

Ask it to classify a headline and it calls the transformer. Ask about the
student and it searches the notes, answering "I do not have that information"
when they do not cover the question rather than inventing one. Ask it to send
something to Telegram and it chains two tools: classify first, then send.

### Telegram over MCP

`src/telegram_mcp_server.py` is a Model Context Protocol server exposing
`send_telegram_message` and `get_telegram_bot_name`. The agent starts it as a
subprocess and talks to it over stdio, so nothing listens on a port and the
container stays a single process.

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `src/.env` — see
`src/.env.example`. Without them the agent starts with the other two tools
instead of refusing to run.

One thing worth knowing: Telegram puts the bot token in the URL path, and
`httpx` logs request URLs at INFO level, so an unconfigured setup prints the
token to the console on every call. The server sets that logger to WARNING.

## Running in Docker

```bash
docker build -t news-agent .
docker run -it --rm --env-file src/.env news-agent
```

Both models are baked into the image at build time and `HF_HUB_OFFLINE=1` is
set, so the container never contacts the Hugging Face Hub at run time. The
image is around 4.3 GB: roughly 1.5 GB of dependencies (mostly torch) and
1.2 GB of model weights.

The API key is not in the image — `src/.env` is excluded by `.dockerignore`
and passed in at run time instead.

Note that the agent still needs a network connection: the classifier and the
retrieval index run locally, but the chat model that drives them is a hosted
API.

## Project status

- [x] **Step 1** — fine-tune a transformer for topic classification
- [ ] **Step 2** — agent with two tools: the classifier, and retrieval over
      documents about the student
- [ ] **Step 3** — Docker, MCP integration with Telegram, demo

---

Author: Viktor Syrotiuk ([KOTAYE](https://github.com/KOTAYE))
