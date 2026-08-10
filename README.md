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
