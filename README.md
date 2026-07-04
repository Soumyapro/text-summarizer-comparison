# Text Summarizer: Extractive vs Abstractive

A hands-on comparison of two fundamentally different approaches to automatic text summarization - implemented, evaluated, and visualized in a single notebook.

## What this does

Given a piece of text, this project produces a summary two different ways and compares them:

- **Extractive summarization** — selects the most important existing sentences and stitches them together, using TF-IDF vectorization, cosine similarity, and the TextRank algorithm (a PageRank variant applied to sentences).
- **Abstractive summarization** — generates entirely new text using a pretrained transformer model (`sshleifer/distilbart-cnn-12-6`, a distilled version of BART fine-tuned on CNN/DailyMail).

Both summaries are then scored against a reference summary using ROUGE metrics, and the results are visualized side by side.

## How it works

### Extractive pipeline
1. Split the input text into sentences (`nltk.sent_tokenize`)
2. Vectorize each sentence with TF-IDF, ignoring English stop words
3. Compute pairwise cosine similarity between all sentences
4. Build a graph from the similarity matrix and rank sentences with PageRank
5. Select the top-N highest-scoring sentences and return them in their original order

### Abstractive pipeline
1. Tokenize the input text (truncated to the model's 1024-token limit)
2. Run it through DistilBART's encoder-decoder architecture
3. Generate a summary autoregressively using beam search (4 beams), with a length penalty and repeated-n-gram blocking to keep output fluent and non-repetitive
4. Decode the generated token IDs back into text

### Evaluation
Both summaries are scored against a hand-written reference summary using ROUGE-1, ROUGE-2, and ROUGE-L (precision, recall, and F1), and the results are plotted alongside a word-count comparison showing how much each method compresses the original text.

## Setup

```bash
pip install nltk numpy pandas scikit-learn networkx torch transformers rouge-score matplotlib
```

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

The notebook was built for Google Colab with a T4 GPU, but the abstractive model will also run on CPU (just slower). No GPU is required for the extractive method.

## Usage

Open the notebook and run all cells in order. To summarize your own text, replace the `sample_text` variable, then re-run:

```python
extractive_summary = summarize_extractive(your_text, num_sentences=3)
abstractive_summary = summarize_abstractive(your_text)
```

## Results

On the single sample paragraph included in the notebook:

| Metric | Extractive | Abstractive |
|---|---|---|
| ROUGE-1 F1 | 0.122 | 0.141 |
| ROUGE-2 F1 | 0.025 | 0.000 |
| ROUGE-L F1 | 0.098 | 0.113 |
| Word count | 46 | ~45 |

Note on evaluation: ROUGE is not a fully reliable way to judge summary quality, because it only measures word overlap with a reference summary, not meaning. It counts matching words (or word sequences) between your generated summary and a reference, so it rewards copying the reference's exact wording and penalizes legitimate paraphrasing, even when the paraphrase is accurate. A summary that says "opaque" instead of the reference's "difficult to interpret" would score zero overlap on that phrase despite meaning the same thing. It also cannot detect hallucination, since a fluent summary that states something false would score fine as long as it happens to share vocabulary with the reference. And ROUGE scores are only meaningful when averaged across many examples with independent, high quality references; a single test example (like the one in this notebook) is inherently noisy and should not be read as a verdict on either method's quality.

## Known limitations

- **Single test example.** Proper evaluation would require scoring both methods across many documents from a dataset with reference summaries (e.g. CNN/DailyMail).
- **Hand-written reference summary.** Using your own reference biases the comparison toward your personal writing style rather than an independent standard.
- **1024-token input limit.** The abstractive model truncates anything beyond this, so long documents will lose content past that point without a chunking strategy.
- **ROUGE measures word overlap, not meaning.** A well-paraphrased summary can score poorly on ROUGE even if a human would judge it as accurate.
- **O(n²) similarity computation.** The extractive method's pairwise similarity matrix doesn't scale gracefully to very long documents.

## Possible extensions

- Evaluate on a proper dataset (CNN/DailyMail, XSum) across multiple examples and average the scores
- Add BERTScore or a factual-consistency metric alongside ROUGE to catch semantic similarity and hallucination that ROUGE can't detect
- Add chunk-and-merge handling for documents longer than 1024 tokens

## Tech stack

- `nltk` — sentence tokenization
- `scikit-learn` — TF-IDF vectorization and cosine similarity
- `networkx` — graph construction and PageRank
- `transformers` / `torch` — the DistilBART model
- `rouge-score` — evaluation
- `matplotlib` — visualization
