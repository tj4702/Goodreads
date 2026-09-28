# Goodreads Recommender

Ranks the unread books on my Goodreads shelf by how likely I am to enjoy them, learned from the reading history and review text of the books I've already read.

## The problem

A Goodreads library export gives you ratings and shelves, but nothing that describes what a book is *about*. Two books can share a genre and an author and still be nothing alike. So a recommender built only on the export's metadata has almost nothing to work with.

The approach here is to use **review text as a proxy for content**: if the crowd's reviews of an unread book use language similar to the reviews of books I rated highly, that book is a better bet.

## Data

| Source | What it provides | Included here |
|---|---|---|
| `Goodreads Library Export.csv` | My library: 307 books, ratings, shelves | Yes |
| `Goodreads Books Dataset.json` | Book metadata, UCSD Book Graph | No (several GB) |
| `Goodreads Reviews Dedup.json` | Public review text, UCSD Book Graph | No (several GB) |

The UCSD files come from the [UCSD Book Graph](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) dataset. Download them separately to reproduce.

After matching against UCSD:

- **96 read books** with both a rating and review text → training signal
- **98 of 165 to-read books** matched to review text → candidate pool

## Pipeline

1. **Clean the export.** Ratings of 0 mean "unrated", not "zero stars", so they're converted to NaN. Missing publishers and publication years filled from lookups.
2. **Pull review text** for both read and to-read books from the UCSD reviews file, concatenated per book.
3. **Normalise text** — lowercase, strip punctuation and digits, remove stopwords, lemmatise.
4. **Score sentiment** with VADER, averaged across individual reviews per book.
5. **Vectorise** both sets with a single TF-IDF fit (500 features, custom stopwords).
6. **Rank** candidates by cosine similarity to read books, weighted by how much I liked each one.

## Decisions worth explaining

**Both sides go through identical cleaning.** An early version cleaned candidate text properly but left the read-book text with punctuation intact, because of a `regex=False` flag that silently made a `str.replace` call a no-op. TF-IDF is literal about tokens, so `dont` and `don't` would have become different features and every similarity score would have been quietly wrong. Worth stating because it's invisible when it happens — no error, just worse results.

**One TF-IDF fit across both corpora, then transform each separately.** Fitting two vectorisers would produce two different vocabularies with different column orderings, making cosine similarity between the resulting matrices meaningless.

**Rating weights are centred on my own mean rating, not the 1–5 midpoint.** With raw ratings as weights, every book contributes positively and a 1-star book still nudges a candidate *up*. Centring means books I disliked push candidates down. My own mean is more defensible than 3.0 because it reflects how I actually use the rating scale.

**Sentiment had to be recomputed to be useful.** Scoring the concatenated review blob per book gave scores between 0.98 and 0.9995 — completely saturated, no discriminating power, and useless in a ranking. Recomputing as the mean of per-review scores gave a range of 0.00–0.73 with a standard deviation of 0.17. The signal was fine; the aggregation was wrong.

**Unrated read books imputed as 4.** Some books I read years ago were never rated. Rather than dropping them or filling with the mean, I filled with 4 based on recall — they're all books I remember liking. A judgement call, flagged here rather than buried.

## Validation

**Popularity confound check.** A recommender built on review text could easily just surface books with the most reviews. Correlation between match score and review-text length: **−0.04**. The ranking isn't tracking popularity.

**Rating vs sentiment correlation: 0.196.** Weak, and expected — a flatly-written review can accompany a 5-star rating. This is why the two signals are combined rather than treated as interchangeable.

**Face validity.** The top-ranked books are ones I'd genuinely want to read next.

## Limitations

- **Coverage: 98 of 165 candidates.** The remaining 67 aren't in the UCSD dataset and are unscored, not ranked low. They're excluded from the output rather than penalised.
- **Evaluation is face-validity only.** There's no held-out test set, because there's no ground truth for "would I have enjoyed this" on a book I haven't read. The popularity check rules out one specific confound; it doesn't establish accuracy.
- **Sentiment uses the first 5,000 characters** of each review. VADER's regex matching degrades badly on very long strings.
- **Cold-start:** a book with no reviews in the dataset can't be scored at all.

## Running it

```
pip install pandas numpy scikit-learn nltk vaderSentiment
```

Place the UCSD JSON files and the library export where the notebook expects them, then run `notebook.ipynb` top to bottom.
