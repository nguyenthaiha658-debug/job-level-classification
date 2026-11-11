"""Feature pipeline: TF-IDF on text, one-hot on low-cardinality categoricals."""

from __future__ import annotations

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import OneHotEncoder


class Densifier(BaseEstimator, TransformerMixin):
    """Convert a sparse matrix to dense after feature selection.

    Needed so that SMOTE and LogisticRegression(solver='lbfgs') can run
    on the reduced feature matrix. After SelectPercentile the width is
    a few hundred columns, so the dense form is inexpensive.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if sparse.issparse(X) else X


def build_preprocessor() -> ColumnTransformer:
    """Same feature families as the course script, rewritten cleanly.

    * title, industry: unigram TF-IDF
    * description: unigram+bigram TF-IDF, with min_df/max_df to drop
      extremely rare and extremely common terms
    * location, function: one-hot, ignoring unseen labels at test time
    """
    return ColumnTransformer(
        transformers=[
            ("title", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
            ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
            (
                "description",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=0.01,
                    max_df=0.95,
                ),
                "description",
            ),
            ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
            (
                "industry",
                TfidfVectorizer(stop_words="english", ngram_range=(1, 1)),
                "industry",
            ),
        ],
        remainder="drop",
    )


def build_selector(percentile: int = 10) -> SelectPercentile:
    return SelectPercentile(chi2, percentile=percentile)
