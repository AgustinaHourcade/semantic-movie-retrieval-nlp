import os
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

# Download nltk data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Path resolution
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

class BM25Searcher:
    def __init__(self, df, text_column='content_narrative', index_path=None):
        self.df = df
        self.text_column = text_column
        self.index_path = index_path if index_path is not None else os.path.join(DATA_DIR, 'bm25_index.pkl')
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'\w+')
        
        if os.path.exists(self.index_path):
            print(f"Loading precalculated BM25 index from {self.index_path}...")
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
            print("BM25 Index loaded successfully.")
        else:
            print("Tokenizing corpus for BM25...")
            corpus = self.df[self.text_column].fillna('').tolist()
            tokenized_corpus = [self._tokenize(doc) for doc in corpus]
            print("Building BM25 Index...")
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"Saving BM25 index to {self.index_path}...")
            with open(self.index_path, 'wb') as f:
                pickle.dump({'bm25': self.bm25}, f)
            print("BM25 Index built and saved successfully.")

    def _tokenize(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        return [t for t in tokens if t not in self.stop_words]

    def search(self, query, top_k=5):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            title = self.df.iloc[idx]['title']
            results.append({
                'title': title,
                'score': float(scores[idx]),
                'rank': rank,
                'method': 'bm25'
            })
        return results


class DenseSearcher:
    def __init__(self, df, model_name='multi-qa-MiniLM-L6-cos-v1', embedding_col='embedding', index_path=None, map_path=None):
        self.df = df
        self.embedding_col = embedding_col
        self.index_path = index_path if index_path is not None else os.path.join(DATA_DIR, 'faiss_index.bin')
        self.map_path = map_path if map_path is not None else os.path.join(DATA_DIR, 'faiss_id_map.pkl')
        
        print(f"Loading SentenceTransformer model ({model_name})...")
        self.model = SentenceTransformer(model_name)
        
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            print(f"Loading precalculated FAISS index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
            with open(self.map_path, 'rb') as f:
                self.id_map = pickle.load(f)
            print("FAISS Index and ID map loaded successfully.")
        else:
            print("Building FAISS Index...")
            # Extract embeddings as a numpy array
            embeddings = np.stack(self.df[self.embedding_col].values).astype('float32')
            # Normalize embeddings for cosine similarity search (inner product)
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            
            # Map index offset to DataFrame index/ID
            self.id_map = {i: self.df.index[i] for i in range(len(self.df))}
            
            print(f"Saving FAISS index to {self.index_path}...")
            faiss.write_index(self.index, self.index_path)
            with open(self.map_path, 'wb') as f:
                pickle.dump(self.id_map, f)
            print("FAISS Index built and saved successfully.")

    def search(self, query, top_k=5):
        # Embed the query
        query_embedding = self.model.encode([query]).astype('float32')
        # Normalize query vector
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        scores = scores[0]
        indices = indices[0]
        
        results = []
        for rank, (score, offset) in enumerate(zip(scores, indices), start=1):
            if offset == -1: # FAISS padding if not enough results
                continue
            df_idx = self.id_map[int(offset)]
            title = self.df.loc[df_idx, 'title']
            results.append({
                'title': title,
                'score': float(score),
                'rank': rank,
                'method': 'dense'
            })
        return results


# ---------------------------------------------------------------------------
# Hybrid Searcher — combines BM25 (lexical) and Dense (semantic) retrieval
# with multi-field exact-match boosting and configurable fusion strategy.
# ---------------------------------------------------------------------------

class HybridSearcher:
    """Hybrid search that fuses BM25 and Dense scores.

    Fusion strategies:
        * ``'alpha'`` (default) – Min-Max normalises raw scores from both
          retrievers to [0, 1] and interpolates them:
              ``score = alpha * dense + (1 - alpha) * bm25``
        * ``'rrf'`` – Reciprocal Rank Fusion with a configurable *k* constant.

    Exact-match boosting:
        When ``use_exact_match=True``, rows whose **title**, **cast** or
        **director** fields contain the query string (case-insensitive) are
        promoted to the very top of the results list.
    """

    # Fields checked for exact-match boosting (order = priority).
    _EXACT_MATCH_FIELDS = ['title', 'cast', 'director']

    def __init__(self, bm25_searcher: BM25Searcher, dense_searcher: DenseSearcher):
        self.bm25_searcher = bm25_searcher
        self.dense_searcher = dense_searcher
        self.df = self.bm25_searcher.df

        # Pre-compute lowercased versions of boost fields for O(n) substring
        # matching at search time (avoids re-lowering on every call).
        self._fields_lower: dict[str, list[str]] = {}
        for field in self._EXACT_MATCH_FIELDS:
            if field in self.df.columns:
                self._fields_lower[field] = (
                    self.df[field].fillna('').str.lower().tolist()
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        method: str = 'alpha',
        alpha: float = 0.1,
        rrf_k: int = 60,
        use_exact_match: bool = True,
    ) -> list[dict]:
        """Run a hybrid search and return the top-k results.

        Parameters
        ----------
        query : str
            Natural-language search query.
        top_k : int
            Number of results to return.
        method : ``'alpha'`` | ``'rrf'``
            Fusion strategy.
        alpha : float
            Interpolation weight for the dense retriever (only used when
            ``method='alpha'``). ``0.0`` = pure BM25, ``1.0`` = pure Dense.
        rrf_k : int
            Smoothing constant for RRF (only used when ``method='rrf'``).
        use_exact_match : bool
            Whether to boost exact substring matches in title / cast /
            director to the top of results.

        Returns
        -------
        list[dict]
            Each dict contains ``title``, ``score``, ``rank``, ``method``.
        """
        # ----- 1. Exact-match boost (multi-field) -------------------------
        boosted_indices: list[int] = []
        if use_exact_match:
            boosted_indices = self._find_exact_matches(query)

        # Collect boosted results (already ranked by field priority).
        results: list[dict] = []
        boosted_set = set(boosted_indices)
        for idx in boosted_indices:
            results.append(self._make_result(idx, score=999.0, results_so_far=results))
            if len(results) >= top_k:
                return results

        # ----- 2. Retrieve raw scores from both retrievers ----------------
        remaining_k = top_k - len(results)
        initial_k = max(remaining_k * 3, 50)

        bm25_scores_all = self._raw_bm25_scores(query)
        dense_scores_map, dense_faiss_indices = self._raw_dense_scores(query, initial_k)

        # ----- 3. Fuse scores according to the chosen strategy ------------
        if method == 'rrf':
            fused = self._fuse_rrf(
                bm25_scores_all, dense_faiss_indices,
                boosted_set, initial_k, rrf_k,
            )
        elif method == 'alpha':
            fused = self._fuse_alpha(
                bm25_scores_all, dense_scores_map,
                boosted_set, initial_k, alpha,
            )
        else:
            raise ValueError(f"Unknown fusion method '{method}'. Use 'alpha' or 'rrf'.")

        # ----- 4. Sort and assemble final results -------------------------
        sorted_fused = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        for idx, score in sorted_fused:
            results.append(self._make_result(idx, score=score, results_so_far=results))
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Exact-match helpers
    # ------------------------------------------------------------------

    def _find_exact_matches(self, query: str) -> list[int]:
        """Return row indices whose title/cast/director contain *query*.

        Results are deduplicated and ordered by field priority (title first,
        then cast, then director) so that a title hit always ranks above a
        cast-only hit.
        """
        query_lower = query.lower()
        seen: set[int] = set()
        ordered: list[int] = []

        for field, values in self._fields_lower.items():
            for i, val in enumerate(values):
                if i not in seen and query_lower in val:
                    seen.add(i)
                    ordered.append(i)
        return ordered

    # ------------------------------------------------------------------
    # Raw-score retrieval (thin wrappers around BM25 / FAISS internals)
    # ------------------------------------------------------------------

    def _raw_bm25_scores(self, query: str) -> np.ndarray:
        """Return the full BM25 score array across the corpus."""
        tokenized_query = self.bm25_searcher._tokenize(query)
        return self.bm25_searcher.bm25.get_scores(tokenized_query)

    def _raw_dense_scores(self, query: str, k: int) -> tuple[dict[int, float], np.ndarray]:
        """Return dense scores as {df_row_offset: score} and raw FAISS indices."""
        query_embedding = self.dense_searcher.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        faiss_scores, faiss_indices = self.dense_searcher.index.search(query_embedding, k)
        faiss_scores = faiss_scores[0]
        faiss_indices = faiss_indices[0]

        scores_map: dict[int, float] = {}
        for score, offset in zip(faiss_scores, faiss_indices):
            if offset == -1:
                continue
            row_idx = self.dense_searcher.id_map[int(offset)]
            scores_map[row_idx] = float(score)

        return scores_map, faiss_indices

    # ------------------------------------------------------------------
    # Fusion strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        """Scale *values* to [0, 1] via Min-Max. Returns zeros if range is 0."""
        v_min, v_max = values.min(), values.max()
        span = v_max - v_min
        if span == 0:
            return np.zeros_like(values)
        return (values - v_min) / span

    def _fuse_alpha(
        self,
        bm25_scores_all: np.ndarray,
        dense_scores_map: dict[int, float],
        exclude: set[int],
        initial_k: int,
        alpha: float,
    ) -> dict[int, float]:
        """Min-Max normalised interpolation: α·dense + (1-α)·bm25."""
        # --- Collect the candidate pool (union of top-k from each) --------
        bm25_top = np.argsort(bm25_scores_all)[::-1][:initial_k]
        candidates = set(int(i) for i in bm25_top) | set(dense_scores_map.keys())
        candidates -= exclude

        if not candidates:
            return {}

        # --- Extract raw scores for the candidate pool --------------------
        candidate_list = sorted(candidates)
        bm25_raw = np.array([bm25_scores_all[i] for i in candidate_list], dtype=np.float64)
        dense_raw = np.array(
            [dense_scores_map.get(i, 0.0) for i in candidate_list], dtype=np.float64,
        )

        # --- Normalise to [0, 1] -----------------------------------------
        bm25_norm = self._minmax(bm25_raw)
        dense_norm = self._minmax(dense_raw)

        # --- Interpolate -------------------------------------------------
        fused_scores = alpha * dense_norm + (1.0 - alpha) * bm25_norm

        return {idx: float(score) for idx, score in zip(candidate_list, fused_scores)}

    def _fuse_rrf(
        self,
        bm25_scores_all: np.ndarray,
        faiss_indices: np.ndarray,
        exclude: set[int],
        initial_k: int,
        rrf_k: int,
    ) -> dict[int, float]:
        """Reciprocal Rank Fusion: Σ 1/(k + rank)."""
        bm25_top = np.argsort(bm25_scores_all)[::-1][:initial_k]

        rrf_scores: dict[int, float] = {}

        # BM25 contribution
        for rank, idx in enumerate(bm25_top, start=1):
            idx = int(idx)
            if idx in exclude:
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)

        # Dense contribution
        for rank, offset in enumerate(faiss_indices, start=1):
            if offset == -1:
                continue
            idx = self.dense_searcher.id_map[int(offset)]
            if idx in exclude:
                continue
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)

        return rrf_scores

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------

    def _make_result(self, idx: int, *, score: float, results_so_far: list) -> dict:
        """Build a single result dict with title lookup and auto-ranking."""
        return {
            'title': self.df.iloc[idx]['title'],
            'score': score,
            'rank': len(results_so_far) + 1,
            'method': 'hybrid',
        }
