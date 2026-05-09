import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Download punkt tokenizer data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class BM25Searcher:
    def __init__(self, df, text_column='content_narrative'):
        self.df = df
        self.text_column = text_column
        self.corpus = self.df[self.text_column].fillna('').tolist()
        
        # Tokenize the corpus
        print("Tokenizing corpus for BM25...")
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 Index built successfully.")

    def _tokenize(self, text):
        return word_tokenize(text.lower())

    def search(self, query, top_k=5):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                'title': self.df.iloc[idx]['title'],
                'score': scores[idx],
                'rank': rank + 1,
                'method': 'bm25'
            })
        return results


class DenseSearcher:
    def __init__(self, df, model_name='paraphrase-multilingual-MiniLM-L12-v2', embedding_col='embedding'):
        self.df = df
        self.embedding_col = embedding_col
        print(f"Loading SentenceTransformer model ({model_name})...")
        self.model = SentenceTransformer(model_name)
        
        # Extract embeddings as a numpy array for fast cosine similarity
        self.doc_embeddings = np.stack(self.df[self.embedding_col].values)
        print("Dense embeddings loaded successfully.")

    def search(self, query, top_k=5):
        # Embed the query
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                'title': self.df.iloc[idx]['title'],
                'score': similarities[idx],
                'rank': rank + 1,
                'method': 'dense'
            })
        return results


class HybridSearcher:
    def __init__(self, bm25_searcher, dense_searcher):
        self.bm25_searcher = bm25_searcher
        self.dense_searcher = dense_searcher

    def search(self, query, top_k=5, rrf_k=60):
        # To do a good RRF fusion, we need to retrieve more items initially
        initial_k = max(top_k * 2, 50)
        
        bm25_results = self.bm25_searcher.search(query, top_k=initial_k)
        dense_results = self.dense_searcher.search(query, top_k=initial_k)
        
        # RRF Scoring
        rrf_scores = {}
        
        for res in bm25_results:
            title = res['title']
            if title not in rrf_scores:
                rrf_scores[title] = 0.0
            rrf_scores[title] += 1.0 / (rrf_k + res['rank'])
            
        for res in dense_results:
            title = res['title']
            if title not in rrf_scores:
                rrf_scores[title] = 0.0
            rrf_scores[title] += 1.0 / (rrf_k + res['rank'])
            
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        
        results = []
        for rank, (title, score) in enumerate(sorted_results):
            results.append({
                'title': title,
                'score': score,
                'rank': rank + 1,
                'method': 'hybrid'
            })
            
        return results
