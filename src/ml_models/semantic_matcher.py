"""
Simplified BERT Matcher using sklearn only
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
class SimplifiedMatcher:
    def __init__(self):
        self.data_path = Path("data/processed/resumes_with_features.csv")
        self.df = None
        self.vectorizer = None
        self.embeddings = None
    def load_data(self):
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        self.df['skills'] = self.df['skills'].apply(lambda x: eval(x) if isinstance(x, str) else [])
        print(f"✓ Loaded {len(self.df)} resumes")
    def create_advanced_tfidf(self):
        """Advanced TF-IDF with better parameters"""
        print("\nCreating advanced TF-IDF embeddings...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),  # Trigrams for better context
            stop_words='english',
            min_df=2,
            max_df=0.8,
            sublinear_tf=True  # Better for long documents
        )
        self.embeddings = self.vectorizer.fit_transform(self.df['cleaned_text'].fillna(''))
        print(f"✓ Created embeddings: {self.embeddings.shape}")
    def semantic_search(self, job_description: str, top_k: int = 10):
        print("\n" + "="*60)
        print("ADVANCED SEMANTIC MATCHING")
        print("="*60)
        job_vec = self.vectorizer.transform([job_description])
        scores = cosine_similarity(job_vec, self.embeddings).flatten()
        top_idx = scores.argsort()[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_idx, 1):
            r = self.df.iloc[idx]
            results.append({
                'rank': int(rank),
                'filename': str(r['filename']),
                'category': str(r['category']),
                'score': float(scores[idx]),
                'years_exp': int(r['years_experience']),
                'education': str(r['education_level'])
            })
        return results
def demo():
    job_desc = """
    Senior Machine Learning Engineer with Python, TensorFlow, PyTorch, AWS.
    Strong problem-solving and deep learning experience required.
    """
    print("="*60)
    print("SEMANTIC MATCHING (Advanced TF-IDF)")
    print("="*60)
    matcher = SimplifiedMatcher()
    matcher.load_data()
    matcher.create_advanced_tfidf()
    results = matcher.semantic_search(job_desc, top_k=10)
    print("\nTop 10 Candidates:")
    for r in results:
        print(f"{r['rank']:2d}. {r['filename']:20s} | Score: {r['score']:.3f} | {r['category']}")
    # Save
    with open("data/processed/semantic_results.json", 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print("\n✓ PHASE 3 COMPLETE (Simplified version)")
    print("✓ ALL PHASES DONE!")
if __name__ == "__main__":
    demo()
