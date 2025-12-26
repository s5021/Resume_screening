"""
BERT-based Semantic Resume-Job Matching
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict
import time
class BERTMatcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize BERT matcher
        Args:
            model_name: SentenceTransformer model name
                       'all-MiniLM-L6-v2' - Fast, 384 dimensions (RECOMMENDED)
                       'all-mpnet-base-v2' - More accurate, 768 dimensions (slower)
        """
        self.model_name = model_name
        self.model = None
        self.data_path = Path("data/processed/resumes_with_features.csv")
        self.embeddings_path = Path("data/embeddings")
        self.embeddings_path.mkdir(exist_ok=True)
        self.df = None
        self.resume_embeddings = None
    def load_model(self):
        """Load pre-trained BERT model"""
        print(f"\nLoading BERT model: {self.model_name}...")
        print("(First time will download ~80MB model)")
        start_time = time.time()
        self.model = SentenceTransformer(self.model_name)
        print(f"✓ Model loaded in {time.time() - start_time:.1f} seconds")
        print(f"✓ Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        return self.model
    def load_data(self):
        """Load resume data"""
        print("\nLoading resume data...")
        self.df = pd.read_csv(self.data_path)
        # Parse skills
        self.df['skills'] = self.df['skills'].apply(
            lambda x: eval(x) if isinstance(x, str) else []
        )
        print(f"✓ Loaded {len(self.df)} resumes")
        return self.df
    def create_resume_embeddings(self, force_recreate: bool = False):
        """
        Create embeddings for all resumes
        This is done once and cached for reuse
        """
        embeddings_file = self.embeddings_path / f"resume_embeddings_{self.model_name}.npy"
        # Check if embeddings already exist
        if embeddings_file.exists() and not force_recreate:
            print(f"\nLoading cached embeddings from {embeddings_file}...")
            self.resume_embeddings = np.load(embeddings_file)
            print(f"✓ Loaded embeddings shape: {self.resume_embeddings.shape}")
            return self.resume_embeddings
        # Create new embeddings
        print(f"\nCreating embeddings for {len(self.df)} resumes...")
        print("This will take 2-5 minutes...")
        start_time = time.time()
        # Prepare texts (use cleaned_text)
        texts = self.df['cleaned_text'].fillna('').tolist()
        # Encode in batches for efficiency
        self.resume_embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        # Save embeddings for future use
        np.save(embeddings_file, self.resume_embeddings)
        elapsed = time.time() - start_time
        print(f"\n✓ Created embeddings in {elapsed:.1f} seconds")
        print(f"✓ Embeddings shape: {self.resume_embeddings.shape}")
        print(f"✓ Saved to: {embeddings_file}")
        return self.resume_embeddings
    def semantic_search(
        self,
        job_description: str,
        top_k: int = 10,
        category_filter: str = None
    ) -> List[Dict]:
        """
        Semantic search using BERT embeddings
        Args:
            job_description: Job description text
            top_k: Number of top candidates to return
            category_filter: Optional category to filter (e.g., 'ENGINEERING')
        Returns:
            List of top matching resumes with scores
        """
        print("\n" + "="*60)
        print("BERT SEMANTIC MATCHING")
        print("="*60)
        # Encode job description
        print("Encoding job description...")
        job_embedding = self.model.encode(
            [job_description],
            convert_to_numpy=True
        )
        # Calculate semantic similarity
        print("Computing similarities...")
        similarities = cosine_similarity(job_embedding, self.resume_embeddings).flatten()
        # Apply category filter if specified
        if category_filter:
            mask = self.df['category'] == category_filter
            similarities = np.where(mask, similarities, -1)
            print(f"Filtered to category: {category_filter}")
        # Get top K
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if similarities[idx] < 0:  # Skip filtered out
                continue
            resume = self.df.iloc[idx]
            results.append({
                'rank': int(rank),
                'filename': str(resume['filename']),
                'category': str(resume['category']),
                'semantic_score': float(similarities[idx]),
                'years_exp': int(resume['years_experience']),
                'education': str(resume['education_level']),
                'skills': [str(s) for s in resume['skills']],
                'word_count': int(resume['word_count'])
            })
        return results
    def hybrid_search(
        self,
        job_description: str,
        required_skills: List[str],
        top_k: int = 10,
        semantic_weight: float = 0.7,
        skill_weight: float = 0.3
    ) -> List[Dict]:
        """
        Hybrid: Combine BERT semantic matching + skill overlap
        Args:
            semantic_weight: Weight for semantic similarity (default 0.7)
            skill_weight: Weight for skill matching (default 0.3)
        """
        print("\n" + "="*60)
        print("HYBRID MATCHING (BERT + Skills)")
        print("="*60)
        # 1. Semantic similarity
        job_embedding = self.model.encode([job_description], convert_to_numpy=True)
        semantic_scores = cosine_similarity(job_embedding, self.resume_embeddings).flatten()
        # 2. Skill overlap
        required_skills_set = set(s.lower() for s in required_skills)
        def skill_score(resume_skills):
            if not resume_skills or not required_skills_set:
                return 0.0
            resume_skills_set = set(s.lower() for s in resume_skills)
            matched = len(required_skills_set & resume_skills_set)
            return matched / len(required_skills_set)
        skill_scores = self.df['skills'].apply(skill_score).values
        # 3. Combine scores
        final_scores = (
            semantic_weight * semantic_scores +
            skill_weight * skill_scores
        )
        # Get top K
        top_indices = final_scores.argsort()[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            resume = self.df.iloc[idx]
            matched_skills = set(s.lower() for s in resume['skills']) & required_skills_set
            results.append({
                'rank': int(rank),
                'filename': str(resume['filename']),
                'category': str(resume['category']),
                'hybrid_score': float(final_scores[idx]),
                'semantic_score': float(semantic_scores[idx]),
                'skill_score': float(skill_scores[idx]),
                'matched_skills': [str(s) for s in matched_skills],
                'years_exp': int(resume['years_experience']),
                'education': str(resume['education_level'])
            })
        return results
def demo_bert_matching():
    """Demo: BERT-based matching"""
    job_description = """
    We are seeking a Senior Machine Learning Engineer with strong Python skills.
    The ideal candidate will have experience with deep learning frameworks like 
    TensorFlow or PyTorch, cloud platforms (AWS), and excellent problem-solving abilities.
    You will work on cutting-edge AI projects and collaborate with cross-functional teams.
    """
    required_skills = [
        'python', 'machine learning', 'deep learning',
        'tensorflow', 'pytorch', 'aws', 'problem solving'
    ]
    print("="*60)
    print("BERT-BASED SEMANTIC MATCHING DEMO")
    print("="*60)
    print("\nJob Description:")
    print(job_description)
    # Initialize
    matcher = BERTMatcher(model_name='all-MiniLM-L6-v2')
    matcher.load_model()
    matcher.load_data()
    # Create/load embeddings
    matcher.create_resume_embeddings()
    # Pure semantic search
    results_semantic = matcher.semantic_search(job_description, top_k=10)
    print("\n" + "="*60)
    print("TOP 10 CANDIDATES (Pure Semantic Matching)")
    print("="*60)
    for r in results_semantic:
        print(f"{r['rank']:2d}. {r['filename']:20s} | Score: {r['semantic_score']:.3f} | "
              f"{r['category']:20s} | {r['years_exp']} yrs")
    # Hybrid search
    results_hybrid = matcher.hybrid_search(
        job_description,
        required_skills,
        top_k=10,
        semantic_weight=0.7,
        skill_weight=0.3
    )
    print("\n" + "="*60)
    print("TOP 10 CANDIDATES (Hybrid: 70% Semantic + 30% Skills)")
    print("="*60)
    for r in results_hybrid:
        print(f"{r['rank']:2d}. {r['filename']:20s} | Final: {r['hybrid_score']:.3f} | "
              f"Semantic: {r['semantic_score']:.3f} | Skills: {r['skill_score']:.3f}")
    # Save results
    output_path = Path("data/processed/bert_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            'job_description': job_description,
            'required_skills': required_skills,
            'model_used': matcher.model_name,
            'semantic_results': results_semantic,
            'hybrid_results': results_hybrid
        }, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")
    # Performance summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print("\n📊 Semantic vs Baseline Methods:")
    print("  • Baseline (TF-IDF): Keyword matching, fast but limited")
    print("  • BERT Semantic: Understands context and meaning")
    print("  • Hybrid: Best of both - meaning + hard skills")
    print("\n✨ Key Advantages of BERT:")
    print("  ✓ Understands synonyms ('ML Engineer' = 'Machine Learning Engineer')")
    print("  ✓ Captures context ('experienced in Python' vs 'learning Python')")
    print("  ✓ Better matching for similar but differently worded resumes")
    print("  ✓ More robust to spelling variations")
    print("\n" + "="*60)
    print("✓ PHASE 3 COMPLETE!")
    print("="*60)
    print("\n🎉 ALL 3 PHASES COMPLETED!")
    print("\nYou now have:")
    print("  ✓ Phase 1: Advanced feature engineering")
    print("  ✓ Phase 2: 3 baseline matching algorithms")
    print("  ✓ Phase 3: State-of-the-art BERT semantic matching")
    print("\n🚀 Ready for production deployment!")
if __name__ == "__main__":
    demo_bert_matching()
