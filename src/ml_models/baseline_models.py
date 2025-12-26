"""
Baseline Models for Resume-Job Matching
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple
class ResumeJobMatcher:
    def __init__(self):
        self.data_path = Path("data/processed/resumes_with_features.csv")
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
    def load_data(self):
        """Load feature-engineered resumes"""
        print("Loading resume data...")
        self.df = pd.read_csv(self.data_path)
        # Parse skills column (it saved as string)
        self.df['skills'] = self.df['skills'].apply(
            lambda x: eval(x) if isinstance(x, str) else []
        )
        print(f"✓ Loaded {len(self.df)} resumes")
        return self.df
    def create_tfidf_matrix(self):
        """Create TF-IDF matrix for resume texts"""
        print("\nCreating TF-IDF matrix...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.df['cleaned_text'].fillna('')
        )
        print(f"✓ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        return self.tfidf_matrix
    def method1_tfidf_cosine(self, job_description: str, top_k: int = 10) -> List[Dict]:
        """Method 1: TF-IDF + Cosine Similarity"""
        print("\n" + "="*60)
        print("METHOD 1: TF-IDF + Cosine Similarity")
        print("="*60)
        job_tfidf = self.tfidf_vectorizer.transform([job_description])
        similarities = cosine_similarity(job_tfidf, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            resume = self.df.iloc[idx]
            results.append({
                'rank': int(rank),
                'filename': str(resume['filename']),
                'category': str(resume['category']),
                'score': float(similarities[idx]),
                'method': 'TF-IDF',
                'years_exp': int(resume['years_experience']),
                'education': str(resume['education_level']),
                'skills': [str(s) for s in resume['skills']]
            })
        return results
    def method2_skill_overlap(self, required_skills: List[str], top_k: int = 10) -> List[Dict]:
        """Method 2: Skill Overlap (Jaccard Similarity)"""
        print("\n" + "="*60)
        print("METHOD 2: Skill Overlap (Jaccard)")
        print("="*60)
        required_skills_set = set(s.lower() for s in required_skills)
        def jaccard_similarity(resume_skills):
            if not resume_skills:
                return 0.0
            resume_skills_set = set(s.lower() for s in resume_skills)
            intersection = len(required_skills_set & resume_skills_set)
            union = len(required_skills_set | resume_skills_set)
            return intersection / union if union > 0 else 0.0
        self.df['skill_score'] = self.df['skills'].apply(jaccard_similarity)
        top_resumes = self.df.nlargest(top_k, 'skill_score')
        results = []
        for rank, (idx, resume) in enumerate(top_resumes.iterrows(), 1):
            matched_skills = set(s.lower() for s in resume['skills']) & required_skills_set
            results.append({
                'rank': int(rank),
                'filename': str(resume['filename']),
                'category': str(resume['category']),
                'score': float(resume['skill_score']),
                'method': 'Skill Overlap',
                'matched_skills': [str(s) for s in matched_skills],
                'years_exp': int(resume['years_experience']),
                'education': str(resume['education_level'])
            })
        return results
    def method3_weighted_scoring(
        self, 
        job_description: str,
        required_skills: List[str],
        min_experience: int = 0,
        preferred_education: str = None,
        top_k: int = 10
    ) -> List[Dict]:
        """Method 3: Weighted Multi-Factor Scoring"""
        print("\n" + "="*60)
        print("METHOD 3: Weighted Multi-Factor Scoring")
        print("="*60)
        # Text similarity
        job_tfidf = self.tfidf_vectorizer.transform([job_description])
        text_scores = cosine_similarity(job_tfidf, self.tfidf_matrix).flatten()
        # Skill overlap
        required_skills_set = set(s.lower() for s in required_skills)
        def skill_match_score(resume_skills):
            if not resume_skills or not required_skills_set:
                return 0.0
            resume_skills_set = set(s.lower() for s in resume_skills)
            matched = len(required_skills_set & resume_skills_set)
            return matched / len(required_skills_set)
        skill_scores = self.df['skills'].apply(skill_match_score)
        # Experience
        def experience_score(years):
            if years < min_experience:
                return 0.0
            elif years == min_experience:
                return 1.0
            else:
                return min(1.0 + (years - min_experience) * 0.1, 1.5)
        exp_scores = self.df['years_experience'].apply(experience_score)
        # Education
        education_levels = {'PhD': 4, 'Masters': 3, 'Bachelors': 2, 'Diploma': 1, 'Unknown': 0}
        target_level = education_levels.get(preferred_education, 0) if preferred_education else 0
        def education_score(edu):
            edu_level = education_levels.get(edu, 0)
            if edu_level >= target_level:
                return 1.0
            else:
                return edu_level / target_level if target_level > 0 else 0.5
        edu_scores = self.df['education_level'].apply(education_score)
        # Combine
        weights = {'text': 0.40, 'skills': 0.40, 'experience': 0.15, 'education': 0.05}
        final_scores = (
            weights['text'] * text_scores +
            weights['skills'] * skill_scores +
            weights['experience'] * exp_scores +
            weights['education'] * edu_scores
        )
        top_indices = final_scores.argsort()[-top_k:][::-1]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            resume = self.df.iloc[idx]
            matched_skills = set(s.lower() for s in resume['skills']) & required_skills_set
            results.append({
                'rank': int(rank),
                'filename': str(resume['filename']),
                'category': str(resume['category']),
                'final_score': float(final_scores[idx]),
                'text_score': float(text_scores[idx]),
                'skill_score': float(skill_scores[idx]),
                'exp_score': float(exp_scores[idx]),
                'edu_score': float(edu_scores[idx]),
                'method': 'Weighted',
                'matched_skills': [str(s) for s in matched_skills],
                'years_exp': int(resume['years_experience']),
                'education': str(resume['education_level'])
            })
        return results
def demo_job_matching():
    """Demo: Match resumes to a sample job"""
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
    print("RESUME-JOB MATCHING DEMO")
    print("="*60)
    print("\nJob Description:")
    print(job_description)
    print("\nRequired Skills:", ', '.join(required_skills))
    matcher = ResumeJobMatcher()
    matcher.load_data()
    matcher.create_tfidf_matrix()
    results1 = matcher.method1_tfidf_cosine(job_description, top_k=5)
    print("\nTop 5 Candidates (TF-IDF Method):")
    for r in results1:
        print(f"  {r['rank']}. {r['filename']} - Score: {r['score']:.3f} - {r['category']}")
    results2 = matcher.method2_skill_overlap(required_skills, top_k=5)
    print("\nTop 5 Candidates (Skill Overlap Method):")
    for r in results2:
        print(f"  {r['rank']}. {r['filename']} - Score: {r['score']:.3f} - Matched: {len(r['matched_skills'])} skills")
    results3 = matcher.method3_weighted_scoring(
        job_description, required_skills, min_experience=3, preferred_education='Masters', top_k=5
    )
    print("\nTop 5 Candidates (Weighted Method):")
    for r in results3:
        print(f"  {r['rank']}. {r['filename']} - Final: {r['final_score']:.3f}")
    # Save results (all types converted to Python native)
    output_path = Path("data/processed/baseline_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            'job_description': job_description,
            'required_skills': required_skills,
            'method1_tfidf': results1,
            'method2_skill_overlap': results2,
            'method3_weighted': results3
        }, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")
    print("\n" + "="*60)
    print("✓ PHASE 2 COMPLETE!")
    print("="*60)
if __name__ == "__main__":
    demo_job_matching()
