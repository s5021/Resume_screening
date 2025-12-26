"""
Advanced Feature Engineering for Resume Screening
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import Counter
class FeatureEngineer:
    def __init__(self):
        self.data_path = Path("data/processed/parsed_resumes.json")
    def load_data(self):
        """Load parsed resume data"""
        with open(self.data_path, "r", encoding="utf-8") as f:
            resumes = json.load(f)
        return pd.DataFrame(resumes)
    def extract_years_experience(self, text):
        """Extract years of experience from text"""
        if not text:
            return 0
        text_lower = text.lower()
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience',
            r'experience.*?(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*in',
        ]
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            years.extend([int(m) for m in matches])
        return max(years) if years else 0
    def extract_education_level(self, text):
        """Extract highest education level"""
        if not text:
            return 'Unknown'
        text_lower = text.lower()
        if any(word in text_lower for word in ['phd', 'doctorate', 'ph.d', 'doctoral']):
            return 'PhD'
        elif any(word in text_lower for word in ['master', 'm.sc', 'm.tech', 'mba', 'm.a', 'ms', 'msc']):
            return 'Masters'
        elif any(word in text_lower for word in ['bachelor', 'b.sc', 'b.tech', 'b.a', 'b.e', 'bs', 'bsc']):
            return 'Bachelors'
        elif any(word in text_lower for word in ['diploma', 'associate']):
            return 'Diploma'
        else:
            return 'Unknown'
    def categorize_skills(self, skills_list):
        """Categorize skills into different types"""
        if not skills_list:
            return {
                'programming': 0,
                'ml_ai': 0,
                'cloud': 0,
                'database': 0,
                'soft': 0
            }
        categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'react', 'node', 'django', 'flask'],
            'ml_ai': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'nlp', 'data science'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes'],
            'database': ['sql', 'mongodb', 'postgresql', 'mysql', 'nosql'],
            'soft': ['leadership', 'communication', 'project management', 'problem solving']
        }
        skill_counts = {cat: 0 for cat in categories}
        for skill in skills_list:
            for category, keywords in categories.items():
                if skill.lower() in keywords:
                    skill_counts[category] += 1
        return skill_counts
    def extract_certifications(self, text):
        """Check for common certifications"""
        if not text:
            return 0
        text_lower = text.lower()
        cert_keywords = [
            'certified', 'certification', 'certificate',
            'aws certified', 'pmp', 'scrum master', 'cissp',
            'comptia', 'microsoft certified', 'google certified'
        ]
        count = sum(1 for keyword in cert_keywords if keyword in text_lower)
        return min(count, 5)
    def create_text_features(self, text):
        """Create additional text-based features"""
        if not text:
            return {
                'avg_word_length': 0,
                'sentence_count': 0,
                'unique_words': 0
            }
        words = text.split()
        sentences = text.split('.')
        return {
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'sentence_count': len(sentences),
            'unique_words': len(set(words))
        }
    def create_all_features(self):
        """Create all engineered features"""
        print("Loading data...")
        df = self.load_data()
        print("Creating features...")
        # FIRST: Create skill_count if it doesn't exist
        if 'skill_count' not in df.columns:
            print("  - Calculating skill count...")
            df['skill_count'] = df['skills'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        # Experience
        print("  - Extracting years of experience...")
        df['years_experience'] = df['cleaned_text'].apply(self.extract_years_experience)
        # Education
        print("  - Extracting education level...")
        df['education_level'] = df['cleaned_text'].apply(self.extract_education_level)
        # Skill categories
        print("  - Categorizing skills...")
        skill_categories = df['skills'].apply(self.categorize_skills)
        df['programming_skills'] = skill_categories.apply(lambda x: x['programming'])
        df['ml_ai_skills'] = skill_categories.apply(lambda x: x['ml_ai'])
        df['cloud_skills'] = skill_categories.apply(lambda x: x['cloud'])
        df['database_skills'] = skill_categories.apply(lambda x: x['database'])
        df['soft_skills'] = skill_categories.apply(lambda x: x['soft'])
        # Certifications
        print("  - Extracting certifications...")
        df['certification_count'] = df['cleaned_text'].apply(self.extract_certifications)
        # Contact info
        df['has_email'] = df['email'].notna().astype(int)
        df['has_phone'] = df['phone'].notna().astype(int)
        df['contact_completeness'] = df['has_email'] + df['has_phone']
        # Text features
        print("  - Creating text features...")
        text_features = df['cleaned_text'].apply(self.create_text_features)
        df['avg_word_length'] = text_features.apply(lambda x: x['avg_word_length'])
        df['sentence_count'] = text_features.apply(lambda x: x['sentence_count'])
        df['unique_words'] = text_features.apply(lambda x: x['unique_words'])
        # Derived features (NOW skill_count exists!)
        df['skill_density'] = df['skill_count'] / (df['word_count'] + 1)
        df['experience_level'] = pd.cut(df['years_experience'], 
                                        bins=[-1, 0, 2, 5, 10, 100],
                                        labels=['Entry', 'Junior', 'Mid', 'Senior', 'Expert'])
        print("✓ All features created!")
        return df
    def save_features(self, df):
        """Save enhanced dataset"""
        output_csv = Path("data/processed/resumes_with_features.csv")
        feature_cols = [
            'id', 'filename', 'category', 'word_count', 'skill_count',
            'years_experience', 'education_level',
            'programming_skills', 'ml_ai_skills', 'cloud_skills', 'database_skills', 'soft_skills',
            'certification_count', 'has_email', 'has_phone', 'contact_completeness',
            'avg_word_length', 'sentence_count', 'unique_words', 'skill_density', 'experience_level',
            'cleaned_text', 'skills', 'email', 'phone'
        ]
        df[feature_cols].to_csv(output_csv, index=False)
        print(f"\n✓ Saved features to: {output_csv}")
        summary = {
            'total_resumes': int(len(df)),
            'experience_distribution': {str(k): int(v) for k, v in df['experience_level'].value_counts().to_dict().items()},
            'education_distribution': {str(k): int(v) for k, v in df['education_level'].value_counts().to_dict().items()},
            'avg_years_experience': float(df['years_experience'].mean()),
            'avg_skills_per_resume': float(df['skill_count'].mean()),
            'top_categories': {str(k): int(v) for k, v in df['category'].value_counts().head(10).to_dict().items()}
        }
        summary_path = Path("data/processed/feature_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary to: {summary_path}")
        return output_csv
def main():
    print("="*60)
    print("FEATURE ENGINEERING FOR RESUME SCREENING")
    print("="*60)
    fe = FeatureEngineer()
    df = fe.create_all_features()
    output_path = fe.save_features(df)
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    print(f"\nTotal resumes: {len(df)}")
    print(f"\n📊 Experience Distribution:")
    print(df['experience_level'].value_counts())
    print(f"\n🎓 Education Distribution:")
    print(df['education_level'].value_counts())
    print(f"\n💻 Skill Breakdown:")
    print(f"  Programming: {df['programming_skills'].sum()} total")
    print(f"  ML/AI: {df['ml_ai_skills'].sum()} total")
    print(f"  Cloud: {df['cloud_skills'].sum()} total")
    print(f"  Database: {df['database_skills'].sum()} total")
    print(f"\n📜 Certifications: {df['certification_count'].sum()} total")
    print(f"\n✓ Average years experience: {df['years_experience'].mean():.1f}")
    print(f"✓ Average skill density: {df['skill_density'].mean():.4f}")
    print("\n" + "="*60)
    print("SAMPLE RESUME WITH FEATURES")
    print("="*60)
    sample = df.iloc[0]
    print(f"Filename: {sample['filename']}")
    print(f"Category: {sample['category']}")
    print(f"Words: {sample['word_count']}")
    print(f"Years Experience: {sample['years_experience']}")
    print(f"Education: {sample['education_level']}")
    print(f"Experience Level: {sample['experience_level']}")
    print(f"Programming Skills: {sample['programming_skills']}")
    print(f"ML/AI Skills: {sample['ml_ai_skills']}")
    print(f"Certifications: {sample['certification_count']}")
    print("\n" + "="*60)
    print("✓ PHASE 1 COMPLETE!")
    print("="*60)
if __name__ == "__main__":
    main()
