"""
Resume Parser - Fixed to search subfolders
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document
import re
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
class ResumeParser:
    def __init__(self):
        self.input_folder = RAW_DATA_DIR
        self.output_folder = PROCESSED_DATA_DIR
        self.output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Input folder: {self.input_folder}")
        logger.info(f"Output folder: {self.output_folder}")
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        try:
            reader = PdfReader(str(pdf_path))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path.name}: {e}")
            return ""
    def extract_text_from_docx(self, docx_path: Path) -> str:
        try:
            doc = Document(str(docx_path))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading DOCX {docx_path.name}: {e}")
            return ""
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    def extract_email(self, text: str) -> Optional[str]:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else None
    def extract_phone(self, text: str) -> Optional[str]:
        phone_patterns = [
            r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}',
            r'\b\d{10}\b',
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                return phones[0]
        return None
    def extract_skills(self, text: str) -> List[str]:
        skill_keywords = [
            'python', 'java', 'javascript', 'sql', 'machine learning',
            'deep learning', 'tensorflow', 'pytorch', 'aws', 'docker',
            'react', 'node', 'excel', 'data analysis'
        ]
        text_lower = text.lower()
        found_skills = [skill for skill in skill_keywords if skill in text_lower]
        return list(set(found_skills))
    def parse_resume(self, file_path: Path) -> Optional[Dict]:
        if file_path.suffix.lower() == '.pdf':
            raw_text = self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            raw_text = self.extract_text_from_docx(file_path)
        else:
            return None
        if not raw_text or len(raw_text) < 50:
            return None
        cleaned_text = self.clean_text(raw_text)
        parsed_data = {
            'id': file_path.stem,
            'filename': file_path.name,
            'category': file_path.parent.name,
            'parsed_at': datetime.now().isoformat(),
            'cleaned_text': cleaned_text,
            'email': self.extract_email(raw_text),
            'phone': self.extract_phone(raw_text),
            'skills': self.extract_skills(cleaned_text),
            'text_length': len(cleaned_text),
            'word_count': len(cleaned_text.split()),
        }
        return parsed_data
    def parse_all_resumes(self, limit: int = None) -> List[Dict]:
        logger.info("=" * 60)
        logger.info("Starting resume parsing...")
        logger.info("=" * 60)
        resume_files = []
        for ext in ['*.pdf', '*.docx', '*.doc']:
            found = list(self.input_folder.rglob(ext))
            resume_files.extend(found)
            logger.info(f"Found {len(found)} {ext} files")
        if not resume_files:
            logger.warning(f"No files found in {self.input_folder}")
            return []
        if limit:
            logger.info(f"Limiting to first {limit} resumes")
            resume_files = resume_files[:limit]
        logger.info(f"Total to process: {len(resume_files)}")
        all_resumes = []
        failed = 0
        for i, file_path in enumerate(resume_files, 1):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(resume_files)}")
            parsed_data = self.parse_resume(file_path)
            if parsed_data:
                all_resumes.append(parsed_data)
            else:
                failed += 1
        logger.info(f"Successfully parsed: {len(all_resumes)}")
        logger.info(f"Failed: {failed}")
        return all_resumes
    def save_results(self, parsed_resumes: List[Dict]):
        if not parsed_resumes:
            return
        output_file = self.output_folder / 'parsed_resumes.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_resumes, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved to: {output_file}")
def main():
    parser = ResumeParser()
    parsed_resumes = parser.parse_all_resumes()
    if parsed_resumes:
        parser.save_results(parsed_resumes)
        print(f"\n✓ Processed {len(parsed_resumes)} resumes successfully!")
    else:
        print("\n⚠ No resumes found!")
if __name__ == "__main__":
    main()
