"""
Test if setup is complete
"""

def test_imports():
    print("Testing imports...")
    print("=" * 60)
    
    # Test pandas
    try:
        import pandas
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas - {e}")
    
    # Test transformers
    try:
        import transformers
        print("✓ transformers")
    except ImportError as e:
        print(f"✗ transformers - {e}")
    
    # Test PyPDF2
    try:
        import PyPDF2
        print("✓ PyPDF2")
    except ImportError as e:
        print(f"✗ PyPDF2 - {e}")
    
    # Test python-docx (import name is 'docx')
    try:
        import docx
        print("✓ python-docx")
    except ImportError as e:
        print(f"✗ python-docx - {e}")
    
    print("=" * 60)


def test_structure():
    print("\nTesting project structure...")
    from pathlib import Path
    dirs = ['data/raw', 'data/processed', 'src/data_processing']
    for d in dirs:
        if Path(d).exists():
            print(f"✓ {d}")
        else:
            print(f"✗ {d}")


def main():
    test_imports()
    test_structure()
    print("\n✓ Setup verification complete!")


if __name__ == "__main__":
    main()