#!/usr/bin/env python3
"""
Normalize SAFE_DRAFT_2.md and generate beautiful PDF using Pandoc/XeLaTeX
"""

import re
import sys
import os
import subprocess

# Define the workspace directory
WORKSPACE = "/Users/akashchatake/Downloads/Chatake-Innoworks-Organization/Projects/Publications/Machine-Learning-Textbook"

def normalize_text(text):
    """
    Normalize text by replacing smart quotes, emojis, and special characters
    """
    # Replace curly/smart quotes with straight ASCII quotes
    text = text.replace('"', '"').replace('"', '"')  # Double quotes
    text = text.replace(''', "'").replace(''', "'")  # Single quotes
    text = text.replace('…', '...')  # Ellipsis
    text = text.replace('—', '--').replace('–', '--')  # Em-dash and en-dash
    
    # Remove emojis (comprehensive Unicode emoji ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+"
        , flags=re.UNICODE)
    
    text = emoji_pattern.sub('', text)
    
    return text

def main():
    os.chdir(WORKSPACE)
    
    print("📖 Starting PDF generation process...")
    print("=" * 60)
    
    # Step 1: Read SAFE_DRAFT_2.md
    print("\n✓ Step 1: Reading SAFE_DRAFT_2.md...")
    with open('SAFE_DRAFT_2.md', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"  - Read {len(content)} characters")
    
    # Step 2: Normalize the content
    print("\n✓ Step 2: Normalizing content...")
    print("  - Replacing curly quotes with straight ASCII...")
    print("  - Removing emojis...")
    print("  - Fixing special characters...")
    
    normalized_content = normalize_text(content)
    
    # Step 3: Save as FINAL_TEXTBOOK.md
    print("\n✓ Step 3: Saving normalized content as FINAL_TEXTBOOK.md...")
    with open('FINAL_TEXTBOOK.md', 'w', encoding='utf-8') as f:
        f.write(normalized_content)
    
    print(f"  - Saved {len(normalized_content)} characters")
    
    # Step 4: Create build directory if it doesn't exist
    print("\n✓ Step 4: Creating build directory...")
    os.makedirs('build', exist_ok=True)
    print("  - build/ directory ready")
    
    # Step 5: Run Pandoc command
    print("\n✓ Step 5: Generating PDF with XeLaTeX...")
    print("  - This may take 1-2 minutes...")
    
    pandoc_cmd = [
        'pandoc',
        'FINAL_TEXTBOOK.md',
        '-o', 'build/Machine_Learning_Textbook_Beautiful.pdf',
        '--from', 'markdown+tex_math_dollars+fenced_code_attributes',
        '--pdf-engine=xelatex',
        '-V', 'mainfont=TeX Gyre Pagella',
        '-V', 'monofont=Fira Code',
        '-V', 'sansfont=TeX Gyre Heros',
        '-V', 'fontsize=11pt',
        '-V', 'geometry:top=1in,bottom=1in,left=1in,right=1in',
        '-V', 'colorlinks=true',
        '-V', 'linkcolor=MidnightBlue',
        '-V', 'urlcolor=RoyalBlue',
        '-V', 'toccolor=Gray',
        '-V', 'header-includes=\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhead[CO,CE]{Machine Learning: Foundations & Futures}\\fancyfoot[CO,CE]{Akash Chatake — MindforgeAI Press}',
        '--toc',
        '--toc-depth=3',
        '--number-sections',
        '--syntax-highlighting=pygments',
        '--metadata', 'title=Machine Learning: Foundations & Futures',
        '--metadata', 'subtitle=A Comprehensive Guide to Artificial Intelligence and Data Science',
        '--metadata', 'author=Akash Chatake (MindforgeAI / Chatake Innoworks Pvt. Ltd.)',
        '--metadata', 'date=November 2025',
        '--metadata', 'rights=© 2025 Chatake Innoworks Organization. All rights reserved.',
    ]
    
    try:
        result = subprocess.run(pandoc_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("  - PDF generated successfully!")
        else:
            print(f"  - Pandoc exited with code {result.returncode}")
            if result.stderr:
                print("  - Errors/Warnings:")
                for line in result.stderr.split('\n')[:10]:
                    if line.strip():
                        print(f"    {line}")
    
    except subprocess.TimeoutExpired:
        print("  - ERROR: Pandoc command timed out")
        sys.exit(1)
    except FileNotFoundError:
        print("  - ERROR: Pandoc not found. Please install Pandoc and try again.")
        sys.exit(1)
    
    # Step 6: Verify output
    print("\n✓ Step 6: Verifying output...")
    pdf_path = 'build/Machine_Learning_Textbook_Beautiful.pdf'
    
    if os.path.exists(pdf_path):
        size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"  - PDF created successfully!")
        print(f"  - File size: {size_mb:.2f} MB")
        print(f"  - Location: {os.path.abspath(pdf_path)}")
    else:
        print(f"  - ERROR: PDF file not found at {pdf_path}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Beautifully formatted MindforgeAI Founder's Edition PDF")
    print("   generated successfully!")
    print("✅ Machine_Learning_Textbook_Beautiful.pdf is ready for review.")
    print("=" * 60)

if __name__ == '__main__':
    main()
