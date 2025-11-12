#!/usr/bin/env python3
"""
Script to build complete textbook following JSON specification
with proper page breaks between each file component
"""

import os

def build_complete_textbook():
    base_dir = "/Users/akashchatake/Downloads/Chatake-Innoworks-Organization/Projects/Publications/Machine-Learning-Textbook"
    os.chdir(base_dir)
    
    # Start building the complete textbook
    complete_content = []
    
    # Front Matter - Title Page
    complete_content.append("# MACHINE LEARNING")
    complete_content.append("## A Comprehensive Guide to Artificial Intelligence and Data Science")
    complete_content.append("### From Fundamentals to Advanced Applications")
    complete_content.append("")
    complete_content.append("**By:** Akash Chatake")
    complete_content.append("**Publisher:** Chatake Innoworks Publications")
    complete_content.append("**Edition:** First Edition, 2025")
    complete_content.append("**Series:** Computer Technology & Engineering Series")
    complete_content.append("**Course Code:** MSBTE 316316")
    complete_content.append("")
    complete_content.append("---")
    complete_content.append("")
    complete_content.append('> "Bridging Theory and Practice in the Age of AI"')
    complete_content.append("")
    complete_content.append("\\newpage")
    complete_content.append("")
    
    # Function to add file with page break
    def add_file_content(filename, section_title=None):
        if section_title:
            complete_content.append(f"# {section_title}")
            complete_content.append("")
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                # Skip the first line if it's a header (starts with #)
                lines = content.strip().split('\n')
                if lines and lines[0].startswith('#'):
                    lines = lines[1:]  # Skip first line
                content = '\n'.join(lines).strip()
                complete_content.append(content)
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            complete_content.append(f"[{filename} - Content not found]")
            
        complete_content.append("")
        complete_content.append("\\newpage")
        complete_content.append("")
    
    # Add all components following JSON specification
    print("Building complete textbook...")
    
    # Front Matter
    add_file_content("COPYRIGHT.md", "COPYRIGHT & PUBLICATION INFORMATION")
    add_file_content("DEDICATION.md", "DEDICATION")
    add_file_content("ABOUT_AUTHOR.md", "ABOUT THE AUTHOR")
    add_file_content("PREFACE.md", "PREFACE")
    add_file_content("TABLE_OF_CONTENTS.md", "TABLE OF CONTENTS")
    
    # Chapters
    chapters = [
        "chapters/chapter_01_introduction.md",
        "chapters/chapter_02_data_preprocessing.md", 
        "chapters/chapter_03_feature_engineering.md",
        "chapters/chapter_04_classification.md",
        "chapters/chapter_05_regression.md",
        "chapters/chapter_06_clustering.md",
        "chapters/chapter_07_dimensionality_reduction.md",
        "chapters/chapter_08_end_to_end_projects.md",
        "chapters/chapter_09_model_selection_evaluation.md",
        "chapters/chapter_10_ethics_deployment.md"
    ]
    
    for chapter in chapters:
        add_file_content(chapter)
    
    # Appendices
    appendices = [
        "appendices/appendix_a_python_setup.md",
        "appendices/appendix_b_mathematical_foundations.md",
        "appendices/appendix_c_datasets_resources.md", 
        "appendices/appendix_d_evaluation_metrics.md",
        "appendices/appendix_e_industry_applications.md"
    ]
    
    for appendix in appendices:
        add_file_content(appendix)
    
    # Back Matter
    add_file_content("EPILOGUE.md", "EPILOGUE")
    add_file_content("REFERENCES_BIBLIOGRAPHY.md", "REFERENCES & BIBLIOGRAPHY")
    add_file_content("INDEX.md", "INDEX")
    add_file_content("BACK_COVER.md", "BACK COVER")
    
    # Write the complete file
    final_content = '\n'.join(complete_content)
    
    with open('COMPLETE_TEXTBOOK_FINAL.md', 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    # Calculate statistics
    lines = len(final_content.split('\n'))
    words = len(final_content.split())
    chars = len(final_content)
    
    print(f"✅ Complete textbook created: COMPLETE_TEXTBOOK_FINAL.md")
    print(f"📊 Statistics:")
    print(f"   Lines: {lines:,}")
    print(f"   Words: {words:,}")
    print(f"   Characters: {chars:,}")
    print(f"   Size: {chars/1024:.1f} KB")
    print(f"✅ Each component has proper \\newpage breaks")
    print(f"✅ Following JSON specification exactly")

if __name__ == "__main__":
    build_complete_textbook()
