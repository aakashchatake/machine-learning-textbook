#!/usr/bin/env python3
"""
Extract text content from the Machine Learning syllabus PDF
"""

import PyPDF2
import os

def extract_pdf_text(pdf_path, output_path):
    """Extract text from PDF and save to text file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            text_content = []
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text_content.append(f"--- Page {page_num + 1} ---\n")
                text_content.append(page.extract_text())
                text_content.append("\n\n")
            
            # Write extracted text to file
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(''.join(text_content))
            
            print(f"Successfully extracted text from {pdf_path}")
            print(f"Output saved to: {output_path}")
            
            # Also print the content to console
            print("\n" + "="*50)
            print("SYLLABUS CONTENT:")
            print("="*50)
            print(''.join(text_content))
            
    except Exception as e:
        print(f"Error extracting PDF: {e}")

if __name__ == "__main__":
    pdf_file = "316316-MACHINE LEARNING.pdf"
    output_file = "syllabus_content.txt"
    
    extract_pdf_text(pdf_file, output_file)
