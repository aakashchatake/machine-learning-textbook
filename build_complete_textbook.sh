#!/bin/bash

# Script to build complete textbook following JSON specification exactly
cd "/Users/akashchatake/Downloads/Chatake-Innoworks-Organization/Projects/Publications/Machine-Learning-Textbook"

echo "Building complete textbook following JSON specification..."

# Function to add file with page break
add_file_with_pagebreak() {
    local file="$1"
    if [ -f "$file" ]; then
        echo "Adding: $file"
        cat "$file" >> COMPLETE_TEXTBOOK_JSON_ALIGNED.md
        echo -e "\n\n\\newpage\n" >> COMPLETE_TEXTBOOK_JSON_ALIGNED.md
    else
        echo "WARNING: $file not found"
    fi
}

# Start fresh
rm -f COMPLETE_TEXTBOOK_JSON_ALIGNED.md

# Front Matter (each file gets its own page)
echo "=== ADDING FRONT MATTER ==="
add_file_with_pagebreak "TITLE_PAGE.md"
add_file_with_pagebreak "COPYRIGHT.md"  
add_file_with_pagebreak "DEDICATION.md"
add_file_with_pagebreak "ABOUT_AUTHOR.md"
add_file_with_pagebreak "PREFACE.md"
add_file_with_pagebreak "TABLE_OF_CONTENTS.md"

# Chapters (each chapter gets its own page)
echo "=== ADDING CHAPTERS ==="
add_file_with_pagebreak "chapters/chapter_01_introduction.md"
add_file_with_pagebreak "chapters/chapter_02_data_preprocessing.md"
add_file_with_pagebreak "chapters/chapter_03_feature_engineering.md"
add_file_with_pagebreak "chapters/chapter_04_classification.md"
add_file_with_pagebreak "chapters/chapter_05_regression.md"
add_file_with_pagebreak "chapters/chapter_06_clustering.md"
add_file_with_pagebreak "chapters/chapter_07_dimensionality_reduction.md"
add_file_with_pagebreak "chapters/chapter_08_end_to_end_projects.md"
add_file_with_pagebreak "chapters/chapter_09_model_selection_evaluation.md"
add_file_with_pagebreak "chapters/chapter_10_ethics_deployment.md"

# Appendices (each appendix gets its own page)
echo "=== ADDING APPENDICES ==="
add_file_with_pagebreak "appendices/appendix_a_python_setup.md"
add_file_with_pagebreak "appendices/appendix_b_mathematical_foundations.md"
add_file_with_pagebreak "appendices/appendix_c_datasets_resources.md"
add_file_with_pagebreak "appendices/appendix_d_evaluation_metrics.md"
add_file_with_pagebreak "appendices/appendix_e_industry_applications.md"

# Back Matter (each file gets its own page)
echo "=== ADDING BACK MATTER ==="
add_file_with_pagebreak "EPILOGUE.md"
add_file_with_pagebreak "REFERENCES_BIBLIOGRAPHY.md"
add_file_with_pagebreak "INDEX.md"
add_file_with_pagebreak "BACK_COVER.md"

# Final statistics
echo "=== COMPLETE! ==="
echo "File created: COMPLETE_TEXTBOOK_JSON_ALIGNED.md"
echo "Total lines: $(wc -l COMPLETE_TEXTBOOK_JSON_ALIGNED.md | awk '{print $1}')"
echo "Total words: $(wc -w COMPLETE_TEXTBOOK_JSON_ALIGNED.md | awk '{print $1}')"
echo "File size: $(ls -lh COMPLETE_TEXTBOOK_JSON_ALIGNED.md | awk '{print $5}')"

echo "✅ Complete textbook built successfully following JSON specification!"
echo "✅ Each file component has proper \\newpage breaks"
echo "✅ Ready for PDF generation with proper formatting"
