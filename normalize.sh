#!/bin/bash

# Script to normalize the textbook file and generate PDF

cd "/Users/akashchatake/Downloads/Chatake-Innoworks-Organization/Projects/Publications/Machine-Learning-Textbook"

echo "🔧 Starting normalization process..."

# Step 1: Copy source file
cp "SAFE_DRAFT_2.md" "FINAL_TEXTBOOK.md"

# Step 2: Replace curly quotes with straight ASCII quotes
sed -i '' 's/[""]/"/g' "FINAL_TEXTBOOK.md"
sed -i '' "s/['']/'/g" "FINAL_TEXTBOOK.md"

# Step 3: Ensure UTF-8 encoding
iconv -f UTF-8 -t UTF-8 "FINAL_TEXTBOOK.md" > "FINAL_TEXTBOOK_temp.md" 2>/dev/null
mv "FINAL_TEXTBOOK_temp.md" "FINAL_TEXTBOOK.md" 2>/dev/null

echo "✅ Normalization complete."

# Step 4: Create build directory
mkdir -p build

echo "📁 Build directory ready."

# Step 5: Generate beautiful PDF with Pandoc and XeLaTeX
echo "🎨 Generating beautiful PDF..."

pandoc FINAL_TEXTBOOK.md \
  -o build/Machine_Learning_Textbook_Beautiful.pdf \
  --from markdown+tex_math_dollars+fenced_code_attributes \
  --pdf-engine=xelatex \
  -V mainfont="EB Garamond" \
  -V monofont="Fira Code" \
  -V sansfont="Montserrat" \
  -V fontsize=11pt \
  -V geometry:"top=1in,bottom=1in,left=1in,right=1in" \
  -V colorlinks=true \
  -V linkcolor=MidnightBlue \
  -V urlcolor=RoyalBlue \
  -V toccolor=Gray \
  -V header-includes='\usepackage{fancyhdr}\pagestyle{fancy}\fancyhead[CO,CE]{Machine Learning: Foundations & Futures}\fancyfoot[CO,CE]{Akash Chatake — MindforgeAI Press}' \
  --toc --toc-depth=3 --number-sections \
  --syntax-highlighting=pygments \
  --metadata title="Machine Learning: Foundations & Futures" \
  --metadata subtitle="A Comprehensive Guide to Artificial Intelligence and Data Science" \
  --metadata author="Akash Chatake (MindforgeAI / Chatake Innoworks Pvt. Ltd.)" \
  --metadata date="November 2025" \
  --metadata rights="© 2025 Chatake Innoworks Organization. All rights reserved."

# Step 6: Verify output
if [ -f "build/Machine_Learning_Textbook_Beautiful.pdf" ]; then
    echo "✅ Beautifully formatted MindforgeAI Founder's Edition PDF generated successfully — Machine_Learning_Textbook_Beautiful.pdf is ready for review."
    ls -lh build/Machine_Learning_Textbook_Beautiful.pdf
else
    echo "❌ PDF generation failed. Please check the error messages above."
fi
