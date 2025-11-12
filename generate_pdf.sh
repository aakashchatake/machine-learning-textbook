#!/bin/bash

# PDF Generation Script for Machine Learning Textbook
cd "/Users/akashchatake/Downloads/Chatake-Innoworks-Organization/Projects/Publications/Machine-Learning-Textbook"

echo "🔧 Starting PDF generation process..."

# Step 1: Create clean version
echo "Step 1: Creating clean version..."
cp "SAFE_DRAFT_2.md" "FINAL_TEXTBOOK.md"

# Step 2: Remove problematic Unicode characters
echo "Step 2: Cleaning Unicode characters..."
sed -i '' 's/📚/[BOOK]/g' "FINAL_TEXTBOOK.md"
sed -i '' 's/📋/[CHECKLIST]/g' "FINAL_TEXTBOOK.md"  
sed -i '' 's/✅/[COMPLETED]/g' "FINAL_TEXTBOOK.md"
sed -i '' 's/🚀//g' "FINAL_TEXTBOOK.md"
sed -i '' 's/🎯//g' "FINAL_TEXTBOOK.md"
sed -i '' 's/💡//g' "FINAL_TEXTBOOK.md"
sed -i '' 's/⚡//g' "FINAL_TEXTBOOK.md"
sed -i '' 's/🔥//g' "FINAL_TEXTBOOK.md"
sed -i '' 's/🌟//g' "FINAL_TEXTBOOK.md"
sed -i '' 's/💻//g' "FINAL_TEXTBOOK.md"
sed -i '' 's/📊//g' "FINAL_TEXTBOOK.md"
sed -i '' 's/🔍//g' "FINAL_TEXTBOOK.md"
sed -i '' 's/[""]/"/g' "FINAL_TEXTBOOK.md"
sed -i '' "s/['']/'/g" "FINAL_TEXTBOOK.md"

# Step 3: Create build directory
echo "Step 3: Creating build directory..."
mkdir -p build

# Step 4: Generate PDF with comprehensive options
echo "Step 4: Generating PDF..."
pandoc FINAL_TEXTBOOK.md \
  -o build/Machine_Learning_Textbook_Beautiful.pdf \
  --from markdown+tex_math_dollars+fenced_code_attributes \
  --pdf-engine=xelatex \
  -V mainfont="Times New Roman" \
  -V monofont="Courier New" \
  -V sansfont="Arial" \
  -V fontsize=11pt \
  -V geometry:"top=1in,bottom=1in,left=1in,right=1in" \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  --toc --toc-depth=3 --number-sections \
  --metadata title="Machine Learning: Foundations & Futures" \
  --metadata subtitle="A Comprehensive Guide to Artificial Intelligence and Data Science" \
  --metadata author="Akash Chatake (MindforgeAI / Chatake Innoworks Pvt. Ltd.)" \
  --metadata date="November 2025" \
  --metadata rights="© 2025 Chatake Innoworks Organization. All rights reserved." \
  2>&1

# Check if PDF was created
if [ -f "build/Machine_Learning_Textbook_Beautiful.pdf" ]; then
    echo ""
    echo "✅ Beautifully formatted MindforgeAI Founder's Edition PDF generated successfully — Machine_Learning_Textbook_Beautiful.pdf is ready for review."
    echo ""
    ls -lh build/Machine_Learning_Textbook_Beautiful.pdf
else
    echo "❌ PDF generation failed. Please check the error messages above."
    exit 1
fi
