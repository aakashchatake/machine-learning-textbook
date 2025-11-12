#!/bin/bash

cd "/Users/akashchatake/Downloads/Chatake-Innoworks-Organization/Projects/Publications/Machine-Learning-Textbook"

echo "📖 Starting PDF generation process..."
echo "=========================================="

# Step 1: Fix UTF-8 encoding and remove problematic characters
echo ""
echo "✓ Step 1: Normalizing text (fixing encoding)..."
iconv -f UTF-8 -t UTF-8//IGNORE SAFE_DRAFT_2.md > TEMP_UTF8.md

# Step 2: Replace smart quotes and special characters using perl
echo "✓ Step 2: Replacing smart quotes and special characters..."
perl -i -pe 's/[""]/"/g; s/['\'\']/'"'"'/g; s/…/.../g; s/—/--/g; s/–/--/g' TEMP_UTF8.md

# Step 3: Remove emojis using sed
echo "✓ Step 3: Removing emojis..."
sed -i '' 's/[📚📋✅🚀🎯💡⚡🔥🌟💻📊🔍🎉🔧📈🎨🔴🟡🟢✨🎓📝]//g' TEMP_UTF8.md

# Step 4: Save as FINAL_TEXTBOOK.md
echo "✓ Step 4: Saving as FINAL_TEXTBOOK.md..."
cp TEMP_UTF8.md FINAL_TEXTBOOK.md
rm TEMP_UTF8.md

# Step 5: Create build directory
echo "✓ Step 5: Creating build directory..."
mkdir -p build

# Step 6: Run Pandoc
echo "✓ Step 6: Generating PDF with XeLaTeX (this may take 1-2 minutes)..."
pandoc FINAL_TEXTBOOK.md \
  -o build/Machine_Learning_Textbook_Beautiful.pdf \
  --from markdown+tex_math_dollars+fenced_code_attributes \
  --pdf-engine=xelatex \
  -V mainfont="TeX Gyre Pagella" \
  -V monofont="Fira Code" \
  -V sansfont="TeX Gyre Heros" \
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

# Step 7: Verify
echo ""
echo "✓ Step 7: Verifying output..."
if [ -f "build/Machine_Learning_Textbook_Beautiful.pdf" ]; then
  SIZE=$(ls -lh build/Machine_Learning_Textbook_Beautiful.pdf | awk '{print $5}')
  echo "  PDF created successfully! (Size: $SIZE)"
  echo ""
  echo "=========================================="
  echo "✅ Beautifully formatted MindforgeAI"
  echo "   Founder's Edition PDF generated"
  echo "   successfully!"
  echo "✅ Machine_Learning_Textbook_Beautiful.pdf"
  echo "   is ready for review."
  echo "=========================================="
else
  echo "  ERROR: PDF file not created!"
  exit 1
fi
