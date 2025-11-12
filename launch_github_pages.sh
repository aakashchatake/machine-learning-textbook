#!/bin/bash

# 🚀 Machine Learning Textbook - GitHub Pages Launch Script
# This script helps you quickly deploy the textbook to GitHub Pages

echo "🚀 Machine Learning Textbook - GitHub Pages Deployment Helper"
echo "============================================================"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a Git repository"
    echo "📋 To deploy to GitHub Pages:"
    echo "   1. Initialize git: git init"
    echo "   2. Add remote: git remote add origin <your-repo-url>"
    echo "   3. Add files: git add ."
    echo "   4. Commit: git commit -m 'Add Machine Learning Textbook'"
    echo "   5. Push: git push -u origin main"
    echo "   6. Enable GitHub Pages in repository settings → Pages → Source: /docs"
    exit 1
fi

echo "✅ Git repository detected"

# Check if docs folder exists
if [ ! -d "docs" ]; then
    echo "❌ Error: docs folder not found"
    exit 1
fi

echo "✅ GitHub Pages structure ready in /docs folder"

# Display current status
echo ""
echo "📊 Current Repository Status:"
echo "├── docs/index.html (Homepage)"
echo "├── docs/download.html (Download Portal)" 
echo "├── docs/downloads/ (All formats: PDF, DOCX, EPUB, HTML, MD)"
echo "├── docs/books/ (Online reading version)"
echo "└── Master source: Machine_Learning_A_Comprehensive_Guide_to_Artificial_Intelligence_and_Data_Science.md"

echo ""
echo "🌐 File Sizes Ready for Web:"
git ls-files docs/ | head -10 | while read file; do
    if [ -f "$file" ]; then
        size=$(ls -lah "$file" | awk '{print $5}')
        echo "   $file ($size)"
    fi
done

echo ""
echo "🚀 Next Steps for GitHub Pages Deployment:"
echo "   1. Commit all changes: git add . && git commit -m 'Ready for GitHub Pages'"
echo "   2. Push to GitHub: git push origin main"
echo "   3. Go to repository Settings → Pages"
echo "   4. Select Source: 'Deploy from a branch'"
echo "   5. Choose branch: 'main' and folder: '/docs'"
echo "   6. Save and wait for deployment (usually 2-5 minutes)"
echo ""
echo "📱 Your textbook will be live at: https://username.github.io/repository-name/"
echo ""
echo "✨ Features ready:"
echo "   • Professional download portal"
echo "   • Mobile-responsive design" 
echo "   • Multiple format downloads (PDF, DOCX, EPUB, HTML, MD)"
echo "   • Online reading interface"
echo "   • SEO optimized"
echo ""
echo "🎯 Status: READY FOR GLOBAL PUBLICATION! 🎊"
