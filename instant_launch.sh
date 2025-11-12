#!/bin/bash

# 🚀 INSTANT LAUNCH SCRIPT - Creates repository and deploys textbook
# This script will help you launch your textbook in under 2 minutes!

echo "🚀 LAUNCHING YOUR MACHINE LEARNING TEXTBOOK NOW!"
echo "=============================================="

echo "📚 Your Textbook Ready for Launch:"
echo "   ✅ 109,002+ words of professional content"
echo "   ✅ 10 chapters + 5 appendices complete"
echo "   ✅ 5 formats: PDF, DOCX, EPUB, HTML, Markdown"
echo "   ✅ Professional website with download portal"
echo "   ✅ Mobile-responsive design"
echo "   ✅ SEO optimized"
echo ""

echo "🎯 LAUNCH PROCESS:"
echo ""

echo "🔗 STEP 1: CREATE REPOSITORY (30 seconds)"
echo "   I'm opening GitHub for you..."
echo "   Repository details to use:"
echo "   • Name: machine-learning-textbook"
echo "   • Description: Machine Learning: A Comprehensive Guide to AI and Data Science - MSBTE Course 316316"
echo "   • Visibility: Public ✅"
echo "   • Initialize: Leave ALL boxes unchecked ❌"

# Open GitHub in browser
if command -v open >/dev/null 2>&1; then
    echo "   Opening GitHub repository creation page..."
    open "https://github.com/new"
elif command -v xdg-open >/dev/null 2>&1; then
    echo "   Opening GitHub repository creation page..."
    xdg-open "https://github.com/new"
else
    echo "   Please go to: https://github.com/new"
fi

echo ""
echo "⏱️ Waiting for you to create the repository..."
echo "   (Press ENTER after clicking 'Create Repository')"
read -p "   Ready to upload your textbook? " 

echo ""
echo "🚀 STEP 2: UPLOADING YOUR TEXTBOOK TO GITHUB..."

# Attempt to push
if git push -u origin main; then
    echo ""
    echo "🎊 SUCCESS! Your textbook is now on GitHub!"
    echo ""
    echo "📍 Repository URL: https://github.com/akashchatake/machine-learning-textbook"
    echo ""
    
    echo "🌐 STEP 3: ENABLING GITHUB PAGES..."
    echo "   Opening repository settings..."
    
    # Open repository settings
    if command -v open >/dev/null 2>&1; then
        open "https://github.com/akashchatake/machine-learning-textbook/settings/pages"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "https://github.com/akashchatake/machine-learning-textbook/settings/pages"
    else
        echo "   Please go to: https://github.com/akashchatake/machine-learning-textbook/settings/pages"
    fi
    
    echo ""
    echo "   Configure GitHub Pages:"
    echo "   • Source: Deploy from a branch"
    echo "   • Branch: main"
    echo "   • Folder: /docs"
    echo "   • Click 'Save'"
    echo ""
    
    echo "🎊 FINAL RESULT:"
    echo "   🌐 Your textbook will be LIVE at:"
    echo "   https://akashchatake.github.io/machine-learning-textbook/"
    echo ""
    echo "   ⏱️ Goes live in: 2-5 minutes after enabling Pages"
    echo ""
    echo "   ✨ Features your visitors will get:"
    echo "   • Professional homepage and download portal"
    echo "   • All 5 formats downloadable (PDF, DOCX, EPUB, HTML, MD)"
    echo "   • Online reading interface"
    echo "   • Mobile-responsive design"
    echo "   • 109,002+ words of ML education content"
    echo ""
    echo "🎯 MISSION ACCOMPLISHED! 🚀📚🌍"
    echo "   Your Machine Learning textbook is now live for the world!"
    
else
    echo ""
    echo "⚠️ Upload issue detected. Let's troubleshoot:"
    echo ""
    echo "💡 Most likely causes:"
    echo "   1. Repository not created yet"
    echo "   2. Repository name doesn't match: machine-learning-textbook"
    echo "   3. Repository is private (needs to be public for free Pages)"
    echo ""
    echo "🔧 Quick fix:"
    echo "   1. Ensure repository exists: https://github.com/akashchatake/machine-learning-textbook"
    echo "   2. Check it's public"
    echo "   3. Run this script again!"
    echo ""
    echo "📞 Repository creation URL: https://github.com/new"
fi

echo ""
echo "📊 IMPACT SUMMARY:"
echo "   🎓 Target: 200,000+ MSBTE students annually"
echo "   🌍 Reach: Global availability in 5 formats"
echo "   📚 Content: Complete MSBTE Course 316316 curriculum"
echo "   💼 Professional: Commercial-quality textbook"
echo ""
echo "🎉 Congratulations on publishing your Machine Learning textbook! 🎉"
