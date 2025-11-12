#!/bin/bash

# 🚀 Complete GitHub Launch Script for Machine Learning Textbook
# This script automates the entire GitHub deployment process

echo "🚀 Machine Learning Textbook - Complete GitHub Launch"
echo "====================================================="

echo "📋 Repository Details:"
echo "   📧 GitHub User: akashchatake"
echo "   📁 Repository: machine-learning-textbook"
echo "   🌐 Future URL: https://akashchatake.github.io/machine-learning-textbook/"
echo ""

# Check current git status
echo "📊 Current Git Status:"
git status --short | head -5
echo "✅ $(git rev-list --count HEAD) commits ready"
echo "✅ $(git ls-files | wc -l | tr -d ' ') files ready for upload"
echo ""

# Show what will be deployed
echo "📚 Your Textbook Content Ready:"
echo "   📖 Master File: Machine_Learning_A_Comprehensive_Guide_to_Artificial_Intelligence_and_Data_Science.md ($(ls -lah Machine_Learning_A_Comprehensive_Guide_to_Artificial_Intelligence_and_Data_Science.md | awk '{print $5}'))"
echo "   🌐 Website: docs/index.html ($(ls -lah docs/index.html | awk '{print $5}'))"
echo "   📥 Downloads: docs/downloads/ (5 formats available)"
echo "   📱 Mobile: Fully responsive design"
echo ""

echo "🎯 What happens when you create the repository:"
echo "   1. ✅ Repository 'machine-learning-textbook' gets created"
echo "   2. 📤 All your textbook files upload to GitHub" 
echo "   3. 🌐 GitHub Pages automatically activates"
echo "   4. 🚀 Your textbook goes LIVE worldwide!"
echo ""

echo "📋 Step-by-Step Launch Process:"
echo ""
echo "🔗 STEP 1: Create Repository on GitHub"
echo "   • Click this link: https://github.com/new"
echo "   • Repository name: machine-learning-textbook"
echo "   • Description: Machine Learning: A Comprehensive Guide to AI and Data Science - MSBTE Course 316316"
echo "   • Visibility: Public ✅ (required for free GitHub Pages)"
echo "   • Initialize: DO NOT check any boxes ❌"
echo "   • Click 'Create Repository' 🟢"
echo ""

echo "📤 STEP 2: Push Your Textbook (I'll do this for you!)"
echo "   Ready to run: git push -u origin main"
echo ""

echo "⚙️ STEP 3: Enable GitHub Pages"
echo "   • Go to: Repository Settings → Pages"
echo "   • Source: Deploy from a branch" 
echo "   • Branch: main"
echo "   • Folder: /docs"
echo "   • Click Save"
echo ""

echo "🎊 STEP 4: Your Textbook Goes Live!"
echo "   • URL: https://akashchatake.github.io/machine-learning-textbook/"
echo "   • Time to live: 2-5 minutes after enabling Pages"
echo "   • Features: Download portal, online reading, mobile-friendly"
echo ""

echo "❓ Ready to push to GitHub now? (y/n)"
read -p "   Enter 'y' when you've created the repository: " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo ""
    echo "🚀 Launching your textbook to GitHub..."
    
    if git push -u origin main; then
        echo ""
        echo "🎊 SUCCESS! Your textbook is now on GitHub!"
        echo ""
        echo "📋 Final Steps:"
        echo "   1. Go to: https://github.com/akashchatake/machine-learning-textbook"
        echo "   2. Click: Settings → Pages"
        echo "   3. Set Source: Deploy from branch"
        echo "   4. Set Branch: main, Folder: /docs"
        echo "   5. Click Save"
        echo ""
        echo "🌐 Your site will be live at:"
        echo "   https://akashchatake.github.io/machine-learning-textbook/"
        echo ""
        echo "✨ Features your visitors will get:"
        echo "   • Professional homepage with book info"
        echo "   • Download portal with PDF, DOCX, EPUB, HTML, MD"
        echo "   • Online reading interface"
        echo "   • Mobile-responsive design"
        echo "   • SEO optimized for discovery"
        echo ""
        echo "🎯 MISSION ACCOMPLISHED! 🚀📚🌍"
    else
        echo ""
        echo "⚠️  Push failed. Please check:"
        echo "   1. Repository exists on GitHub"
        echo "   2. Repository name is: machine-learning-textbook"
        echo "   3. You have push access"
        echo ""
        echo "💡 Try again after creating the repository!"
    fi
else
    echo ""
    echo "📋 No problem! When you're ready:"
    echo "   1. Create repository: https://github.com/new"
    echo "   2. Run this script again"
    echo "   3. Your textbook will go live!"
fi

echo ""
echo "📞 Need help? Check GITHUB_SETUP_GUIDE.md for detailed instructions!"
