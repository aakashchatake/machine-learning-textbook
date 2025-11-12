#!/bin/bash

# 🚀 Quick GitHub Setup Script for Machine Learning Textbook
# This script helps you connect your textbook to GitHub quickly

echo "🚀 Machine Learning Textbook - GitHub Quick Setup"
echo "================================================"

# Get user input
echo "📝 Please provide your GitHub details:"
read -p "GitHub Username: " username
read -p "Repository Name (or press Enter for 'machine-learning-textbook'): " repo_name

# Set default repo name if empty
if [ -z "$repo_name" ]; then
    repo_name="machine-learning-textbook"
fi

echo ""
echo "🔧 Setting up repository connection..."
echo "   Username: $username"
echo "   Repository: $repo_name"
echo "   URL: https://github.com/$username/$repo_name"

# Check if remote already exists
if git remote get-url origin >/dev/null 2>&1; then
    echo "⚠️  Remote 'origin' already exists. Removing old remote..."
    git remote remove origin
fi

# Add the new remote
git remote add origin "https://github.com/$username/$repo_name.git"

echo "✅ Remote repository configured!"
echo ""

echo "🚀 Next steps:"
echo "1. Create the repository on GitHub:"
echo "   • Go to https://github.com/new"
echo "   • Repository name: $repo_name"  
echo "   • Description: Machine Learning Textbook - MSBTE Course 316316"
echo "   • Make it Public (for free GitHub Pages)"
echo "   • DO NOT initialize with README"
echo ""
echo "2. Push your textbook to GitHub:"
echo "   git push -u origin main"
echo ""
echo "3. Enable GitHub Pages:"
echo "   • Repository Settings → Pages"
echo "   • Source: Deploy from branch"
echo "   • Branch: main, Folder: /docs"
echo ""
echo "🌐 Your site will be live at:"
echo "   https://$username.github.io/$repo_name/"
echo ""
echo "📚 Ready to share your textbook with the world! 🎊"
