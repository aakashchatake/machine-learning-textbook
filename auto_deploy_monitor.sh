#!/bin/bash

# 🚀 AUTO-DEPLOY MONITOR - Watches for repository and auto-uploads textbook
# This script checks if the repository exists and automatically uploads when ready

echo "🔍 AUTO-DEPLOY MONITOR ACTIVE"
echo "============================="
echo "Watching for repository: akashchatake/machine-learning-textbook"
echo ""

# Function to check if repository exists
check_repo() {
    git ls-remote --heads origin >/dev/null 2>&1
    return $?
}

echo "⏱️ Checking repository every 10 seconds..."
echo "   (Create the repository at: https://github.com/new)"
echo ""

# Monitor loop
attempt=1
while true; do
    echo "🔍 Check #$attempt - $(date '+%H:%M:%S')"
    
    if check_repo; then
        echo ""
        echo "🎊 REPOSITORY DETECTED! Starting upload..."
        echo ""
        
        if git push -u origin main; then
            echo ""
            echo "🚀 SUCCESS! Your textbook is now LIVE on GitHub!"
            echo ""
            echo "📍 Repository: https://github.com/akashchatake/machine-learning-textbook"
            echo "🌐 Enable Pages at: https://github.com/akashchatake/machine-learning-textbook/settings/pages"
            echo ""
            echo "⚙️ GitHub Pages Setup:"
            echo "   • Source: Deploy from a branch"
            echo "   • Branch: main"  
            echo "   • Folder: /docs"
            echo "   • Click 'Save'"
            echo ""
            echo "🎯 Your site will be live at:"
            echo "   https://akashchatake.github.io/machine-learning-textbook/"
            echo ""
            echo "🎊 LAUNCH COMPLETE! Your textbook is now globally accessible! 🌍📚"
            
            # Open the repository and settings
            if command -v open >/dev/null 2>&1; then
                echo "   Opening repository settings for you..."
                open "https://github.com/akashchatake/machine-learning-textbook/settings/pages"
            fi
            
            break
        else
            echo "❌ Upload failed. Please check repository settings."
        fi
    else
        echo "   ⏳ Repository not found yet..."
    fi
    
    ((attempt++))
    sleep 10
done

echo ""
echo "✨ Your Machine Learning textbook deployment is complete!"
echo "📊 Final stats: 109,002+ words, 5 formats, professional website ready!"
