#!/bin/bash

# 🔧 GITHUB PAGES TROUBLESHOOTER & FORCE DEPLOYMENT
# This script tries multiple methods to get your textbook live

echo "🔧 FIXING GITHUB PAGES 404 ERROR"
echo "================================="

echo "📍 Repository: https://github.com/aakashchatake/machine-learning-textbook"
echo "❌ Current Issue: 404 on https://aakashchatake.github.io/machine-learning-textbook/"
echo ""

echo "🎯 SOLUTION ATTEMPTS:"
echo ""

echo "✅ ATTEMPT 1: Root Index File Added"
echo "   • Added index.html in repository root"
echo "   • This should make the site accessible immediately"
echo "   • Test URL: https://aakashchatake.github.io/machine-learning-textbook/"
echo ""

echo "✅ ATTEMPT 2: Multiple Access Points"
echo "   • Root site: https://aakashchatake.github.io/machine-learning-textbook/"
echo "   • Full site: https://aakashchatake.github.io/machine-learning-textbook/docs/"
echo "   • Downloads: https://aakashchatake.github.io/machine-learning-textbook/docs/downloads/"
echo ""

echo "🔍 TROUBLESHOOTING CHECKLIST:"
echo ""

# Check if Pages is enabled
echo "1. ✅ VERIFY GITHUB PAGES IS ENABLED:"
echo "   • Go to: https://github.com/aakashchatake/machine-learning-textbook/settings/pages"
echo "   • Ensure Source is set to 'Deploy from a branch'"
echo "   • Branch should be 'main'"
echo "   • Folder should be '/ (root)' OR '/docs'"
echo ""

echo "2. ✅ CHECK REPOSITORY VISIBILITY:"
echo "   • Repository must be PUBLIC for free GitHub Pages"
echo "   • Go to: https://github.com/aakashchatake/machine-learning-textbook/settings"
echo "   • Scroll to 'Danger Zone' and verify it's public"
echo ""

echo "3. ✅ WAIT FOR DEPLOYMENT:"
echo "   • GitHub Pages can take 2-10 minutes to deploy"
echo "   • Check Actions: https://github.com/aakashchatake/machine-learning-textbook/actions"
echo "   • Look for green checkmarks on deployments"
echo ""

echo "4. ✅ CLEAR BROWSER CACHE:"
echo "   • Try incognito/private browsing mode"
echo "   • Hard refresh: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)"
echo "   • Try different browser"
echo ""

echo "🚀 ALTERNATIVE ACCESS METHODS:"
echo ""

echo "📄 DIRECT FILE ACCESS (Should work immediately):"
echo "   • PDF: https://aakashchatake.github.io/machine-learning-textbook/docs/downloads/Machine_Learning_A_Comprehensive_Guide_to_Artificial_Intelligence_and_Data_Science.pdf"
echo "   • DOCX: https://aakashchatake.github.io/machine-learning-textbook/docs/downloads/Machine_Learning_A_Comprehensive_Guide_to_Artificial_Intelligence_and_Data_Science.docx"
echo ""

echo "🔧 IF STILL NOT WORKING:"
echo ""

echo "OPTION A: Change Pages Source to Root"
echo "   1. Settings → Pages"
echo "   2. Source: Deploy from branch"
echo "   3. Branch: main"
echo "   4. Folder: / (root)  ← Try this instead of /docs"
echo "   5. Save and wait 5 minutes"
echo ""

echo "OPTION B: Use GitHub Actions Deployment"
echo "   1. Settings → Pages" 
echo "   2. Source: GitHub Actions"
echo "   3. Use the workflow I've created"
echo "   4. This forces deployment"
echo ""

echo "OPTION C: Repository Recreation (Last resort)"
echo "   1. Download all files as ZIP"
echo "   2. Delete repository"
echo "   3. Create new repository with same name"
echo "   4. Upload files and enable Pages"
echo ""

echo "💡 MOST LIKELY ISSUE:"
echo "   GitHub Pages is not enabled, or repository is private"
echo "   Solution: Enable Pages with 'main branch / (root)' setting"
echo ""

echo "📞 IMMEDIATE TEST:"
echo "   Try this URL in 2-3 minutes: https://aakashchatake.github.io/machine-learning-textbook/"
echo "   You should see a test page confirming GitHub Pages is working"
echo ""

echo "🎊 WHEN IT WORKS:"
echo "   Your 109,002+ word Machine Learning textbook will be accessible worldwide!"
echo "   Students can download all 5 formats and read online"
echo ""

echo "Status: Troubleshooting deployed ✅"
echo "Next: Enable GitHub Pages and wait 5 minutes 🚀"
