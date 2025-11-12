#!/bin/bash

# Script to create properly formatted textbook with correct page breaks and structure
cd "/Users/akashchatake/Downloads/Chatake-Innoworks-Organization/Projects/Publications/Machine-Learning-Textbook"

echo "Creating properly structured textbook with all components..."

# Create the final structured textbook
cat > PROPERLY_FORMATTED_TEXTBOOK.md << 'EOF'
# MACHINE LEARNING

## A Comprehensive Guide to Artificial Intelligence and Data Science

### From Fundamentals to Advanced Applications

**By:** Akash Chatake

**Publisher:** Chatake Innoworks Publications

**Edition:** First Edition, 2025

**Series:** Computer Technology & Engineering Series

**Course Code:** MSBTE 316316

---

> "Bridging Theory and Practice in the Age of AI"

\newpage

# COPYRIGHT

© 2025 Chatake Innoworks Publications. All rights reserved.

No part of this publication may be reproduced, stored in a retrieval system, or transmitted in any form or by any means—electronic, mechanical, photocopying, recording, or otherwise—without the prior written permission of the publisher.

**Publisher:** Chatake Innoworks Publications
**Printed in:** India
**Edition:** First Edition, 2025
**ISBN:** [Add ISBN]

**Course Code Compliance:** MSBTE 316316

**Permissions & Requests:**
Requests for permissions should be addressed to: permissions@chatakeinnoworks.example

**Credits:**
Cover design: [Designer Name]
Production: Chatake Innoworks Production Team

**Disclaimer:**
The information contained in this book is provided for educational purposes only. The authors and publisher make no guarantees of results and assume no liability for any damages that may result from the use of the information contained herein.

\newpage

# DEDICATION

> Dedicated to every learner, teacher, and practitioner who chooses curiosity over comfort.

*For my students — the minds shaping tomorrow's intelligence.*

\newpage

# ABOUT THE AUTHOR

**Akash Chatake** is a Lecturer in Mathematics and an AI researcher associated with Chatake Innoworks Pvt. Ltd. He leads educational initiatives through the MindforgeAI division, focusing on bridging rigorous mathematical foundations with practical machine learning implementations. Akash has taught undergraduate and diploma students, supervised industry-linked student projects, and published research on engineering transitions.

**Contact / Affiliations:**
- Chatake Innoworks Pvt. Ltd. / MindforgeAI
- Email: [add email]

\newpage

# PREFACE

This textbook is written to provide a unified introduction to machine learning for diploma and undergraduate students. It balances mathematical rigor, algorithmic intuition, and hands-on implementation.

**Goals:**
- Build strong foundations in the mathematics behind ML.
- Provide clear algorithmic explanations with worked examples.
- Supply practical Python code snippets and end-to-end projects.

How to use this book:
- Read theory sections for conceptual clarity.
- Try the code examples in a Jupyter notebook or Google Colab.
- Solve exercises at the end of chapters and consult appended solutions.

Acknowledgements and influences are listed in the Acknowledgments file.

\newpage

EOF

# Now append the actual chapter content starting from Chapter 1
echo "Appending actual chapter content..."
tail -n +927 SAFE_DRAFT_2.md >> PROPERLY_FORMATTED_TEXTBOOK.md

echo "✅ Properly formatted textbook created: PROPERLY_FORMATTED_TEXTBOOK.md"
wc -l PROPERLY_FORMATTED_TEXTBOOK.md
