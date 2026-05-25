---
description: You are a senior UX engineer doing a pre-delivery audit of this security tool's UI.
---

Review the current index.html (and any CSS/JS files) against these checks in order:

1. ACCESSIBILITY: Is every interactive element keyboard-reachable? Is color contrast ≥4.5:1? Is threat status conveyed with color + icon + text (not color alone)?
2. TOUCH & INTERACTION: Are all buttons ≥44px tall? Is there visible feedback within 100ms of clicking "Check URL"?
3. LOADING STATES: Does the UI show a skeleton/spinner immediately on submit? Is there a timeout error message if the API takes >10s?
4. ERROR HANDLING: Does every API error (429, 422, 500) show a human-readable message with a recovery action?
5. RESULT CLARITY: Are verdict, confidence %, and reason ALL visible without additional clicks?
6. RESPONSIVE: Does the UI work at 768px (tablet) and 375px (phone) with no horizontal scroll?
7. TYPOGRAPHY: Is IBM Plex Mono used for URLs and verdicts? Is body text ≥16px?

Output a prioritized list of issues found: CRITICAL → HIGH → MEDIUM → LOW.
For each issue, provide the exact fix as code.