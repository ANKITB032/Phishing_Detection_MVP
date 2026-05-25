---
description: Redesign index.html from scratch using the global UI/UX rules for PhishGuard.
---

Context:
- Stack: Vanilla HTML + CSS + JavaScript (no framework, no bundler — single file)
- Backend API: POST /predict at https://ab2403-phishing-detection-mvp.hf.space/predict
- Response shape: { label, verdict, confidence, reason, security_analysis }
- Current fonts already used: IBM Plex fonts (keep them)

Design requirements:
1. Dark navy background (#0F172A), card surfaces (#1E293B)
2. Cyan accent (#06B6D4) for benign results, Red (#EF4444) for malicious, Amber (#F59E0B) for borderline (<75% confidence)
3. Hero section: Logo + tagline + URL input (monospace) + "Analyze" button
4. Result card: verdict badge (colored + icon), confidence bar, reason text, collapsible "Security Analysis" section showing threat_flags
5. Loading state: animated skeleton card that appears within 100ms of submit
6. Error states: human-readable messages for 429 (rate limit: "Too many requests. Wait 60 seconds and try again."), 422 (invalid URL), 500 (server error)
7. Footer: API status indicator (ping /health on load), version badge
8. Desktop-first layout optimized for 1280px+. Must also not break at 768px tablet or 375px phone.
9. Accessibility: WCAG AA contrast, focus rings, aria-labels on all interactive elements, verdict never conveyed by color alone

Deliver the complete updated index.html.