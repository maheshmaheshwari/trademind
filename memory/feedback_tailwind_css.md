---
name: feedback-tailwind-css
description: CSS styling approach for TradeMind frontend — Tailwind first, inline and external CSS only as fallback
metadata:
  type: feedback
---

Always use **Tailwind CSS utilities as the primary styling method** for TradeMind frontend components.

**Why:** User explicitly prefers Tailwind over inline styles and external CSS classes.

**How to apply:**
- Use Tailwind utility classes for layout, spacing, colors, typography, borders, shadows, hover states
- CSS variables (like `var(--accent)`, `var(--surface)`) can be used via Tailwind arbitrary values: `text-[var(--accent)]`, `bg-[var(--surface)]`, `border-[var(--border)]`
- Inline `style={{}}` props are only acceptable when Tailwind has no equivalent — e.g. `background: 'linear-gradient(...)'`, complex CSS variables in dynamic values, `animation`, CSS grid with non-standard column definitions
- External CSS / `className` with custom CSS classes in index.css only when Tailwind absolutely cannot express it (e.g. `@keyframes`, complex `::before`/`::after` pseudo-elements, `::-webkit-scrollbar`)
- Never reach for inline styles just because they're familiar — check Tailwind first
