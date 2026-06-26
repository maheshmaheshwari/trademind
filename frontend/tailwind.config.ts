import type { Config } from 'tailwindcss';
import plugin from 'tailwindcss/plugin';

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],

  // Activate dark: variant when data-theme="dark" is on <html>
  darkMode: ['selector', '[data-theme="dark"]'],

  theme: {
    extend: {
      // Every color below references the CSS custom properties defined per-theme
      // in src/index.css ([data-theme="dark"] / [data-theme="light"]). These were
      // previously hardcoded to literal dark-mode hex values, completely
      // disconnected from the theme system — any Tailwind-generated utility that
      // wasn't ALSO manually duplicated in index.css (anything using a modifier
      // like hover: or an opacity suffix, e.g. bg-surface/70) stayed stuck in
      // dark colors even in light mode. This is what caused the light-mode navbar
      // background and the black row-hover color on /orders.
      colors: {
        bg:      'var(--bg)',
        // surface.DEFAULT and accent.DEFAULT use the rgb(var(...) / <alpha-value>)
        // pattern (not a plain var() string) specifically so Tailwind's opacity
        // modifier (e.g. bg-surface/70, bg-accent/10) keeps working correctly per
        // theme — a plain var() reference can't be modified with /<N> by Tailwind.
        surface: {
          DEFAULT: 'rgb(var(--surface-rgb) / <alpha-value>)',
          2:       'var(--surface-2)',
          3:       'var(--surface-3)',
          hover:   'var(--surface-hover)',
        },
        line: {
          DEFAULT: 'var(--border)',
          strong:  'var(--border-strong)',
        },
        ink: {
          DEFAULT: 'var(--text)',
          2:       'var(--text-2)',
          3:       'var(--text-3)',
        },
        accent: {
          DEFAULT: 'rgb(var(--accent-rgb) / <alpha-value>)',
          soft:    'var(--accent-soft)',
          2:       'var(--accent-2)',
        },
        gain: {
          DEFAULT: 'var(--green)',
          soft:    'var(--green-soft)',
        },
        loss: {
          DEFAULT: 'var(--red)',
          soft:    'var(--red-soft)',
        },
        gold: {
          DEFAULT: 'var(--gold)',
          soft:    'var(--gold-soft)',
        },
      },

      fontFamily: {
        sans: ['DM Sans', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SF Mono', 'Menlo', 'monospace'],
      },

      borderRadius: {
        card:    '14px',
        'card-sm': '9px',
        'card-lg': '18px',
      },

      boxShadow: {
        sm: '0 1px 2px rgba(0,0,0,.18)',
        md: '0 6px 24px rgba(0,0,0,.28)',
        lg: '0 18px 50px rgba(0,0,0,.40)',
      },

      keyframes: {
        shimmer: {
          '0%':   { backgroundPosition: '100% 0' },
          '100%': { backgroundPosition: '-100% 0' },
        },
        pulseDot: {
          '0%,100%': { boxShadow: '0 0 0 0 rgba(16,185,129,0)' },
          '70%':     { boxShadow: '0 0 0 7px rgba(16,185,129,0)' },
        },
        slideIn: {
          from: { transform: 'translateX(34px)' },
          to:   { transform: 'translateX(0)' },
        },
        pop: {
          from: { transform: 'translateY(10px)' },
          to:   { transform: 'translateY(0)' },
        },
        toastIn: {
          from: { transform: 'translateY(12px)', opacity: '0' },
          to:   { transform: 'translateY(0)',    opacity: '1' },
        },
        toastOut: {
          from: { transform: 'translateY(0)',  opacity: '1' },
          to:   { transform: 'translateY(8px)', opacity: '0' },
        },
        pageIn: {
          from: { transform: 'translateY(8px)' },
          to:   { transform: 'none' },
        },
        scrimIn: {
          from: { backdropFilter: 'blur(0)' },
          to:   { backdropFilter: 'blur(2px)' },
        },
      },

      animation: {
        shimmer:    'shimmer 1.3s ease-in-out infinite',
        'pulse-dot':'pulseDot 2s infinite',
        'slide-in': 'slideIn .3s cubic-bezier(.4,0,.2,1) both',
        pop:        'pop .22s cubic-bezier(.4,0,.2,1) both',
        'toast-in': 'toastIn .3s cubic-bezier(.4,0,.2,1) both',
        'toast-out':'toastOut .25s cubic-bezier(.4,0,.2,1) both',
        'page-in':  'pageIn .32s cubic-bezier(.4,0,.2,1) both',
        'scrim-in': 'scrimIn .2s cubic-bezier(.4,0,.2,1) both',
      },
    },
  },

  plugins: [
    // light: variant — applies when data-theme="light"
    plugin(({ addVariant }) => {
      addVariant('light', '[data-theme="light"] &');
    }),
  ],
} satisfies Config;
