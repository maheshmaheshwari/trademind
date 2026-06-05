import type { Config } from 'tailwindcss';
import plugin from 'tailwindcss/plugin';

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],

  // Activate dark: variant when data-theme="dark" is on <html>
  darkMode: ['selector', '[data-theme="dark"]'],

  theme: {
    extend: {
      colors: {
        bg:      '#0A0E1A',
        surface: {
          DEFAULT: '#111827',
          2:       '#161F33',
          3:       '#1C2740',
          hover:   '#1A2438',
        },
        line: {
          DEFAULT: 'rgba(255,255,255,0.07)',
          strong:  'rgba(255,255,255,0.13)',
        },
        ink: {
          DEFAULT: '#EEF2F9',
          2:       '#AEB9CE',
          3:       '#6B7890',
        },
        accent: {
          DEFAULT: '#3B82F6',
          soft:    'rgba(59,130,246,0.14)',
          2:       '#60A5FA',
        },
        gain: {
          DEFAULT: '#10B981',
          soft:    'rgba(16,185,129,0.14)',
        },
        loss: {
          DEFAULT: '#EF4444',
          soft:    'rgba(239,68,68,0.14)',
        },
        gold: {
          DEFAULT: '#F59E0B',
          soft:    'rgba(245,158,11,0.16)',
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
