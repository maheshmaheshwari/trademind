/** @type {import('tailwindcss').Config} */
import plugin from 'tailwindcss/plugin';

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'primary': '#0d7ff2',
        'background-dark': '#101922',
        'surface-dark': '#1a2632',
        'background-light': '#f8fafc',
        'surface-light': '#ffffff',
      },
      fontFamily: {
        'display': ['Space Grotesk', 'sans-serif'],
      },
    },
  },
  plugins: [
    plugin(function ({ addVariant }) {
      addVariant('light', '.light &');
    }),
  ],
}
