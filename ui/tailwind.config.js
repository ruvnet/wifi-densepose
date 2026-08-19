/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './app/**/*.js'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Brand teal (existing theme-color #21808d) extended to a full ramp.
        brand: {
          50: '#ecfeff',
          100: '#cffafe',
          200: '#a5f3fc',
          300: '#67e8f9',
          400: '#22d3ee',
          500: '#21808d',
          600: '#1a6772',
          700: '#155059',
          800: '#123f47',
          900: '#0f3138',
        },
        // Neutral surface ramp (dark-first operator console).
        ink: {
          0: '#0b0f12',
          1: '#11161b',
          2: '#171e25',
          3: '#1f2831',
          4: '#2a353f',
          muted: '#8a9aa8',
          soft: '#b6c2cd',
          fg: '#e8eef3',
        },
        ok: '#22c55e',
        warn: '#f59e0b',
        bad: '#ef4444',
      },
      fontFamily: {
        sans: ['system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 2px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.25)',
      },
      borderRadius: {
        xl2: '1rem',
      },
    },
  },
  plugins: [],
};
