/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        obsidian: {
          950: '#020617',
          900: '#070d1f',
          850: '#0c1324',
          800: '#0f172a',
          750: '#151b2d',
          700: '#191f31',
          600: '#1e293b',
          500: '#2e3447',
          400: '#33394c',
        },
        emerald: {
          glow: '#10b981',
          neon: '#4edea3',
          bright: '#6ffbbe',
          dim: '#005236',
          deep: '#002113',
        },
        cyan: {
          glow: '#06b6d4',
          neon: '#4cd7f6',
          bright: '#acedff',
          dim: '#004e5c',
        },
        indigo: {
          glow: '#6366f1',
          neon: '#9699ff',
          bright: '#e1e0ff',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace', 'Menlo', 'Consolas'],
      },
      animation: {
        'cursor-blink': 'blink 1s step-start infinite',
        'pulse-glow': 'pulseGlow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 2s infinite linear',
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        pulseGlow: {
          '0%, 100%': { opacity: '0.4', transform: 'scale(1)' },
          '50%': { opacity: '0.8', transform: 'scale(1.02)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      boxShadow: {
        'glow-emerald': '0 0 15px -3px rgba(16, 185, 129, 0.3)',
        'glow-cyan': '0 0 15px -3px rgba(6, 182, 212, 0.3)',
        'glow-indigo': '0 0 15px -3px rgba(99, 102, 241, 0.3)',
      },
    },
  },
  plugins: [],
}


