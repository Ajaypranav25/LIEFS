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
        background: '#0b0c10',
        surface: {
          DEFAULT: '#0b0c10',
          dim: '#0b0c10',
          lowest: '#07080a',
          low: '#0f1116',
          container: '#14171f',
          high: '#1c202a',
          highest: '#242936',
          bright: '#2c3242',
          variant: '#1f2430',
        },
        'on-background': '#f1f5f9',
        'on-surface': {
          DEFAULT: '#e2e8f0',
          variant: '#94a3b8',
        },
        primary: {
          DEFAULT: '#38bdf8',
          light: '#bae6fd',
          dim: '#0284c7',
          container: '#0369a1',
          'on-container': '#f0f9ff',
        },
        secondary: {
          DEFAULT: '#2dd4bf',
          light: '#99f6e4',
          container: '#134e4a',
          'on-container': '#ccfbf1',
        },
        tertiary: {
          DEFAULT: '#a78bfa',
          light: '#ddd6fe',
          container: '#4c1d95',
        },
        emerald: {
          DEFAULT: '#10b981',
          neon: '#34d399',
          dim: '#064e3b',
        },
        cyan: {
          DEFAULT: '#0ea5e9',
          neon: '#38bdf8',
          dim: '#0c4a6e',
        },
        outline: {
          DEFAULT: 'rgba(255, 255, 255, 0.12)',
          variant: 'rgba(255, 255, 255, 0.08)',
        },
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'Menlo', 'Consolas', 'monospace'],
      },
      boxShadow: {
        subtle: '0 1px 2px 0 rgba(0, 0, 0, 0.35)',
        card: '0 4px 20px -2px rgba(0, 0, 0, 0.5)',
        popover: '0 10px 30px -5px rgba(0, 0, 0, 0.8)',
        'glow-cyan': '0 2px 8px -2px rgba(56, 189, 248, 0.15)',
        'glow-emerald': '0 2px 8px -2px rgba(16, 185, 129, 0.15)',
        'glow-purple': '0 2px 8px -2px rgba(167, 139, 250, 0.15)',
        'glass-card': '0 4px 20px -2px rgba(0, 0, 0, 0.5)',
      },
    },
  },
  plugins: [],
}
