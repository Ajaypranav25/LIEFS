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
        surface: {
          DEFAULT: '#0c1324',
          dim: '#0c1324',
          bright: '#33394c',
          variant: '#2e3447',
          lowest: '#070d1f',
          low: '#151b2d',
          base: '#191f31',
          high: '#23293c',
          highest: '#2e3447',
        },
        'on-surface': {
          DEFAULT: '#dce1fb',
          variant: '#bbcabf',
        },
        primary: {
          DEFAULT: '#4edea3',
          container: '#10b981',
          fixed: '#6ffbbe',
          'fixed-dim': '#4edea3',
        },
        secondary: {
          DEFAULT: '#4cd7f6',
          container: '#03b5d3',
          fixed: '#acedff',
          'fixed-dim': '#4cd7f6',
        },
        tertiary: {
          DEFAULT: '#c0c1ff',
          container: '#9699ff',
          fixed: '#e1e0ff',
        },
        outline: {
          DEFAULT: '#86948a',
          variant: '#3c4a42',
        },
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
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace', 'Menlo', 'Consolas'],
        'headline-lg': ['Inter', 'sans-serif'],
        'label-mono-lg': ['"JetBrains Mono"', 'monospace'],
        'label-mono-sm': ['"JetBrains Mono"', 'monospace'],
        'code-block': ['"JetBrains Mono"', 'monospace'],
      },
      spacing: {
        'stack-xs': '4px',
        'stack-sm': '8px',
        'stack-md': '16px',
        'stack-lg': '32px',
        margin: '32px',
        gutter: '24px',
        unit: '4px',
      },
      boxShadow: {
        'glow-emerald': '0 0 15px -3px rgba(78, 222, 163, 0.35)',
        'glow-cyan': '0 0 15px -3px rgba(76, 215, 246, 0.35)',
        'glow-active': '0 0 12px rgba(78, 222, 163, 0.25)',
      },
    },
  },
  plugins: [],
}
