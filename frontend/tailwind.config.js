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
        background: '#081326',
        surface: {
          DEFAULT: '#081326',
          dim: '#081326',
          lowest: '#040e21',
          low: '#111b2f',
          container: '#151f33',
          high: '#202a3e',
          highest: '#2b354a',
          bright: '#2f394e',
          variant: '#2b354a',
        },
        'on-background': '#d8e2fd',
        'on-surface': {
          DEFAULT: '#d8e2fd',
          variant: '#bec8ce',
        },
        primary: {
          DEFAULT: '#7dd3fc',
          light: '#c5eaff',
          dim: '#7bd1fa',
          container: '#0284c7',
          'on-container': '#e0f2fe',
        },
        secondary: {
          DEFAULT: '#4cd7f6',
          light: '#a0cde5',
          container: '#1f4e63',
          'on-container': '#92bed7',
        },
        tertiary: {
          DEFAULT: '#c084fc',
          light: '#f1ddff',
          container: '#ddbaff',
        },
        emerald: {
          DEFAULT: '#10b981',
          neon: '#4edea3',
          dim: '#005236',
        },
        cyan: {
          DEFAULT: '#06b6d4',
          neon: '#00e5ff',
          dim: '#004e5c',
        },
        outline: {
          DEFAULT: '#899298',
          variant: '#3f484e',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace', 'Menlo', 'Consolas'],
      },
      boxShadow: {
        'glow-cyan': '0 0 15px -2px rgba(125, 211, 252, 0.4)',
        'glow-emerald': '0 0 15px -2px rgba(78, 222, 163, 0.4)',
        'glow-purple': '0 0 15px -2px rgba(192, 132, 252, 0.4)',
        'glass-card': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
      },
    },
  },
  plugins: [],
}
