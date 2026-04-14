/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: '#0a0d14',
        bg2: '#0f1420',
        surface: '#1a2035',
        border: '#2a3555',
        accent: '#4f8ef7',
        accent3: '#29d8a8',
        textMain: '#c8d4f0',
        textDim: '#6b7fa8',
        textBright: '#e8eeff',
      }
    },
  },
  plugins: [],
}