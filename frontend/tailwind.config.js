/** @type {import('tailwindcss').Config} */
export default { // Tailwind CSS config
  content: [ // Scan these files
    "./index.html", // Root HTML file
    "./src/**/*.{js,ts,jsx,tsx}", // All JS/TS source files
  ],
  theme: { // Design system theme
    extend: {}, // Custom theme extensions
  },
  plugins: [], // No additional plugins
}
