/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: {
                    DEFAULT: '#2E7D32', // Deep Farm Green
                    light: '#4CAF50',
                    dark: '#1B5E20',
                },
                secondary: {
                    DEFAULT: '#6D4C41', // Soil Brown
                    light: '#8D6E63',
                },
                accent: {
                    DEFAULT: '#FFD54F', // Harvest Yellow
                },
                background: '#F7F3EE', // Paper Beige
                surface: '#FFFFFF',
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
