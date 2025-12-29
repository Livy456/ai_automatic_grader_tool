import type { Config } from 'tailwindcss';
// import { extend } from 'zod/v4/mini';


export default {
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        './src/app/**/*.{js,ts,jsx,tsx,mdx}',
        './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
        container: {
            center: true,
            padding: '2rem',
            screens:{
                sm: '1500px',
            },
        },
        extend: {
            colors: {
                background: "var(--backgroud)",
                primary: '#1a73e8',
             },
        },
    },
    plugins: [require("tailwindcss-animate")], // added this in to fix the globals.css issue
} satisfies Config;