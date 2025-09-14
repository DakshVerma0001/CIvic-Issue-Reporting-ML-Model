# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
- This component is tailored for a simple React app. You can mount it in an index.jsx:

import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css' // include Tailwind or your own styles

createRoot(document.getElementById('root')).render(<App />)

- Quick start (Vite):
  1. npm create vite@latest my-ui --template react
  2. copy this file as src/App.jsx
  3. install Tailwind or use simple CSS
  4. npm install
  5. npm run dev