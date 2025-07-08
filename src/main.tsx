import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Tailwind styles (if tailwind configured)
import './index.css';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Root element #root not found');
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);