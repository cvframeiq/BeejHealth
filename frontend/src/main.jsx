import React from 'react';
import { createRoot } from 'react-dom/client';
import './i18n';
import BeejHealthApp from './BeejHealth.jsx';

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BeejHealthApp />
  </React.StrictMode>
);
