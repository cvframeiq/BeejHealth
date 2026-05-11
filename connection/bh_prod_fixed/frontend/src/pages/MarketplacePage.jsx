import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import { API, saveSession, clearSession, loadSession } from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody } from '../utils/helpers.jsx';
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary } from '../utils/constants.jsx';

import BeejHealthApp from '../components/BeejHealthApp.jsx';
import RobotAnalyticsPage from './RobotAnalyticsPage.jsx';
import RobotMaintenancePage from './RobotMaintenancePage.jsx';
import RobotControlPage from './RobotControlPage.jsx';
import RobotMapPage from './RobotMapPage.jsx';
import RobotCameraPage from './RobotCameraPage.jsx';
import RobotSprayPage from './RobotSprayPage.jsx';
import RobotDashboard from './RobotDashboard.jsx';
import GovtMapPage from './GovtMapPage.jsx';
import InsurancePage from './InsurancePage.jsx';
import B2BPage from './B2BPage.jsx';
import SoilSensorPage from './SoilSensorPage.jsx';
import ForecastPage from './ForecastPage.jsx';
import SatellitePage from './SatellitePage.jsx';
import VoiceInputPage from './VoiceInputPage.jsx';
import SettingsPage from './SettingsPage.jsx';
import EarningsPage from './EarningsPage.jsx';
import ProfilePage from './ProfilePage.jsx';
import SupportPage from './SupportPage.jsx';
import NotifPage from './NotifPage.jsx';
import MyFarmPage from './MyFarmPage.jsx';
import CaseDetailPage from './CaseDetailPage.jsx';
import ChatPage from './ChatPage.jsx';
import BookingPage from './BookingPage.jsx';
import ExpertsPage from './ExpertsPage.jsx';
import FarmerDash from './FarmerDash.jsx';
import ExpertDash from './ExpertDash.jsx';
import MyConsultPage from './MyConsultPage.jsx';
import AIReportPage from './AIReportPage.jsx';
import ConsultPage from './ConsultPage.jsx';
import HomePage from './HomePage.jsx';
import ExpertOnboarding from '../components/ExpertOnboarding.jsx';
import FarmerOnboarding from '../components/FarmerOnboarding.jsx';
import AuthModal from '../components/AuthModal.jsx';

export default /* ════════════════════════════════════════════════════════════════
   6. INPUT MARKETPLACE 📦
════════════════════════════════════════════════════════════════ */
function MarketplacePage({
  user,
  nav,
  toast
}) {
  // Static market prices (updated weekly from Agmarknet)
  // Replace with live API: https://agmarknet.gov.in or data.gov.in
  const MARKET_PRICES = [{
    crop: 'Tomato🍅',
    price: '₹18-28/kg',
    trend: '+12%',
    market: 'Pune APMC',
    updated: 'Aaj'
  }, {
    crop: 'Wheat🌾',
    price: '₹22-26/kg',
    trend: '-3%',
    market: 'Delhi Mandi',
    updated: 'Aaj'
  }, {
    crop: 'Potato🥔',
    price: '₹12-18/kg',
    trend: '+8%',
    market: 'Agra Mandi',
    updated: 'Kal'
  }, {
    crop: 'Cotton🌸',
    price: '₹55-65/kg',
    trend: '+5%',
    market: 'Akola APMC',
    updated: 'Aaj'
  }, {
    crop: 'Onion🧅',
    price: '₹15-22/kg',
    trend: '-8%',
    market: 'Nashik APMC',
    updated: 'Aaj'
  }, {
    crop: 'Coconut🥥',
    price: '₹28-40/piece',
    trend: '+15%',
    market: 'Palakkad',
    updated: 'Kal'
  }, {
    crop: 'Soybean🫘',
    price: '₹42-48/kg',
    trend: '+2%',
    market: 'Indore APMC',
    updated: 'Aaj'
  }, {
    crop: 'Corn🌽',
    price: '₹18-24/kg',
    trend: '+6%',
    market: 'Nizamabad',
    updated: 'Aaj'
  }, {
    crop: 'Sugarcane',
    price: '₹3.5-4.2/kg',
    trend: '+1%',
    market: 'UP Govt Rate',
    updated: 'Mahina'
  }, {
    crop: 'Grape🍇',
    price: '₹35-55/kg',
    trend: '+20%',
    market: 'Nashik APMC',
    updated: 'Kal'
  }, {
    crop: 'Apple🍎',
    price: '₹80-150/kg',
    trend: '-5%',
    market: 'Delhi Market',
    updated: 'Aaj'
  }, {
    crop: 'Rice/Paddy',
    price: '₹22-28/kg',
    trend: '+3%',
    market: 'Punjab Mandi',
    updated: 'Aaj'
  }];
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('crop');
  const [cart, setCart] = useState([]);
  const [cat, setCat] = useState('all');
  const products = [{
    id: 1,
    name: 'Mancozeb 75% WP',
    type: 'Fungicide',
    price: 280,
    unit: 'kg',
    emoji: '🧪',
    ai: true,
    rating: 4.8,
    stock: 'In Stock',
    desc: 'Early & Late Blight'
  }, {
    id: 2,
    name: 'Copper Oxychloride',
    type: 'Fungicide',
    price: 320,
    unit: 'kg',
    emoji: '⚗️',
    ai: false,
    rating: 4.6,
    stock: 'In Stock',
    desc: 'Bacterial diseases'
  }, {
    id: 3,
    name: 'Neem Oil Extract',
    type: 'Organic',
    price: 180,
    unit: 'L',
    emoji: '🌿',
    ai: false,
    rating: 4.5,
    stock: 'In Stock',
    desc: 'Eco-friendly option'
  }, {
    id: 4,
    name: 'NPK 19:19:19',
    type: 'Fertilizer',
    price: 680,
    unit: 'kg',
    emoji: '🌱',
    ai: true,
    rating: 4.9,
    stock: 'In Stock',
    desc: 'Balanced nutrition'
  }, {
    id: 5,
    name: 'Imidacloprid 17.8%',
    type: 'Insecticide',
    price: 420,
    unit: '100ml',
    emoji: '🐛',
    ai: false,
    rating: 4.4,
    stock: 'Low Stock',
    desc: 'Sucking pests control'
  }, {
    id: 6,
    name: 'BT Hybrid Tomato Seeds',
    type: 'Seeds',
    price: 850,
    unit: '50g',
    emoji: '🍅',
    ai: true,
    rating: 4.9,
    stock: 'In Stock',
    desc: 'Disease resistant variety'
  }, {
    id: 7,
    name: 'Potassium Humate',
    type: 'Fertilizer',
    price: 540,
    unit: 'kg',
    emoji: '💧',
    ai: false,
    rating: 4.7,
    stock: 'In Stock',
    desc: 'Soil conditioner'
  }, {
    id: 8,
    name: 'Trichoderma Viride',
    type: 'Bio-pesticide',
    price: 240,
    unit: 'kg',
    emoji: '🦠',
    ai: false,
    rating: 4.6,
    stock: 'In Stock',
    desc: 'Soil-borne disease control'
  }, {
    id: 9,
    name: 'Chlorpyrifos 20% EC',
    type: 'Insecticide',
    price: 380,
    unit: '500ml',
    emoji: '🧴',
    ai: false,
    rating: 4.3,
    stock: 'In Stock',
    desc: 'Broad-spectrum insect control'
  }];
  const cats = ['all', 'Fungicide', 'Fertilizer', 'Seeds', 'Insecticide', 'Organic', 'Bio-pesticide'];
  const filtered = products.filter(p => (cat === 'all' || p.type === cat) && (p.name.toLowerCase().includes(search.toLowerCase()) || p.desc.toLowerCase().includes(search.toLowerCase())));
  const addCart = p => {
    setCart(c => [...c.filter(x => x.id !== p.id), {
      ...p,
      qty: 1
    }]);
    toast(`${p.name} cart mein add kiya ✅`);
  };
  return <div className="wrap">
      <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      marginBottom: 18
    }}>
        <div>
          <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 26,
          fontWeight: 900,
          color: 'var(--g1)'
        }}>📦 AgriMart</div>
          <div style={{
          fontSize: 13,
          color: 'var(--tx2)',
          marginTop: 3
        }}>AI-recommended medicines, seeds, fertilizers • Village delivery</div>
        </div>
        {cart.length > 0 && <div style={{
        padding: '9px 18px',
        background: 'var(--g3)',
        color: 'white',
        borderRadius: 10,
        fontSize: 14,
        fontWeight: 700,
        cursor: 'pointer'
      }} onClick={() => toast(`Cart mein ${cart.length} items — Checkout coming soon!`, 'inf')}>🛒 Cart ({cart.length})</div>}
      </div>

      {/* AI Banner */}
      <div style={{
      padding: 15,
      background: 'linear-gradient(135deg,var(--gp),var(--gpb))',
      border: '1.5px solid var(--br2)',
      borderRadius: 'var(--rad)',
      marginBottom: 20,
      display: 'flex',
      gap: 12,
      alignItems: 'center'
    }}>
        <span style={{
        fontSize: 26
      }}>🤖</span>
        <div>
          <div style={{
          fontSize: 13.5,
          fontWeight: 800,
          color: 'var(--g1)'
        }}>AI Recommended for You</div>
          <div style={{
          fontSize: 12.5,
          color: 'var(--tx2)'
        }}>Aapki tomato Early Blight case ke hisaab se — Mancozeb 75% WP + NPK 19:19:19 recommended hai</div>
        </div>
      </div>

      {/* Search + Filter */}
      <div style={{
      display: 'flex',
      gap: 11,
      marginBottom: 18,
      flexWrap: 'wrap'
    }}>
        <input className="finp" style={{
        flex: 1,
        minWidth: 200
      }} placeholder="🔍 Medicine ya fertilizer search karein..." value={search} onChange={e => setSearch(e.target.value)} />
        <div style={{
        display: 'flex',
        gap: 6,
        flexWrap: 'wrap'
      }}>
          {cats.map(c => <button key={c} onClick={() => setCat(c)} style={{
          padding: '8px 13px',
          borderRadius: 8,
          fontSize: 12.5,
          fontWeight: 700,
          border: `2px solid ${cat === c ? 'var(--g4)' : 'var(--br)'}`,
          background: cat === c ? 'var(--gp)' : 'none',
          color: cat === c ? 'var(--g3)' : 'var(--tx2)',
          cursor: 'pointer',
          fontFamily: "'Outfit',sans-serif",
          whiteSpace: 'nowrap'
        }}>
              {c}
            </button>)}
        </div>
      </div>

      {/* Products */}
      <div className="mkt-grid">
        {filtered.map(p => <div key={p.id} className="mkt-card">
            <div className="mkt-img">{p.emoji}</div>
            <div className="mkt-body">
              {p.ai && <div className="mkt-ai">🤖 AI Recommended</div>}
              <div className="mkt-nm">{p.name}</div>
              <div className="mkt-type">{p.type} • ⭐ {p.rating}</div>
              <div style={{
            fontSize: 12,
            color: 'var(--tx3)',
            marginBottom: 8
          }}>{p.desc}</div>
              <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 10
          }}>
                <div className="mkt-pr">₹{p.price}<span style={{
                fontFamily: 'Outfit',
                fontSize: 11,
                color: 'var(--tx3)',
                fontWeight: 500
              }}>/{p.unit}</span></div>
                <span style={{
              fontSize: 11,
              fontWeight: 700,
              color: p.stock === 'In Stock' ? 'var(--g4)' : 'var(--a2)'
            }}>{p.stock === 'In Stock' ? '✅' : '⚠️'} {p.stock}</span>
              </div>
              <button className="btn btn-g btn-sm" style={{
            width: '100%'
          }} onClick={() => addCart(p)}>🛒 Cart Mein Add</button>
            </div>
          </div>)}
      </div>

      {cart.length > 0 && <div className="cart-badge" onClick={() => toast(`Cart: ${cart.length} items — Checkout coming soon!`, 'inf')}>
          🛒 {cart.length} items • ₹{cart.reduce((a, b) => a + b.price, 0)} — Checkout →
        </div>}
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   7. INSURANCE CLAIM 🏦
════════════════════════════════════════════════════════════════ */
