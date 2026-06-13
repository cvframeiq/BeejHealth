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
import MarketplacePage from './MarketplacePage.jsx';
import B2BPage from './B2BPage.jsx';
import SoilSensorPage from './SoilSensorPage.jsx';
import ForecastPage from './ForecastPage.jsx';
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
   2. SATELLITE FIELD MONITOR 🛰️
════════════════════════════════════════════════════════════════ */
function SatellitePage({
  user,
  nav,
  toast
}) {
  const [mapLoaded, setMapLoaded] = useState(false);
  const mapRef = useRef(null);
  useEffect(() => {
    // Load Leaflet dynamically
    const script = document.createElement('script');
    script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
    script.onload = () => {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
      document.head.appendChild(link);
      setTimeout(() => {
        if (mapRef.current && !mapRef.current._leaflet_id) {
          const map = window.L.map(mapRef.current).setView([18.52, 73.85], 12);
          window.L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
            maxZoom: 20,
            subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
            attribution: 'Google Satellite'
          }).addTo(map);
          // Add farm marker
          window.L.marker([18.52, 73.85]).addTo(map).bindPopup(`<b>${user?.name || 'My Farm'}</b><br>${user?.district || 'Pune'}`).openPopup();
          setMapLoaded(true);
        }
      }, 300);
    };
    document.head.appendChild(script);
    return () => {
      if (mapRef.current && mapRef.current._leaflet_id) mapRef.current._leaflet_id = null;
    };
  }, []);
  const [selField, setSelField] = useState(null);
  const [view, setView] = useState('ndvi');
  const fields = [{
    id: 1,
    name: 'Field 1',
    crop: '🍅 Tomato',
    acres: 2,
    ndvi: .61,
    status: 'warn',
    x: 15,
    y: 20,
    w: 34,
    h: 28,
    color: '#ffc940'
  }, {
    id: 2,
    name: 'Field 2',
    crop: '🌾 Wheat',
    acres: 1.5,
    ndvi: .78,
    status: 'ok',
    x: 55,
    y: 15,
    w: 28,
    h: 32,
    color: '#4dbd7a'
  }, {
    id: 3,
    name: 'Field 3',
    crop: '🥔 Potato',
    acres: 1,
    ndvi: .82,
    status: 'ok',
    x: 20,
    y: 58,
    w: 22,
    h: 26,
    color: '#7dd4a0'
  }];
  const viewColors = {
    'ndvi': 'NDVI Health',
    'rgb': 'True Color',
    'temp': 'Temperature'
  };
  return <div className="wrap-md">
      <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 22
    }}>
        <div>
          <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 26,
          fontWeight: 900,
          color: 'var(--g1)'
        }}>🛰️ Satellite Monitor</div>
          <div style={{
          fontSize: 13,
          color: 'var(--tx2)',
          marginTop: 3
        }}>ISRO Resourcesat + Sentinel-2 • Last updated: 2 hours ago</div>
        </div>
        <div style={{
        display: 'flex',
        gap: 7
      }}>
          {Object.keys(viewColors).map(v => <button key={v} onClick={() => setView(v)} style={{
          padding: '7px 14px',
          borderRadius: 8,
          fontSize: 12.5,
          fontWeight: 700,
          border: `2px solid ${view === v ? 'var(--g4)' : 'var(--br)'}`,
          background: view === v ? 'var(--gp)' : 'none',
          color: view === v ? 'var(--g3)' : 'var(--tx2)',
          cursor: 'pointer',
          fontFamily: "'Outfit',sans-serif"
        }}>
              {v.toUpperCase()}
            </button>)}
        </div>
      </div>

      {/* Satellite Map */}
      <div className="sat-map" onClick={() => setSelField(null)}>
        <div className="sat-grid" />
        <div className="sat-overlay">📡 {viewColors[view]} • Wagholi Farm</div>
        {fields.map(f => <div key={f.id} className="sat-field" onClick={e => {
        e.stopPropagation();
        setSelField(f);
      }} style={{
        left: `${f.x}%`,
        top: `${f.y}%`,
        width: `${f.w}%`,
        height: `${f.h}%`,
        background: f.color + '55',
        borderColor: selField?.id === f.id ? 'white' : f.color + '88'
      }}>
            <div style={{
          textAlign: 'center'
        }}>
              <div style={{
            fontSize: 14
          }}>{f.crop.split(' ')[0]}</div>
              <div style={{
            fontSize: 10,
            fontWeight: 700,
            color: 'white',
            marginTop: 2,
            textShadow: '0 1px 3px rgba(0,0,0,.8)'
          }}>{f.name}</div>
            </div>
          </div>)}
        {/* NDVI Legend */}
        <div style={{
        position: 'absolute',
        bottom: 12,
        right: 12,
        background: 'rgba(0,0,0,.7)',
        borderRadius: 8,
        padding: '8px 12px',
        backdropFilter: 'blur(8px)'
      }}>
          <div style={{
          fontSize: 10,
          color: 'rgba(255,255,255,.7)',
          marginBottom: 5,
          fontWeight: 600
        }}>NDVI Index</div>
          <div className="ndvi-bar" style={{
          width: 120
        }} />
          <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 9,
          color: 'rgba(255,255,255,.6)',
          marginTop: 2
        }}><span>0 Low</span><span>1 High</span></div>
        </div>
      </div>

      {/* Field Detail */}
      {selField ? <div className="card" style={{
      padding: 22,
      marginBottom: 18,
      border: `2px solid ${selField.color}`
    }} key={selField.id}>
          <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 14
      }}>
            <div style={{
          fontSize: 17,
          fontWeight: 900,
          color: 'var(--g1)'
        }}>{selField.crop} — {selField.name}</div>
            <span className={`badge ${selField.status === 'ok' ? 'bg-g' : 'bg-a'}`}>{selField.status === 'ok' ? '✅ Healthy' : '⚠️ Monitor'}</span>
          </div>
          <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3,1fr)',
        gap: 10,
        marginBottom: 14
      }}>
            {[['NDVI Score', (selField.ndvi * 100).toFixed(0) + '%', '🌿'], ['Area', selField.acres + ' Acres', '📐'], ['Last Scan', '2 hrs ago', '📡']].map(([l, v, i]) => <div key={l} style={{
          padding: 12,
          background: 'var(--gp)',
          borderRadius: 10,
          textAlign: 'center'
        }}>
                <div style={{
            fontSize: 18,
            marginBottom: 4
          }}>{i}</div>
                <div style={{
            fontSize: 16,
            fontWeight: 900,
            color: 'var(--g2)'
          }}>{v}</div>
                <div style={{
            fontSize: 11,
            color: 'var(--tx3)',
            marginTop: 2
          }}>{l}</div>
              </div>)}
          </div>
          <div className="ndvi-bar" style={{
        marginBottom: 8
      }}>
            <div className="ndvi-marker" style={{
          left: `${selField.ndvi * 100 - 2}%`
        }} />
          </div>
          {selField.status === 'warn' && <div style={{
        padding: 12,
        background: 'var(--ap)',
        borderRadius: 10,
        fontSize: 13,
        color: 'var(--a1)',
        fontWeight: 600,
        marginBottom: 12
      }}>
              ⚠️ NDVI {(selField.ndvi * 100).toFixed(0)}% — Below optimal (75%). Possible early stress detected. Inspection recommended.
            </div>}
          <div style={{
        display: 'flex',
        gap: 9
      }}>
            <button className="btn btn-ghost btn-sm" style={{
          flex: 1
        }} onClick={() => toast('Historical comparison loading...', 'inf')}>📊 History</button>
            <button className="btn btn-g btn-sm" style={{
          flex: 2
        }} onClick={() => nav('consultation')}>🔬 Consultation Book Karo →</button>
          </div>
        </div> : <div style={{
      padding: 14,
      background: 'var(--gp)',
      borderRadius: 10,
      fontSize: 13,
      color: 'var(--g2)',
      fontWeight: 600,
      marginBottom: 18,
      textAlign: 'center'
    }}>
          👆 Map pe koi field click karein details dekhne ke liye
        </div>}

      {/* NDVI Legend */}
      <div className="sat-legend">
        {[['#ef5350', 'Stress (0-40%)'], ['#ffc940', 'Monitor (40-65%)'], ['#4dbd7a', 'Healthy (65-80%)'], ['#1e7e42', 'Excellent (80%+)']].map(([c, l]) => <div key={l} className="sat-leg-item"><div className="sat-leg-dot" style={{
          background: c
        }} />{l}</div>)}
      </div>

      {/* All Fields Summary */}
      <div className="card" style={{
      padding: 20
    }}>
        <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 14
      }}>📊 Farm Summary</div>
        {fields.map(f => <div key={f.id} style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '10px 0',
        borderBottom: '1px solid var(--gp)',
        cursor: 'pointer'
      }} onClick={() => setSelField(f)}>
            <div style={{
          display: 'flex',
          gap: 10,
          alignItems: 'center'
        }}>
              <div style={{
            width: 10,
            height: 10,
            borderRadius: 2,
            background: f.color,
            flexShrink: 0
          }} />
              <div>
                <div style={{
              fontSize: 13.5,
              fontWeight: 700
            }}>{f.crop} — {f.name}</div>
                <div style={{
              fontSize: 11.5,
              color: 'var(--tx3)'
            }}>{f.acres} Acres</div>
              </div>
            </div>
            <div style={{
          textAlign: 'right'
        }}>
              <div style={{
            fontSize: 14,
            fontWeight: 800,
            color: f.status === 'ok' ? 'var(--g3)' : 'var(--a2)'
          }}>{(f.ndvi * 100).toFixed(0)}% NDVI</div>
              <span className={`badge ${f.status === 'ok' ? 'bg-g' : 'bg-a'}`} style={{
            fontSize: 10
          }}>{f.status === 'ok' ? '✅ OK' : '⚠️ Watch'}</span>
            </div>
          </div>)}
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   3. PREDICTIVE DISEASE FORECAST 📊
════════════════════════════════════════════════════════════════ */
