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
   4. IoT SOIL SENSORS 🌱
════════════════════════════════════════════════════════════════ */
function SoilSensorPage({
  user,
  nav,
  toast
}) {
  const [selField, setSelField] = useState('field1');
  const sensors = {
    field1: [{
      label: 'Soil Moisture',
      val: 62,
      unit: '%',
      status: 'ok',
      color: 'var(--b3)',
      hist: [55, 58, 61, 65, 63, 60, 62],
      min: 0,
      max: 100,
      optimal: '55-75%'
    }, {
      label: 'Temperature',
      val: 24,
      unit: '°C',
      status: 'ok',
      color: 'var(--g3)',
      hist: [22, 23, 24, 25, 24, 23, 24],
      min: 0,
      max: 50,
      optimal: '20-28°C'
    }, {
      label: 'pH Level',
      val: 6.2,
      unit: 'pH',
      status: 'warn',
      color: 'var(--a2)',
      hist: [6.5, 6.4, 6.3, 6.2, 6.2, 6.1, 6.2],
      min: 0,
      max: 14,
      optimal: '6.0-7.0'
    }, {
      label: 'Nitrogen (N)',
      val: 38,
      unit: 'mg/kg',
      status: 'bad',
      color: 'var(--r2)',
      hist: [52, 48, 44, 41, 39, 37, 38],
      min: 0,
      max: 100,
      optimal: '50-80 mg/kg'
    }, {
      label: 'Phosphorus (P)',
      val: 28,
      unit: 'mg/kg',
      status: 'ok',
      color: 'var(--g4)',
      hist: [24, 25, 27, 26, 28, 27, 28],
      min: 0,
      max: 60,
      optimal: '20-40 mg/kg'
    }, {
      label: 'Potassium (K)',
      val: 145,
      unit: 'mg/kg',
      status: 'ok',
      color: 'var(--pu)',
      hist: [130, 135, 140, 142, 144, 143, 145],
      min: 0,
      max: 300,
      optimal: '120-200 mg/kg'
    }]
  };
  const data = sensors[selField] || sensors.field1;
  return <div className="wrap-md">
      <div style={{
      fontFamily: "'Baloo 2',cursive",
      fontSize: 26,
      fontWeight: 900,
      color: 'var(--g1)',
      marginBottom: 5
    }}>🌱 Soil Sensors</div>
      <div style={{
      fontSize: 13,
      color: 'var(--tx2)',
      marginBottom: 16
    }}>Real-time soil health monitoring • IoT sensors</div>

      {/* Field Selector */}
      <div style={{
      display: 'flex',
      gap: 8,
      marginBottom: 22
    }}>
        {[{
        k: 'field1',
        l: '🍅 Field 1 — Tomato'
      }, {
        k: 'field2',
        l: '🌾 Field 2 — Wheat'
      }, {
        k: 'field3',
        l: '🥔 Field 3 — Potato'
      }].map(f => <button key={f.k} onClick={() => setSelField(f.k)} style={{
        flex: 1,
        padding: '9px 8px',
        borderRadius: 9,
        fontSize: 12.5,
        fontWeight: 700,
        border: `2px solid ${selField === f.k ? 'var(--g4)' : 'var(--br)'}`,
        background: selField === f.k ? 'var(--gp)' : 'none',
        color: selField === f.k ? 'var(--g3)' : 'var(--tx2)',
        cursor: 'pointer',
        fontFamily: "'Outfit',sans-serif",
        transition: 'all .18s'
      }}>
            {f.l}
          </button>)}
      </div>

      {/* AI Recommendation Banner */}
      <div style={{
      padding: 16,
      background: 'linear-gradient(135deg,var(--rp),#fff5f5)',
      border: '2px solid var(--rpb)',
      borderRadius: 'var(--rad)',
      marginBottom: 22,
      display: 'flex',
      gap: 13,
      alignItems: 'flex-start'
    }}>
        <div style={{
        fontSize: 26,
        flexShrink: 0
      }}>🤖</div>
        <div>
          <div style={{
          fontSize: 14,
          fontWeight: 800,
          color: 'var(--r1)',
          marginBottom: 4
        }}>AI Soil Alert</div>
          <div style={{
          fontSize: 13,
          color: 'var(--tx2)',
          lineHeight: 1.65
        }}>
            <strong>Nitrogen deficiency detected!</strong> NPK 19:19:19 — 5kg/acre apply karein is hafte. pH bhi thoda low hai — agricultural lime 2kg/acre se correct karein.
          </div>
          <button className="btn btn-red btn-sm" style={{
          marginTop: 10
        }} onClick={() => nav('marketplace')}>🛒 Fertilizer Order Karo</button>
        </div>
      </div>

      {/* Sensor Cards Grid */}
      <div className="sensor-grid">
        {data.map((s, i) => {
        const pct = ((s.val - s.min) / (s.max - s.min) * 100).toFixed(0);
        return <div key={i} className={`sensor-card ${s.status}`}>
              <div className="sensor-lbl">{s.label}</div>
              <div style={{
            display: 'flex',
            alignItems: 'baseline',
            gap: 5
          }}>
                <div className="sensor-val" style={{
              color: s.color
            }}>{s.val}</div>
                <div className="sensor-unit">{s.unit}</div>
              </div>
              <div style={{
            fontSize: 11,
            color: 'var(--tx3)',
            marginTop: 2
          }}>Optimal: {s.optimal}</div>
              <div className="sensor-gauge">
                <div className="sensor-gauge-fill" style={{
              width: `${pct}%`,
              background: s.color
            }} />
              </div>
              <div className="sensor-hist">
                {s.hist.map((v, j) => {
              const hp = (v - s.min) / (s.max - s.min) * 100;
              return <div key={j} className="sh-bar" style={{
                height: `${Math.max(hp, 8)}%`,
                background: j === s.hist.length - 1 ? s.color : 'var(--br2)'
              }} />;
            })}
              </div>
              <div style={{
            fontSize: 10,
            color: 'var(--tx4)',
            marginTop: 3
          }}>Last 7 readings</div>
              {s.status === 'bad' && <div style={{
            marginTop: 7,
            fontSize: 11,
            fontWeight: 700,
            color: 'var(--r2)'
          }}>⚠️ Action needed!</div>}
              {s.status === 'warn' && <div style={{
            marginTop: 7,
            fontSize: 11,
            fontWeight: 700,
            color: 'var(--a2)'
          }}>⚡ Monitor closely</div>}
            </div>;
      })}
      </div>

      {/* Schedule Next Reading */}
      <div className="card" style={{
      padding: 18,
      marginTop: 6
    }}>
        <div style={{
        fontSize: 14,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 11
      }}>⏰ Sensor Schedule</div>
        {[['Last Reading', 'Aaj 8:30 AM', '✅'], ['Next Auto-Reading', 'Aaj 2:30 PM', '⏳'], ['Weekly Report', 'Sunday', '📊']].map(([l, v, i]) => <div key={l} style={{
        display: 'flex',
        justifyContent: 'space-between',
        padding: '8px 0',
        borderBottom: '1px solid var(--gp)',
        fontSize: 13
      }}>
            <span style={{
          color: 'var(--tx2)',
          fontWeight: 600
        }}>{i} {l}</span>
            <span style={{
          fontWeight: 800,
          color: 'var(--tx)'
        }}>{v}</span>
          </div>)}
        <button className="btn btn-g btn-sm" style={{
        width: '100%',
        marginTop: 12
      }} onClick={() => toast('Manual reading triggered! 2 min mein results aayenge', 'inf')}>
          🔄 Manual Reading Trigger Karo
        </button>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   5. B2B DATA INTELLIGENCE 💼
════════════════════════════════════════════════════════════════ */
