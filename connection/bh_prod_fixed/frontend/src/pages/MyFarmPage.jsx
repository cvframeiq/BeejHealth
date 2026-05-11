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
import SatellitePage from './SatellitePage.jsx';
import VoiceInputPage from './VoiceInputPage.jsx';
import SettingsPage from './SettingsPage.jsx';
import EarningsPage from './EarningsPage.jsx';
import ProfilePage from './ProfilePage.jsx';
import SupportPage from './SupportPage.jsx';
import NotifPage from './NotifPage.jsx';
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

export default function MyFarmPage({
  user,
  nav,
  toast
}) {
  const [farmConsults, setFarmConsults] = useState([]);
  const [loadingFarm, setLoadingFarm] = useState(true);
  useEffect(() => {
    if (!user) return;
    API.get('/api/consultations').then(d => {
      setFarmConsults(d.consultations || []);
      setLoadingFarm(false);
    }).catch(() => setLoadingFarm(false));
  }, [user]);

  // Derive farm health from consultations
  const avgConf = farmConsults.length > 0 ? Math.round(farmConsults.reduce((s, c) => s + (100 - c.confidence * 0.3), 0) / farmConsults.length) : 78;
  const farmHealth = Math.min(99, Math.max(40, avgConf));
  const months = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov'];
  const health = [98, 94, 85, 78, 72];
  return <div className="wrap">
      <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      marginBottom: 26
    }}>
        <div>
          <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 28,
          fontWeight: 900,
          color: 'var(--g1)'
        }}>🗺️ My Farm — Digital Twin</div>
          <div style={{
          fontSize: 14,
          color: 'var(--tx2)',
          marginTop: 5
        }}>📍 {user?.district || 'Pune'}, Maharashtra • 4.5 Acres</div>
        </div>
        <button className="btn btn-g btn-sm">+ Add Field</button>
      </div>
      <div className="farm-map">
        <div style={{
        fontSize: 56,
        opacity: .25
      }}>🗺️</div>
        <div className="farm-mark" style={{
        left: 80,
        top: 80
      }}>Field 1 — 🍅 Tomato (2 ac)</div>
        <div className="farm-mark" style={{
        right: 80,
        top: 100
      }}>Field 2 — 🌾 Wheat (1.5 ac)</div>
        <div className="farm-mark" style={{
        left: '38%',
        bottom: 55
      }}>Field 3 — 🥔 Potato (1 ac)</div>
      </div>
      <div className="farm-stats">
        {[['4.5 ac', 'Total Area'], ['78/100', 'Health Score'], ['3', 'Active Crops'], ['₹13,524', 'Season Cost']].map(([n, l]) => <div key={l} className="fs-item"><div className="fs-n">{n}</div><div className="fs-l">{l}</div></div>)}
      </div>
      <div className="dash-2">
        <div className="card" style={{
        padding: 20
      }}>
          <div style={{
          fontSize: 15,
          fontWeight: 800,
          color: 'var(--g1)'
        }}>📊 Tomato Health Timeline</div>
          <div style={{
          fontSize: 12,
          color: 'var(--tx3)',
          marginTop: 2,
          marginBottom: 10
        }}>Last 5 months</div>
          <div className="tl-bars">
            {health.map((h, i) => <div key={i} className="tl-bw">
                <div style={{
              fontSize: 11,
              color: h > 85 ? 'var(--g4)' : h > 70 ? 'var(--a2)' : 'var(--r2)',
              fontWeight: 700,
              marginBottom: 3
            }}>{h}%</div>
                <div className="tl-bar" style={{
              height: `${h * .72}px`,
              background: h > 85 ? 'var(--g5)' : h > 70 ? 'var(--a2)' : 'var(--r3)',
              opacity: .85
            }} />
                <div className="tl-bl">{months[i]}</div>
              </div>)}
          </div>
        </div>
        <div className="card" style={{
        padding: 20
      }}>
          <div style={{
          fontSize: 15,
          fontWeight: 800,
          color: 'var(--g1)',
          marginBottom: 14
        }}>📋 Active Tasks</div>
          {[['✅', 'Oct 24: Mancozeb spray done'], ['⏳', 'Oct 31: Follow-up photo bhejo'], ['📅', 'Nov 5: Second consultation'], ['💊', 'Nov 7: Fertilizer application']].map(([i, t]) => <div key={t} style={{
          display: 'flex',
          gap: 11,
          padding: '9px 0',
          borderBottom: '1px solid var(--gp)',
          alignItems: 'flex-start'
        }}>
              <span style={{
            fontSize: 18,
            flexShrink: 0
          }}>{i}</span>
              <span style={{
            fontSize: 13.5,
            color: 'var(--tx)',
            fontWeight: 600
          }}>{t}</span>
            </div>)}
          <button className="btn btn-ghost btn-sm" style={{
          width: '100%',
          marginTop: 12
        }}>+ Task Add Karo</button>
        </div>
      </div>
      <div className="card" style={{
      padding: 22,
      marginTop: 20
    }}>
        <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 14
      }}>💰 Season Cost Tracker</div>
        <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(5,1fr)',
        gap: 11
      }}>
          {[['Seeds', '₹4,500'], ['Fertilizer', '₹6,200'], ['Medicine', '₹1,840'], ['Consultations', '₹984'], ['Labour', '₹8,000']].map(([k, v]) => <div key={k} style={{
          textAlign: 'center',
          padding: 13,
          background: 'var(--gp)',
          borderRadius: 10
        }}>
              <div style={{
            fontSize: 15,
            fontWeight: 800,
            color: 'var(--g3)'
          }}>{v}</div>
              <div style={{
            fontSize: 12,
            color: 'var(--tx3)',
            marginTop: 2
          }}>{k}</div>
            </div>)}
        </div>
        <div style={{
        marginTop: 14,
        padding: '13px 16px',
        background: 'linear-gradient(135deg,var(--g3),var(--g1))',
        borderRadius: 10,
        color: 'white',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
          <span style={{
          fontWeight: 700
        }}>Expected Revenue:</span>
          <span style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 20,
          fontWeight: 900
        }}>₹85,000</span>
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   NOTIFICATIONS PAGE
════════════════════════════════════════════════════════════════ */
