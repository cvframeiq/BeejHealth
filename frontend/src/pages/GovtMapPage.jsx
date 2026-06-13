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
   8. GOVT DISEASE SURVEILLANCE MAP 📈
════════════════════════════════════════════════════════════════ */
function GovtMapPage({
  nav,
  toast
}) {
  const [selDist, setSelDist] = useState(null);
  const [hoverDist, setHoverDist] = useState(null);
  const [filter, setFilter] = useState('all');
  const districts = [{
    id: 'pune',
    name: 'Pune',
    cases: 342,
    disease: 'Early Blight',
    top: 'Tomato',
    sev: 'high',
    x: 38,
    y: 52,
    w: 12,
    h: 10
  }, {
    id: 'nashik',
    name: 'Nashik',
    cases: 287,
    disease: 'Late Blight',
    top: 'Grape',
    sev: 'high',
    x: 28,
    y: 22,
    w: 11,
    h: 9
  }, {
    id: 'aurangabad',
    name: 'Aurangabad',
    cases: 198,
    disease: 'Bacterial Wilt',
    top: 'Cotton',
    sev: 'med',
    x: 52,
    y: 38,
    w: 10,
    h: 8
  }, {
    id: 'nagpur',
    name: 'Nagpur',
    cases: 156,
    disease: 'Leaf Rust',
    top: 'Wheat',
    sev: 'med',
    x: 72,
    y: 30,
    w: 10,
    h: 9
  }, {
    id: 'solapur',
    name: 'Solapur',
    cases: 89,
    disease: 'Powdery Mildew',
    top: 'Pomegranate',
    sev: 'low',
    x: 45,
    y: 65,
    w: 9,
    h: 8
  }, {
    id: 'kolhapur',
    name: 'Kolhapur',
    cases: 67,
    disease: 'Downy Mildew',
    top: 'Grape',
    sev: 'low',
    x: 22,
    y: 70,
    w: 8,
    h: 8
  }, {
    id: 'amravati',
    name: 'Amravati',
    cases: 201,
    disease: 'Bollworm',
    top: 'Cotton',
    sev: 'high',
    x: 62,
    y: 22,
    w: 9,
    h: 8
  }, {
    id: 'jalgaon',
    name: 'Jalgaon',
    cases: 134,
    disease: 'Fusarium',
    top: 'Banana',
    sev: 'med',
    x: 32,
    y: 12,
    w: 9,
    h: 8
  }];
  const sevColor = {
    high: '#ef5350',
    med: '#ffc940',
    low: '#4dbd7a'
  };
  const filtered = filter === 'all' ? districts : districts.filter(d => d.sev === filter);
  return <div className="wrap-md">
      <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      marginBottom: 18,
      flexWrap: 'wrap',
      gap: 12
    }}>
        <div>
          <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 26,
          fontWeight: 900,
          color: 'var(--b1)'
        }}>📈 Disease Surveillance</div>
          <div style={{
          fontSize: 13,
          color: 'var(--tx2)',
          marginTop: 3
        }}>Maharashtra — Government Portal • Real-time data</div>
        </div>
        <div style={{
        display: 'flex',
        gap: 7
      }}>
          {['all', 'high', 'med', 'low'].map(f => <button key={f} onClick={() => setFilter(f)} style={{
          padding: '7px 13px',
          borderRadius: 8,
          fontSize: 12.5,
          fontWeight: 700,
          border: `2px solid ${filter === f ? 'var(--b3)' : 'var(--br)'}`,
          background: filter === f ? 'var(--bp)' : 'none',
          color: filter === f ? 'var(--b3)' : 'var(--tx2)',
          cursor: 'pointer',
          fontFamily: "'Outfit',sans-serif",
          transition: 'all .18s'
        }}>
              {f === 'all' ? 'All' : f === 'high' ? '🔴 High' : f === 'med' ? '🟡 Med' : '🟢 Low'}
            </button>)}
        </div>
      </div>

      {/* State Summary Cards */}
      <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(4,1fr)',
      gap: 12,
      marginBottom: 20
    }}>
        {[{
        l: 'Total Cases',
        v: '1,574',
        i: '🦠',
        c: 'var(--r2)'
      }, {
        l: 'Districts Affected',
        v: '22/36',
        i: '📍',
        c: 'var(--b3)'
      }, {
        l: 'High Risk Areas',
        v: '8',
        i: '🔴',
        c: 'var(--r2)'
      }, {
        l: 'Farmers Alerted',
        v: '47K+',
        i: '👨‍🌾',
        c: 'var(--g3)'
      }].map(s => <div key={s.l} style={{
        padding: 16,
        borderRadius: 'var(--rad)',
        background: 'white',
        border: '1.5px solid var(--br)',
        textAlign: 'center',
        boxShadow: 'var(--sh)'
      }}>
            <div style={{
          fontSize: 22,
          marginBottom: 5
        }}>{s.i}</div>
            <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 22,
          fontWeight: 900,
          color: s.c
        }}>{s.v}</div>
            <div style={{
          fontSize: 11,
          color: 'var(--tx3)',
          fontWeight: 600,
          marginTop: 3
        }}>{s.l}</div>
          </div>)}
      </div>

      {/* Map */}
      <div className="gov-map">
        <div style={{
        position: 'absolute',
        inset: 0,
        background: 'linear-gradient(160deg,#e8f4fd,#dbedf8)'
      }} />
        <div style={{
        position: 'absolute',
        top: 10,
        left: 12,
        fontSize: 13,
        fontWeight: 700,
        color: 'var(--b2)',
        opacity: .7
      }}>Maharashtra State Map</div>
        {filtered.map(d => <div key={d.id} className="map-district" style={{
        left: `${d.x}%`,
        top: `${d.y}%`,
        width: `${d.w}%`,
        height: `${d.h}%`,
        background: sevColor[d.sev] + (selDist?.id === d.id ? 'ee' : '99'),
        zIndex: selDist?.id === d.id ? 10 : 1
      }} onClick={() => setSelDist(selDist?.id === d.id ? null : d)} onMouseEnter={() => setHoverDist(d)} onMouseLeave={() => setHoverDist(null)}>
            {d.name}
            {hoverDist?.id === d.id && <div className="map-tooltip" style={{
          bottom: '110%',
          left: '50%',
          transform: 'translateX(-50%)'
        }}>
                {d.cases} cases • {d.disease}
              </div>}
          </div>)}
        <div style={{
        position: 'absolute',
        bottom: 12,
        right: 12,
        display: 'flex',
        gap: 7,
        background: 'rgba(255,255,255,.85)',
        padding: '8px 12px',
        borderRadius: 8,
        backdropFilter: 'blur(6px)'
      }}>
          {[['#ef5350', 'High'], ['#ffc940', 'Medium'], ['#4dbd7a', 'Low']].map(([c, l]) => <div key={l} style={{
          display: 'flex',
          alignItems: 'center',
          gap: 5,
          fontSize: 11,
          fontWeight: 600
        }}>
              <div style={{
            width: 10,
            height: 10,
            borderRadius: 2,
            background: c
          }} />{l}
            </div>)}
        </div>
      </div>

      {/* District Detail */}
      {selDist && <div className="card" style={{
      padding: 22,
      marginBottom: 18,
      border: `2px solid ${sevColor[selDist.sev]}`
    }}>
          <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        marginBottom: 14
      }}>
            <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 20,
          fontWeight: 900,
          color: 'var(--b1)'
        }}>{selDist.name} District</div>
            <span className={`badge ${selDist.sev === 'high' ? 'bg-r' : selDist.sev === 'med' ? 'bg-a' : 'bg-g'}`}>{selDist.sev === 'high' ? '🔴 High Risk' : selDist.sev === 'med' ? '🟡 Medium' : '🟢 Low Risk'}</span>
          </div>
          <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3,1fr)',
        gap: 10,
        marginBottom: 14
      }}>
            {[['Total Cases', selDist.cases, '🦠'], ['Main Disease', selDist.disease, '🔬'], ['Top Crop', selDist.top, '🌾']].map(([l, v, i]) => <div key={l} style={{
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
            fontSize: 14,
            fontWeight: 900,
            color: 'var(--g1)'
          }}>{v}</div>
                <div style={{
            fontSize: 11,
            color: 'var(--tx3)',
            marginTop: 2
          }}>{l}</div>
              </div>)}
          </div>
          <div style={{
        display: 'flex',
        gap: 9
      }}>
            <button className="btn btn-ghost btn-sm" style={{
          flex: 1
        }} onClick={() => toast('Full district report download ho raha hai...', 'inf')}>📊 Full Report</button>
            <button className="btn btn-b btn-sm" style={{
          flex: 2
        }} onClick={() => toast('Advisory bheja ja raha hai district farmers ko...', 'inf')}>📢 Send Advisory to Farmers</button>
          </div>
        </div>}

      {/* Outbreak List */}
      <div style={{
      fontSize: 16,
      fontWeight: 800,
      color: 'var(--b1)',
      marginBottom: 13
    }}>🚨 Active Outbreaks</div>
      {filtered.sort((a, b) => b.cases - a.cases).map(d => <div key={d.id} className="outbreak-row" onClick={() => setSelDist(d)}>
          <div className="outbreak-sev" style={{
        background: sevColor[d.sev] + '22'
      }}>
            <span style={{
          fontSize: 17
        }}>{d.sev === 'high' ? '🔴' : d.sev === 'med' ? '🟡' : '🟢'}</span>
          </div>
          <div style={{
        flex: 1
      }}>
            <div style={{
          fontSize: 14,
          fontWeight: 800,
          color: 'var(--tx)'
        }}>{d.name} — {d.disease}</div>
            <div style={{
          fontSize: 12.5,
          color: 'var(--tx3)',
          marginTop: 2
        }}>Top crop: {d.top} • {d.cases} cases reported</div>
          </div>
          <div style={{
        textAlign: 'right'
      }}>
            <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 18,
          fontWeight: 900,
          color: sevColor[d.sev]
        }}>{d.cases}</div>
            <div style={{
          fontSize: 11,
          color: 'var(--tx3)'
        }}>cases</div>
          </div>
        </div>)}
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   ROBOT DASHBOARD 🤖
════════════════════════════════════════════════════════════════ */
