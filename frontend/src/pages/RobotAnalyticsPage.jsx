import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import { API, saveSession, clearSession, loadSession } from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody } from '../utils/helpers.jsx';
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary } from '../utils/constants.jsx';

import BeejHealthApp from '../components/BeejHealthApp.jsx';
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
   ROBOT ANALYTICS 📊
════════════════════════════════════════════════════════════════ */
function RobotAnalyticsPage({
  nav,
  toast
}) {
  const [analytics, setAnalytics] = useState(null);
  useEffect(() => {
    API.get('/api/robots/analytics/summary').then(d => setAnalytics(d)).catch(() => {});
  }, []);
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const sprayData = analytics?.weeklyData?.map(d => d.spray / 150) || [1.2, 2.1, 1.8, 2.4, 1.9, 0.8, 2.3];
  const areaData = analytics?.weeklyData?.map(d => d.area) || [1.0, 1.8, 1.5, 2.0, 1.7, 0.6, 2.1];
  const maxSpray = Math.max(...sprayData);
  return <div className="rob-shell">
      <div className="rob-wrap">
        <div style={{
        display: 'flex',
        gap: 10,
        alignItems: 'center',
        marginBottom: 22,
        flexWrap: 'wrap'
      }}>
          <button className="rob-btn ghost" onClick={() => nav('robot-dashboard')}>← Back</button>
          <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 24,
          fontWeight: 900,
          color: 'var(--g3)'
        }}>📊 Robot Analytics</div>
        </div>

        {/* KPIs */}
        <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4,1fr)',
        gap: 14,
        marginBottom: 24
      }}>
          {[{
          n: analytics?.sprayVolume || '—',
          l: 'Chemical Used (Week)',
          c: 'var(--g3)',
          i: '💊'
        }, {
          n: analytics?.areaCovered || '—',
          l: 'Acres Covered',
          c: 'var(--g4)',
          i: '📐'
        }, {
          n: `${analytics?.totalFlights || 0} flights`,
          l: 'Total Flights',
          c: '#ffd700',
          i: '🛫'
        }, {
          n: `${analytics?.avgBattery || 0}%`,
          l: 'Avg Battery',
          c: '#b347ff',
          i: '🔋'
        }].map(s => <div key={s.l} className="rob-stat">
              <div style={{
            fontSize: 22,
            marginBottom: 6
          }}>{s.i}</div>
              <div className="rob-stat-n" style={{
            color: s.c
          }}>{s.n}</div>
              <div className="rob-stat-l">{s.l}</div>
            </div>)}
        </div>

        <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 18,
        marginBottom: 18
      }}>
          {/* Spray Chart */}
          <div className="rob-card" style={{
          padding: 20
        }}>
            <div style={{
            fontSize: 14,
            fontWeight: 700,
            color: 'rgba(0,0,0,.8)',
            marginBottom: 18,
            textTransform: 'uppercase',
            letterSpacing: '.6px'
          }}>Chemical Usage (L/day)</div>
            <div className="rob-chart-bar">
              {sprayData.map((v, i) => <div key={i} className="rcb-col">
                  <div style={{
                fontSize: 9,
                color: 'rgba(0,0,0,.6)',
                marginBottom: 3
              }}>{v}L</div>
                  <div className="rcb-bar" style={{
                height: `${v / maxSpray * 100}%`,
                background: `linear-gradient(180deg,var(--g3),#0088aa)`
              }} />
                  <div className="rcb-lbl">{days[i]}</div>
                </div>)}
            </div>
          </div>

          {/* Area Chart */}
          <div className="rob-card" style={{
          padding: 20
        }}>
            <div style={{
            fontSize: 14,
            fontWeight: 700,
            color: 'rgba(0,0,0,.8)',
            marginBottom: 18,
            textTransform: 'uppercase',
            letterSpacing: '.6px'
          }}>Area Covered (Acres/day)</div>
            <div className="rob-chart-bar">
              {areaData.map((v, i) => <div key={i} className="rcb-col">
                  <div style={{
                fontSize: 9,
                color: 'rgba(0,0,0,.6)',
                marginBottom: 3
              }}>{v}A</div>
                  <div className="rcb-bar" style={{
                height: `${v / Math.max(...areaData) * 100}%`,
                background: 'linear-gradient(180deg,var(--g4),#00aa66)'
              }} />
                  <div className="rcb-lbl">{days[i]}</div>
                </div>)}
            </div>
          </div>
        </div>

        {/* Per Robot Stats */}
        <div className="rob-card" style={{
        padding: 22,
        marginBottom: 18
      }}>
          <div style={{
          fontSize: 14,
          fontWeight: 700,
          color: 'rgba(0,0,0,.8)',
          marginBottom: 16,
          textTransform: 'uppercase',
          letterSpacing: '.7px'
        }}>Per Robot Performance</div>
          <div style={{
          overflowX: 'auto'
        }}>
            <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: 13
          }}>
              <thead>
                <tr style={{
                borderBottom: '1px solid rgba(0,0,0,.3)'
              }}>
                  {['Robot', 'Missions', 'Area (Acres)', 'Chemical (L)', 'Accuracy', 'Uptime'].map(h => <th key={h} style={{
                  padding: '8px 12px',
                  textAlign: 'left',
                  color: 'rgba(0,0,0,.6)',
                  fontWeight: 700,
                  fontSize: 11,
                  textTransform: 'uppercase',
                  letterSpacing: '.5px'
                }}>{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {[['🚁 DroneBot Alpha', '18', '8.2', '9.1', '97.4%', '94%'], ['🤖 GroundBot Beta', '12', '3.5', '4.3', '95.1%', '88%'], ['📡 SensorBot Delta', '—', '4.5 (scan)', '—', '99.8%', '99%']].map((row, i) => <tr key={i} style={{
                borderBottom: '1px solid rgba(0,0,0,.7)'
              }}>
                    {row.map((cell, j) => <td key={j} style={{
                  padding: '11px 12px',
                  color: j === 0 ? 'white' : 'rgba(0,0,0,.9)',
                  fontWeight: j === 0 ? 700 : 400
                }}>{cell}</td>)}
                  </tr>)}
              </tbody>
            </table>
          </div>
        </div>

        {/* AI Insights */}
        <div className="rob-card" style={{
        padding: 20,
        borderColor: 'rgba(77,189,122,.25)'
      }}>
          <div style={{
          fontSize: 14,
          fontWeight: 700,
          color: 'var(--g4)',
          marginBottom: 13
        }}>🤖 AI Fleet Insights</div>
          {[{
          i: '📈',
          'msg': 'Is hafte spray efficiency 12% improve hui — nozzle cleaning ka positive impact.'
        }, {
          i: '⚡',
          'msg': 'DroneBot Alpha ko Tuesday 6 AM mission pe deploy karein — weather optimal rahega.'
        }, {
          i: '💊',
          'msg': 'Mancozeb usage 23% zyada tha estimate se — Early Blight severity recalibrate karein.'
        }, {
          i: '🔋',
          'msg': 'DroneBot Gamma ka battery health 78% — 6 mahine mein replacement recommend.'
        }].map((ins, i) => <div key={i} style={{
          display: 'flex',
          gap: 10,
          padding: '10px 0',
          borderBottom: '1px solid rgba(0,0,0,.7)',
          fontSize: 13,
          color: 'rgba(0,0,0,.9)',
          lineHeight: 1.6
        }}>
              <span style={{
            fontSize: 16,
            flexShrink: 0
          }}>{ins.i}</span>
              <span>{ins.msg}</span>
            </div>)}
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   ROOT APP
════════════════════════════════════════════════════════════════ */
