import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import { API, saveSession, clearSession, loadSession } from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody } from '../utils/helpers.jsx';
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary } from '../utils/constants.jsx';

import BeejHealthApp from '../components/BeejHealthApp.jsx';
import RobotAnalyticsPage from './RobotAnalyticsPage.jsx';
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
   BATTERY & MAINTENANCE 🔋
════════════════════════════════════════════════════════════════ */
function RobotMaintenancePage({
  nav,
  toast
}) {
  const [selBot, setSelBot] = useState('R01');
  const [maintData, setMaintData] = useState(null);
  useEffect(() => {
    API.get(`/api/robots/${selBot}/maintenance`).then(d => setMaintData(d)).catch(() => {});
  }, [selBot]);
  return <div className="rob-shell">
      <div className="rob-wrap-sm">
        <div style={{
        display: 'flex',
        gap: 10,
        alignItems: 'center',
        marginBottom: 22
      }}>
          <button className="rob-btn ghost" onClick={() => nav('robot-dashboard')}>← Back</button>
          <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 24,
          fontWeight: 900,
          color: 'var(--g3)'
        }}>🔋 Battery & Maintenance</div>
        </div>

        {/* Battery Overview */}
        <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2,1fr)',
        gap: 14,
        marginBottom: 24
      }}>
          {ROBOTS.map(r => ({
          ...r,
          battery: r.robotId === selBot && maintData ? maintData.battery?.level || r.battery : r.battery
        })).map(r => <div key={r.id || r.robotId} onClick={() => setSelBot(r.robotId || r.id)} className="rob-card" style={{
          padding: 18,
          cursor: 'pointer',
          borderColor: (r.robotId || r.id) === selBot ? 'rgba(30,126,66,.6)' : r.battery < 20 ? 'rgba(255,68,68,.4)' : r.battery < 50 ? 'rgba(255,215,0,.3)' : 'rgba(30,126,66,.2)'
        }}>
              <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 12
          }}>
                <div style={{
              display: 'flex',
              gap: 9,
              alignItems: 'center'
            }}>
                  <span style={{
                fontSize: 20
              }}>{r.emoji}</span>
                  <div>
                    <div style={{
                  fontSize: 13.5,
                  fontWeight: 800,
                  color: 'var(--tx)'
                }}>{r.name}</div>
                    <div style={{
                  fontSize: 11,
                  color: 'rgba(0,0,0,.6)'
                }}>{r.model}</div>
                  </div>
                </div>
                <span className={`rob-badge ${r.status}`}><span className={`rob-dot ${r.status}`} />{r.status}</span>
              </div>
              <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 12,
            marginBottom: 6
          }}>
                <span style={{
              color: 'rgba(0,0,0,.7)'
            }}>🔋 Battery</span>
                <span style={{
              fontWeight: 800,
              color: r.battery > 50 ? 'var(--g4)' : r.battery > 20 ? '#ffd700' : '#ff4444'
            }}>{r.battery}%</span>
              </div>
              <div className="rob-prog" style={{
            marginBottom: 10
          }}>
                <div className={`rob-prog-fill ${r.battery > 50 ? 'green' : r.battery > 20 ? 'yellow' : 'red'}`} style={{
              width: `${r.battery}%`
            }} />
              </div>
              <div style={{
            fontSize: 11.5,
            color: 'rgba(0,0,0,.9)'
          }}>
                {r.battery < 20 ? '⚠️ Charge immediately!' : r.battery < 50 ? '⚡ Charge soon' : '✅ Battery good'}
              </div>
              {r.status === 'offline' && r.battery < 20 && <button className="rob-btn primary" style={{
            width: '100%',
            marginTop: 10,
            fontSize: 12
          }} onClick={() => toast(`${r.name} charging initiated ⚡`)}>
                  ⚡ Start Charging
                </button>}
            </div>)}
        </div>

        {/* Maintenance Schedule */}
        <div className="rob-card" style={{
        padding: 22,
        marginBottom: 18
      }}>
          <div style={{
          fontSize: 15,
          fontWeight: 700,
          color: 'rgba(0,0,0,.8)',
          marginBottom: 16,
          textTransform: 'uppercase',
          letterSpacing: '.7px'
        }}>Maintenance Schedule</div>
          {[{
          robot: 'DroneBot Alpha',
          task: 'Propeller inspection',
          due: 'In 3 days',
          status: 'upcoming',
          color: '#ffd700'
        }, {
          robot: 'DroneBot Alpha',
          task: 'Spray nozzle clean',
          due: 'Today',
          status: 'urgent',
          color: '#ff4444'
        }, {
          robot: 'GroundBot Beta',
          task: 'Wheel bearing check',
          due: 'In 7 days',
          status: 'upcoming',
          color: '#ffd700'
        }, {
          robot: 'All Robots',
          task: 'Firmware update v2.4.1',
          due: 'Available now',
          status: 'available',
          color: 'var(--g3)'
        }, {
          robot: 'DroneBot Gamma',
          task: 'Battery replacement',
          due: 'Overdue',
          status: 'critical',
          color: '#ff4444'
        }, {
          robot: 'SensorBot Delta',
          task: 'Sensor calibration',
          due: 'In 14 days',
          status: 'ok',
          color: 'var(--g4)'
        }].map((m, i) => <div key={i} style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '12px 0',
          borderBottom: '1px solid rgba(0,0,0,.8)'
        }}>
              <div>
                <div style={{
              fontSize: 13.5,
              fontWeight: 700,
              color: 'var(--tx)'
            }}>{m.task}</div>
                <div style={{
              fontSize: 11.5,
              color: 'rgba(0,0,0,.6)',
              marginTop: 2
            }}>🤖 {m.robot}</div>
              </div>
              <div style={{
            textAlign: 'right'
          }}>
                <div style={{
              fontSize: 12,
              fontWeight: 700,
              color: m.color
            }}>{m.due}</div>
                <button className="rob-btn ghost" style={{
              marginTop: 6,
              padding: '4px 10px',
              fontSize: 11
            }} onClick={() => toast(`${m.task} — scheduled ✅`)}>
                  {m.status === 'available' ? 'Update Now' : 'Schedule'}
                </button>
              </div>
            </div>)}
        </div>

        {/* Alerts */}
        <div className="rob-card" style={{
        padding: 20,
        borderColor: 'rgba(255,68,68,.3)'
      }}>
          <div style={{
          fontSize: 14,
          fontWeight: 700,
          color: '#ff4444',
          marginBottom: 13
        }}>⚠️ Active Alerts</div>
          {(maintData?.alerts?.length > 0 ? maintData.alerts.map(a => ({
          l: a.msg,
          c: a.level === 'critical' ? '#ff4444' : a.level === 'warning' ? '#ffd700' : 'var(--g3)'
        })) : [{
          l: 'DroneBot Gamma battery critically low (12%) — charge immediately',
          c: '#ff4444'
        }, {
          l: 'DroneBot Alpha spray nozzle needs cleaning — affects spray quality',
          c: '#ffd700'
        }, {
          l: 'Firmware v2.4.1 available — performance improvements + bug fixes',
          c: 'var(--g3)'
        }]).map((a, i) => <div key={i} style={{
          display: 'flex',
          gap: 9,
          padding: '10px 0',
          borderBottom: '1px solid rgba(0,0,0,.7)',
          fontSize: 12.5
        }}>
              <div style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: a.c,
            flexShrink: 0,
            marginTop: 4
          }} />
              <span style={{
            color: 'rgba(0,0,0,.9)',
            lineHeight: 1.6
          }}>{a.l}</span>
            </div>)}
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   ROBOT ANALYTICS 📊
════════════════════════════════════════════════════════════════ */
