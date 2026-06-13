import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import { API, saveSession, clearSession, loadSession } from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody } from '../utils/helpers.jsx';
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary } from '../utils/constants.jsx';

import BeejHealthApp from '../components/BeejHealthApp.jsx';
import RobotAnalyticsPage from './RobotAnalyticsPage.jsx';
import RobotMaintenancePage from './RobotMaintenancePage.jsx';
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
   MANUAL ROBOT CONTROL 🎮
════════════════════════════════════════════════════════════════ */
function RobotControlPage({
  nav,
  toast
}) {
  const [pressed, setPressed] = useState({});
  const [speed, setSpeed] = useState(50);
  const [sprayOn, setSprayOn] = useState(false);
  const [camTilt, setCamTilt] = useState(0);
  const [selRobot, setSelRobot] = useState('R01');
  const [altitude, setAltitude] = useState(8);
  const [cmdLog, setCmdLog] = useState([]);
  const sendCmd = async (command, params = {}) => {
    try {
      const res = await API.post(`/api/robots/${selRobot}/command`, {
        command,
        params
      });
      const entry = `${new Date().toLocaleTimeString('en-IN', {
        hour: '2-digit',
        minute: '2-digit'
      })} ← ${command}`;
      setCmdLog(p => [entry, ...p].slice(0, 8));
      if (res.acknowledged) toast(`${command} sent ✅`, 'inf');
    } catch (e) {
      toast(`Command fail: ${e.message}`, 'err');
    }
  };
  const press = key => setPressed(p => ({
    ...p,
    [key]: true
  }));
  const release = key => setPressed(p => ({
    ...p,
    [key]: false
  }));
  const dirs = [{
    k: 'up',
    l: '↑',
    r: 0,
    c: 1
  }, {
    k: 'left',
    l: '←',
    r: 1,
    c: 0
  }, {
    k: 'down',
    l: '↓',
    r: 2,
    c: 1
  }, {
    k: 'right',
    l: '→',
    r: 1,
    c: 2
  }];
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
        }}>🎮 Manual Control</div>
        </div>

        {/* Robot Selector */}
        <div style={{
        display: 'flex',
        gap: 9,
        marginBottom: 22
      }}>
          {ROBOTS.filter(r => r.status !== 'offline').map(r => <button key={r.id} onClick={() => setSelRobot(r.id)} className={`rob-btn ${selRobot === r.id ? 'primary' : 'ghost'}`} style={{
          flex: 1,
          padding: '10px 8px',
          fontSize: 12.5
        }}>
              {r.emoji} {r.name.split(' ')[0]}
            </button>)}
        </div>

        <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 18
      }}>
          {/* Joystick */}
          <div className="rob-card" style={{
          padding: 24,
          textAlign: 'center'
        }}>
            <div style={{
            fontSize: 13,
            fontWeight: 700,
            color: 'rgba(0,0,0,.7)',
            marginBottom: 20,
            textTransform: 'uppercase',
            letterSpacing: '.7px'
          }}>Movement</div>
            <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3,44px)',
            gap: 8,
            justifyContent: 'center',
            margin: '0 auto 20px'
          }}>
              {Array.from({
              length: 9
            }, (_, i) => {
              const r = Math.floor(i / 3),
                c = i % 3;
              const dir = dirs.find(d => d.r === r && d.c === c);
              if (!dir && !(r === 1 && c === 1)) return <div key={i} />;
              if (r === 1 && c === 1) return <div key={i} style={{
                width: 44,
                height: 44,
                borderRadius: 10,
                background: 'rgba(30,126,66,.08)',
                border: '1px solid rgba(30,126,66,.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 16
              }}>🤖</div>;
              return <button key={i} className={`joy-dir-btn${pressed[dir.k] ? ' pressed' : ''}`} onMouseDown={() => {
                press(dir.k);
                sendCmd('move', {
                  direction: dir.k,
                  speed
                });
              }} onMouseUp={() => release(dir.k)} onTouchStart={() => press(dir.k)} onTouchEnd={() => release(dir.k)}>
                    {dir.l}
                  </button>;
            })}
            </div>

            <div style={{
            marginBottom: 14
          }}>
              <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 12,
              color: 'rgba(0,0,0,.7)',
              marginBottom: 6
            }}>
                <span>Speed</span><span style={{
                color: 'var(--g3)',
                fontWeight: 700
              }}>{speed}%</span>
              </div>
              <input type="range" min={10} max={100} value={speed} onChange={e => setSpeed(+e.target.value)} style={{
              width: '100%',
              accentColor: 'var(--g3)'
            }} />
            </div>

            {ROBOTS.find(r => (r.robotId || r.id) === selRobot)?.type === 'Drone' && <div style={{
            marginBottom: 14
          }}>
                <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 12,
              color: 'rgba(0,0,0,.7)',
              marginBottom: 6
            }}>
                  <span>Altitude</span><span style={{
                color: '#ffd700',
                fontWeight: 700
              }}>{altitude}m</span>
                </div>
                <input type="range" min={2} max={30} value={altitude} onChange={e => setAltitude(+e.target.value)} style={{
              width: '100%',
              accentColor: '#ffd700'
            }} />
              </div>}

            <button className={`rob-btn ${sprayOn ? 'danger' : 'green'}`} style={{
            width: '100%'
          }} onClick={() => {
            setSprayOn(v => !v);
            sendCmd(sprayOn ? 'spray_stop' : 'spray_start');
          }}>
              {sprayOn ? '🛑 Stop Spray' : '💧 Start Spray'}
            </button>
          </div>

          {/* Status Panel */}
          <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 14
        }}>
            <div className="rob-card" style={{
            padding: 18
          }}>
              <div style={{
              fontSize: 13,
              fontWeight: 700,
              color: 'rgba(0,0,0,.7)',
              marginBottom: 12,
              textTransform: 'uppercase',
              letterSpacing: '.7px'
            }}>Robot Status</div>
              {[['Robot', ROBOTS.find(r => (r.robotId || r.id) === selRobot)?.name || '—', 'white'], ['Battery', `${ROBOTS.find(r => (r.robotId || r.id) === selRobot)?.battery}%`, 'var(--g4)'], ['Speed', `${speed}%`, 'var(--g3)'], ['Spray', sprayOn ? 'Active' : 'Idle', sprayOn ? 'var(--g4)' : 'rgba(0,0,0,.6)'], ['Status', 'Connected', 'var(--g4)']].map(([l, v, c]) => <div key={l} style={{
              display: 'flex',
              justifyContent: 'space-between',
              padding: '8px 0',
              borderBottom: '1px solid rgba(0,0,0,.7)',
              fontSize: 13
            }}>
                  <span style={{
                color: 'rgba(0,0,0,.9)'
              }}>{l}</span>
                  <span style={{
                fontWeight: 700,
                color: c
              }}>{v}</span>
                </div>)}
            </div>

            <div className="rob-card" style={{
            padding: 18
          }}>
              <div style={{
              fontSize: 13,
              fontWeight: 700,
              color: 'rgba(0,0,0,.7)',
              marginBottom: 12,
              textTransform: 'uppercase',
              letterSpacing: '.7px'
            }}>Camera</div>
              <div style={{
              marginBottom: 10
            }}>
                <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: 12,
                color: 'rgba(0,0,0,.7)',
                marginBottom: 6
              }}>
                  <span>Tilt</span><span style={{
                  color: 'var(--g3)',
                  fontWeight: 700
                }}>{camTilt}°</span>
                </div>
                <input type="range" min={-90} max={0} value={camTilt} onChange={e => setCamTilt(+e.target.value)} style={{
                width: '100%',
                accentColor: 'var(--g3)'
              }} />
              </div>
              <div style={{
              display: 'flex',
              gap: 9
            }}>
                <button className="rob-btn ghost" style={{
                flex: 1,
                fontSize: 12
              }} onClick={() => nav('robot-camera')}>📡 Live Feed</button>
                <button className="rob-btn ghost" style={{
                flex: 1,
                fontSize: 12
              }} onClick={() => toast('Screenshot saved 📸')}>📸 Snap</button>
              </div>
            </div>

            <button className="rob-btn danger" style={{
            width: '100%',
            padding: '13px'
          }} onClick={() => toast('EMERGENCY STOP! All robots halted 🛑', 'err')}>
              🛑 EMERGENCY STOP
            </button>
          </div>
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   BATTERY & MAINTENANCE 🔋
════════════════════════════════════════════════════════════════ */
