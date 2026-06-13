import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import { API, saveSession, clearSession, loadSession } from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody } from '../utils/helpers.jsx';
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary } from '../utils/constants.jsx';

import BeejHealthApp from '../components/BeejHealthApp.jsx';
import RobotAnalyticsPage from './RobotAnalyticsPage.jsx';
import RobotMaintenancePage from './RobotMaintenancePage.jsx';
import RobotControlPage from './RobotControlPage.jsx';
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
   AUTO FIELD NAVIGATION & MAPPING 🗺️
════════════════════════════════════════════════════════════════ */
function RobotMapPage({
  nav,
  toast
}) {
  const [mode, setMode] = useState('auto');
  const [mission, setMission] = useState(null);
  const [robotLoc, setRobotLoc] = useState(null);
  const [selRobotId, setSelRobotId] = useState('R01');
  useEffect(() => {
    API.get(`/api/robots/${selRobotId}/location`).then(d => setRobotLoc(d)).catch(() => {});
    const iv = setInterval(() => {
      API.get(`/api/robots/${selRobotId}/location`).then(d => setRobotLoc(d)).catch(() => {});
    }, 8000);
    return () => clearInterval(iv);
  }, [selRobotId]);
  const startMission = async modeType => {
    try {
      const res = await API.post(`/api/robots/${selRobotId}/navigate`, {
        mode: modeType,
        field: 'Field 1'
      });
      setMission({
        mode: modeType,
        eta: res.eta,
        started: new Date().toLocaleTimeString()
      });
      toast(`Mission started! ETA: ${res.eta} ✅`);
    } catch (e) {
      toast('Mission start fail', 'err');
    }
  };
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setTick(p => p + 1), 800);
    return () => clearInterval(t);
  }, []);
  const robotX = 30 + Math.sin(tick * 0.3) * 8;
  const robotY = 40 + Math.cos(tick * 0.25) * 6;
  const fields = [{
    name: 'F1',
    x: 12,
    y: 15,
    w: 30,
    h: 25,
    color: 'rgba(255,68,68,.25)',
    border: '#ff4444',
    crop: '🍅'
  }, {
    name: 'F2',
    x: 50,
    y: 12,
    w: 26,
    h: 28,
    color: 'rgba(77,189,122,.15)',
    border: 'var(--g4)',
    crop: '🌾'
  }, {
    name: 'F3',
    x: 18,
    y: 52,
    w: 22,
    h: 24,
    color: 'rgba(30,126,66,.15)',
    border: 'var(--g3)',
    crop: '🥔'
  }];
  const waypoints = [{
    x: 25,
    y: 28
  }, {
    x: 40,
    y: 35
  }, {
    x: 55,
    y: 42
  }, {
    x: 68,
    y: 30
  }, {
    x: 72,
    y: 55
  }];
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
        }}>🗺️ Field Navigation</div>
          <div style={{
          marginLeft: 'auto',
          display: 'flex',
          gap: 7
        }}>
            {['auto', 'manual', 'scan'].map(m => <button key={m} onClick={() => setMode(m)} className={`rob-btn ${mode === m ? 'primary' : 'ghost'}`} style={{
            padding: '7px 14px',
            fontSize: 12
          }}>
                {m === 'auto' ? '🤖 Auto' : m === 'manual' ? '🎮 Manual' : '🔍 Scan'}
              </button>)}
          </div>
        </div>

        {/* Field Map */}
        <div className="rob-field-map" style={{
        marginBottom: 18
      }}>
          <div className="rob-grid-bg" />

          {/* Fields */}
          {fields.map(f => <div key={f.name} style={{
          position: 'absolute',
          left: `${f.x}%`,
          top: `${f.y}%`,
          width: `${f.w}%`,
          height: `${f.h}%`,
          background: f.color,
          border: `1.5px solid ${f.border}`,
          borderRadius: 8,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          gap: 3,
          cursor: 'pointer'
        }} onClick={() => toast(`${f.name} selected — mission set karein`)}>
              <div style={{
            fontSize: 20
          }}>{f.crop}</div>
              <div style={{
            fontSize: 10,
            fontWeight: 700,
            color: f.border
          }}>{f.name}</div>
            </div>)}

          {/* Waypoints */}
          {waypoints.map((w, i) => <div key={i} style={{
          position: 'absolute',
          left: `${w.x}%`,
          top: `${w.y}%`,
          width: 10,
          height: 10,
          background: 'rgba(255,215,0,.7)',
          border: '2px solid #ffd700',
          borderRadius: '50%',
          transform: 'translate(-50%,-50%)',
          zIndex: 3
        }} />)}

          {/* Robot Icon (animated) */}
          <div className="rob-robot-icon" style={{
          left: `${robotX}%`,
          top: `${robotY}%`,
          transform: 'translate(-50%,-50%)'
        }}>🚁</div>
          <div className="rob-ping" style={{
          left: `${robotX}%`,
          top: `${robotY}%`,
          transform: 'translate(-50%,-50%)'
        }} />

          {/* HUD */}
          <div style={{
          position: 'absolute',
          top: 10,
          left: 12,
          background: 'rgba(0,0,0,.65)',
          borderRadius: 8,
          padding: '8px 12px',
          backdropFilter: 'blur(8px)'
        }}>
            <div style={{
            fontSize: 11,
            color: 'var(--g3)',
            fontFamily: 'monospace'
          }}>GPS: 18.5912°N, 73.7389°E</div>
            <div style={{
            fontSize: 11,
            color: 'var(--g4)',
            marginTop: 2
          }}>SPD: 2.8 m/s | ALT: 8m</div>
          </div>

          {/* Legend */}
          <div style={{
          position: 'absolute',
          bottom: 10,
          right: 10,
          background: 'rgba(0,0,0,.65)',
          borderRadius: 8,
          padding: '7px 10px',
          fontSize: 10,
          color: 'rgba(0,0,0,.8)'
        }}>
            🚁 DroneBot Alpha<br />🟡 Waypoints ({waypoints.length})<br />📐 4.5 Acres Total
          </div>
        </div>

        {/* Mission Planner */}
        <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 16
      }}>
          <div className="rob-card" style={{
          padding: 20
        }}>
            <div style={{
            fontSize: 14,
            fontWeight: 700,
            color: 'rgba(0,0,0,.8)',
            marginBottom: 14,
            textTransform: 'uppercase',
            letterSpacing: '.7px'
          }}>Mission Planner</div>
            <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
            marginBottom: 16
          }}>
              {[{
              l: 'Mission Type',
              opts: ['Spray', 'Scan', 'Survey', 'Monitor']
            }, {
              l: 'Robot',
              opts: ['DroneBot Alpha', 'GroundBot Beta']
            }, {
              l: 'Pattern',
              opts: ['Grid', 'Spiral', 'Custom', 'Perimeter']
            }].map(({
              l,
              opts
            }) => <div key={l}>
                  <div style={{
                fontSize: 11,
                color: 'rgba(0,0,0,.6)',
                marginBottom: 5,
                fontWeight: 600
              }}>{l}</div>
                  <select style={{
                width: '100%',
                padding: '9px 12px',
                borderRadius: 9,
                background: 'rgba(0,0,0,.9)',
                border: '1px solid rgba(30,126,66,.2)',
                color: 'var(--tx)',
                fontSize: 13,
                fontFamily: 'Outfit,sans-serif',
                outline: 'none'
              }}>
                    {opts.map(o => <option key={o} style={{
                  background: '#0d1b3e'
                }}>{o}</option>)}
                  </select>
                </div>)}
            </div>
            <button className="rob-btn primary" style={{
            width: '100%'
          }} onClick={() => {
            setMission({
              status: 'running',
              pct: 0
            });
            toast('Mission started! 🚀');
          }}>
              🚀 Start Mission
            </button>
          </div>

          <div className="rob-card" style={{
          padding: 20
        }}>
            <div style={{
            fontSize: 14,
            fontWeight: 700,
            color: 'rgba(0,0,0,.8)',
            marginBottom: 14,
            textTransform: 'uppercase',
            letterSpacing: '.7px'
          }}>Mission Status</div>
            {mission ? <>
                <div style={{
              textAlign: 'center',
              marginBottom: 14
            }}>
                  <div style={{
                fontSize: 36,
                marginBottom: 7
              }}>🚁</div>
                  <div style={{
                fontSize: 15,
                fontWeight: 800,
                color: 'var(--g3)'
              }}>Mission Running</div>
                  <div style={{
                fontSize: 12,
                color: 'rgba(0,0,0,.9)',
                marginTop: 3
              }}>Waypoint 3/5 • ETA 8 min</div>
                </div>
                <div className="rob-prog" style={{
              height: 8,
              marginBottom: 14
            }}>
                  <div className="rob-prog-fill cyan" style={{
                width: '60%'
              }} />
                </div>
                <button className="rob-btn danger" style={{
              width: '100%'
            }} onClick={() => {
              setMission(null);
              toast('Mission cancelled 🛑', 'err');
            }}>🛑 Cancel Mission</button>
              </> : <div style={{
            textAlign: 'center',
            padding: 20,
            color: 'rgba(0,0,0,.5)'
          }}>
                <div style={{
              fontSize: 40,
              marginBottom: 10
            }}>🗺️</div>
                <div style={{
              fontSize: 13
            }}>No active mission.<br />Plan karein aur start karein.</div>
              </div>}
          </div>
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   MANUAL ROBOT CONTROL 🎮
════════════════════════════════════════════════════════════════ */
