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

export default function RobotDashboard({
  user,
  nav,
  toast
}) {
  const [robots, setRobots] = useState(ROBOTS);
  const [selRobot, setSelRobot] = useState(ROBOTS[0] || null);
  const [logs, setLogs] = useState([]);
  const [tick, setTick] = useState(0);

  // Load real robots from API
  useEffect(() => {
    if (!user) return;
    API.get('/api/robots').then(d => {
      if (d.robots && d.robots.length > 0) {
        setRobots(d.robots.map(r => ({
          ...r,
          id: r.robotId,
          lastSeen: r.status === 'online' ? 'Just now' : '2 hrs ago'
        })));
        setSelRobot(r => d.robots[0] || r);
      }
    }).catch(() => {});
    API.get('/api/robots/R01/logs').then(d => {
      if (d.logs) setLogs(d.logs);
    }).catch(() => {});
  }, [user]);

  // Poll every 10s for live updates
  useEffect(() => {
    if (!user) return;
    const iv = setInterval(() => {
      API.get('/api/robots').then(d => {
        if (d.robots) setRobots(d.robots.map(r => ({
          ...r,
          id: r.robotId,
          lastSeen: r.status === 'online' ? 'Just now' : '2 hrs ago'
        })));
      }).catch(() => {});
    }, 10000);
    return () => clearInterval(iv);
  }, [user]);
  useEffect(() => {
    const t = setInterval(() => setTick(p => p + 1), 2000);
    return () => clearInterval(t);
  }, []);
  const sendCommand = async (robotId, command) => {
    try {
      await API.post(`/api/robots/${robotId}/command`, {
        command
      });
      toast(`${command} command bheja gaya! ✅`);
      // Refresh
      API.get('/api/robots').then(d => {
        if (d.robots) setRobots(d.robots.map(r => ({
          ...r,
          id: r.robotId,
          lastSeen: r.status === 'online' ? 'Just now' : '2 hrs ago'
        })));
      }).catch(() => {});
    } catch (e) {
      toast(e.message, 'err');
    }
  };
  const statusColor = {
    online: 'var(--g4)',
    busy: '#ffd700',
    offline: 'rgba(0,0,0,.5)',
    error: '#ff4444'
  };
  return <div className="rob-shell">
      <div className="rob-wrap">
        {/* Header */}
        <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: 28
      }}>
          <div>
            <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            marginBottom: 6
          }}>
              <span style={{
              fontSize: 28
            }}>🤖</span>
              <div style={{
              fontFamily: "'Baloo 2',cursive",
              fontSize: 28,
              fontWeight: 900,
              color: 'var(--g3)'
            }}>Robot Command Center</div>
            </div>
            <div style={{
            fontSize: 13,
            color: 'rgba(0,0,0,.7)'
          }}>BeejHealth Robot Fleet — {user?.district || 'Wagholi Farm'} • Live</div>
          </div>
          <div style={{
          display: 'flex',
          gap: 8
        }}>
            <button className="rob-btn ghost" onClick={() => nav('robot-control')}>🎮 Manual Control</button>
            <button className="rob-btn primary" onClick={() => nav('robot-spray')}>💊 Schedule Spray</button>
          </div>
        </div>

        {/* Fleet Stats */}
        <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4,1fr)',
        gap: 14,
        marginBottom: 28
      }}>
          {[{
          n: robots.length,
          l: 'Total Robots',
          c: 'var(--g3)',
          i: '🤖'
        }, {
          n: robots.filter(r => r.status === 'online').length,
          l: 'Active Now',
          c: 'var(--g4)',
          i: '✅'
        }, {
          n: robots.filter(r => r.status === 'busy').length,
          l: 'Busy',
          c: '#ffd700',
          i: '⚡'
        }, {
          n: robots.filter(r => r.status === 'offline').length,
          l: 'Offline',
          c: 'rgba(0,0,0,.9)',
          i: '🔴'
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
        gridTemplateColumns: '1fr 1.4fr',
        gap: 20
      }}>
          {/* Robot List */}
          <div>
            <div style={{
            fontSize: 14,
            fontWeight: 700,
            color: 'rgba(0,0,0,.8)',
            marginBottom: 13,
            textTransform: 'uppercase',
            letterSpacing: '.8px'
          }}>Fleet</div>
            {robots.map(r => <div key={r.robotId || r.id} className={`robot-row${(selRobot.robotId || selRobot.id) === (r.robotId || r.id) ? ' sel' : ''}`} onClick={() => setSelRobot(r)} style={{
            cursor: 'pointer'
          }}>
                <div className="robot-av" style={{
              background: r.status === 'online' ? 'rgba(77,189,122,.12)' : r.status === 'busy' ? 'rgba(255,215,0,.12)' : 'rgba(0,0,0,.7)'
            }}>
                  {r.emoji}
                </div>
                <div style={{
              flex: 1,
              minWidth: 0
            }}>
                  <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 4
              }}>
                    <div style={{
                  fontSize: 14,
                  fontWeight: 800,
                  color: 'var(--tx)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}>{r.name}</div>
                    <span className={`rob-badge ${r.status}`}><span className={`rob-dot ${r.status}`} />{r.status}</span>
                  </div>
                  <div style={{
                fontSize: 12,
                color: 'rgba(0,0,0,.9)'
              }}>{r.type} • {r.field}</div>
                  <div style={{
                marginTop: 7
              }}>
                    <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: 10,
                  color: 'rgba(0,0,0,.9)',
                  marginBottom: 3
                }}>
                      <span>🔋 {r.battery}%</span><span>📶 {r.signal}%</span>
                    </div>
                    <div className="rob-prog"><div className={`rob-prog-fill ${r.battery > 50 ? 'green' : r.battery > 20 ? 'yellow' : 'red'}`} style={{
                    width: `${r.battery}%`
                  }} /></div>
                  </div>
                </div>
              </div>)}
            <button className="rob-btn ghost" style={{
            width: '100%',
            marginTop: 8
          }} onClick={() => toast('New robot pairing — hardware connection required', 'inf')}>
              + Add New Robot
            </button>
          </div>

          {/* Selected Robot Detail */}
          <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 16
        }}>
            <div className="rob-card" style={{
            padding: 20
          }}>
              <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              marginBottom: 16
            }}>
                <div>
                  <div style={{
                  fontSize: 20,
                  fontWeight: 900,
                  color: 'var(--tx)',
                  marginBottom: 4
                }}>{selRobot.emoji} {selRobot.name}</div>
                  <div style={{
                  fontSize: 12,
                  color: 'rgba(0,0,0,.9)'
                }}>{selRobot.model} • ID: {selRobot.id}</div>
                </div>
                <span className={`rob-badge ${selRobot.status}`}><span className={`rob-dot ${selRobot.status}`} />{selRobot.status.toUpperCase()}</span>
              </div>

              <div style={{
              padding: 13,
              background: 'rgba(30,126,66,.07)',
              borderRadius: 10,
              marginBottom: 14,
              fontSize: 13,
              color: 'var(--g3)',
              fontWeight: 600
            }}>
                📋 Current Task: {selRobot.task}
              </div>

              <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 10,
              marginBottom: 14
            }}>
                {[['🔋 Battery', `${selRobot.battery}%`, selRobot.battery > 50 ? 'var(--g4)' : selRobot.battery > 20 ? '#ffd700' : '#ff4444'], ['📶 Signal', `${selRobot.signal}%`, 'var(--g3)'], ['📍 Location', selRobot.field, 'rgba(0,0,0,.9)'], ['⏱️ Last Seen', selRobot.lastSeen, 'rgba(0,0,0,.9)']].map(([l, v, c]) => <div key={l} style={{
                padding: 11,
                background: 'rgba(0,0,0,.6)',
                borderRadius: 9
              }}>
                    <div style={{
                  fontSize: 11,
                  color: 'rgba(0,0,0,.6)',
                  marginBottom: 3
                }}>{l}</div>
                    <div style={{
                  fontSize: 14,
                  fontWeight: 800,
                  color: c
                }}>{v}</div>
                  </div>)}
              </div>

              {selRobot.status !== 'offline' && <div style={{
              display: 'flex',
              gap: 9
            }}>
                  <button className="rob-btn ghost" style={{
                flex: 1
              }} onClick={() => nav('robot-camera')}>📡 Camera</button>
                  <button className="rob-btn ghost" style={{
                flex: 1
              }} onClick={() => nav('robot-map')}>🗺️ Navigate</button>
                  <button className="rob-btn danger" style={{
                flex: 1
              }} onClick={() => sendCommand(selRobot.robotId || selRobot.id, 'emergency_stop')}>🛑 Stop</button>
                </div>}
              {selRobot.status === 'offline' && <button className="rob-btn primary" style={{
              width: '100%'
            }} onClick={() => sendCommand(selRobot.robotId || selRobot.id, 'wake_up')}>⚡ Wake Up Robot</button>}
            </div>

            {/* Live Activity Log */}
            <div className="rob-card" style={{
            padding: 18
          }}>
              <div style={{
              fontSize: 13,
              fontWeight: 700,
              color: 'rgba(0,0,0,.9)',
              marginBottom: 11,
              textTransform: 'uppercase',
              letterSpacing: '.6px'
            }}>Live Log</div>
              {(logs.length > 0 ? logs : [{
              createdAt: new Date().toISOString(),
              event: 'System online — all sensors nominal',
              level: 'info'
            }, {
              createdAt: new Date(Date.now() - 300000).toISOString(),
              event: 'GPS lock acquired — 18.59°N 73.74°E',
              level: 'info'
            }, {
              createdAt: new Date(Date.now() - 600000).toISOString(),
              event: 'Battery charge complete — 87%',
              level: 'info'
            }]).slice(0, 5).map((log, i) => <div key={i} style={{
              display: 'flex',
              gap: 9,
              padding: '7px 0',
              borderBottom: '1px solid rgba(0,0,0,.7)',
              fontSize: 12
            }}>
                  <span style={{
                color: 'rgba(0,0,0,.5)',
                flexShrink: 0,
                fontFamily: 'monospace'
              }}>{new Date(log.createdAt).toLocaleTimeString('en-IN', {
                  hour: '2-digit',
                  minute: '2-digit'
                })}</span>
                  <span style={{
                color: log.level === 'warning' ? '#ffd700' : log.level === 'error' ? '#ff4444' : 'var(--g3)'
              }}>{log.event}</span>
                </div>)}
            </div>
          </div>
        </div>
        {/* Quick Nav */}
        <div style={{
        marginTop: 24,
        display: 'grid',
        gridTemplateColumns: 'repeat(4,1fr)',
        gap: 12
      }}>
          {[{
          id: 'robot-camera',
          l: '📡 Live Camera',
          d: 'All feeds'
        }, {
          id: 'robot-map',
          l: '🗺️ Navigation',
          d: 'Auto-pilot'
        }, {
          id: 'robot-control',
          l: '🎮 Manual Control',
          d: 'Joystick'
        }, {
          id: 'robot-spray',
          l: '💊 Spray Scheduler',
          d: 'Plan missions'
        }, {
          id: 'robot-maintenance',
          l: '🔋 Maintenance',
          d: 'Battery & alerts'
        }, {
          id: 'robot-analytics',
          l: '📊 Analytics',
          d: 'Reports'
        }, {
          id: 'satellite',
          l: '🛰️ Satellite Map',
          d: 'NDVI view'
        }, {
          id: 'soil-sensors',
          l: '🌱 Soil Sensors',
          d: 'IoT data'
        }].map(q => <div key={q.id} className="rob-card" style={{
          padding: 16,
          cursor: 'pointer',
          transition: 'all .18s'
        }} onClick={() => nav(q.id)} onMouseEnter={e => {
          e.currentTarget.style.borderColor = 'var(--g3)';
          e.currentTarget.style.background = 'rgba(30,126,66,.07)';
        }} onMouseLeave={e => {
          e.currentTarget.style.borderColor = 'rgba(30,126,66,.2)';
          e.currentTarget.style.background = 'rgba(0,0,0,.6)';
        }}>
              <div style={{
            fontSize: 20,
            marginBottom: 7
          }}>{q.l.split(' ')[0]}</div>
              <div style={{
            fontSize: 13,
            fontWeight: 700,
            color: 'var(--tx)'
          }}>{q.l.slice(3)}</div>
              <div style={{
            fontSize: 11,
            color: 'rgba(0,0,0,.6)',
            marginTop: 3
          }}>{q.d}</div>
            </div>)}
        </div>
      </div>
    </div>;
}
