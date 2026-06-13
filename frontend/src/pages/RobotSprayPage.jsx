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

export default function RobotSprayPage({
  nav,
  toast
}) {
  const [zones, setZones] = useState([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    API.get('/api/spray-jobs').then(d => {
      if (d.jobs && d.jobs.length > 0) {
        setZones(d.jobs.map((j, i) => ({
          id: j.jobId || i,
          name: `${j.field} — ${j.crop}`,
          area: j.area,
          disease: j.disease,
          chem: j.chemical,
          dose: j.dose,
          scheduled: new Date(j.scheduledAt).toLocaleString('en-IN', {
            day: '2-digit',
            month: 'short',
            hour: '2-digit',
            minute: '2-digit'
          }),
          priority: j.priority,
          sel: j.status !== 'completed',
          status: j.status,
          jobId: j.jobId
        })));
      }
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);
  const cancelJob = async jobId => {
    try {
      await API.delete(`/api/spray-jobs/${jobId}`);
      setZones(p => p.filter(z => z.jobId !== jobId));
      toast('Spray job cancelled ✅');
    } catch (e) {
      toast('Cancel fail', 'err');
    }
  };
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const startSpray = async () => {
    setRunning(true);
    setProgress(0);
    for (let i = 1; i <= 10; i++) {
      await new Promise(r => setTimeout(r, 400));
      setProgress(i * 10);
    }
    setRunning(false);
    toast('✅ Spray mission complete! DroneBot Alpha finished Field 1');
  };
  const prioColor = {
    high: '#ff4444',
    med: '#ffd700',
    low: 'var(--g4)'
  };
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
        }}>💊 Precision Spray Scheduler</div>
        </div>

        {/* Active Mission */}
        {running && <div className="rob-card rob-card-glow" style={{
        padding: 22,
        marginBottom: 20
      }}>
            <div style={{
          fontSize: 15,
          fontWeight: 800,
          color: 'var(--g3)',
          marginBottom: 14
        }}>🚁 Mission Active — DroneBot Alpha</div>
            <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 13,
          color: 'rgba(0,0,0,.8)',
          marginBottom: 8
        }}>
              <span>Field 1 — Tomato • Mancozeb spray</span><span>{progress}%</span>
            </div>
            <div className="rob-prog" style={{
          height: 10,
          marginBottom: 14
        }}>
              <div className="rob-prog-fill cyan" style={{
            width: `${progress}%`
          }} />
            </div>
            <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3,1fr)',
          gap: 10,
          marginBottom: 14
        }}>
              {[['Area Done', `${(progress / 100 * 2).toFixed(1)} Acres`, 'var(--g3)'], ['Chemical Used', `${(progress / 100 * 5).toFixed(1)}L`, 'var(--g4)'], ['ETA', `${Math.ceil((100 - progress) / 10) * 2} min`, '#ffd700']].map(([l, v, c]) => <div key={l} style={{
            padding: 11,
            background: 'rgba(0,0,0,.6)',
            borderRadius: 9,
            textAlign: 'center'
          }}>
                  <div style={{
              fontSize: 11,
              color: 'rgba(0,0,0,.6)',
              marginBottom: 3
            }}>{l}</div>
                  <div style={{
              fontSize: 16,
              fontWeight: 900,
              color: c
            }}>{v}</div>
                </div>)}
            </div>
            <button className="rob-btn danger" style={{
          width: '100%'
        }} onClick={() => {
          setRunning(false);
          toast('Mission aborted! 🛑', 'err');
        }}>🛑 Abort Mission</button>
          </div>}

        {/* AI Recommendations */}
        <div className="rob-card" style={{
        padding: 18,
        marginBottom: 20,
        borderColor: 'rgba(77,189,122,.3)'
      }}>
          <div style={{
          fontSize: 13,
          fontWeight: 700,
          color: 'var(--g4)',
          marginBottom: 10
        }}>🤖 AI Spray Recommendation</div>
          <div style={{
          fontSize: 13,
          color: 'rgba(0,0,0,.9)',
          lineHeight: 1.7
        }}>
            Weather forecast ke hisaab se <strong style={{
            color: 'var(--tx)'
          }}>aaj 4-6 PM ideal window</strong> hai spray ke liye. Wind speed 8 km/h (optimal), humidity 65% (good). Kal baarish expected — delay mat karein.
          </div>
        </div>

        {/* Spray Zones */}
        <div style={{
        fontSize: 13,
        fontWeight: 700,
        color: 'rgba(0,0,0,.7)',
        marginBottom: 12,
        textTransform: 'uppercase',
        letterSpacing: '.7px'
      }}>Spray Queue</div>
        {zones.map(z => <div key={z.id} className={`spray-zone${z.sel ? ' active' : ''}`} onClick={() => setZones(p => p.map(x => ({
        ...x,
        sel: x.id === z.id
      })))}>
            <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: 10
        }}>
              <div>
                <div style={{
              fontSize: 15,
              fontWeight: 800,
              color: 'var(--tx)'
            }}>{z.name}</div>
                <div style={{
              fontSize: 12,
              color: 'rgba(0,0,0,.9)',
              marginTop: 2
            }}>📐 {z.area} Acres • 🦠 {z.disease}</div>
              </div>
              <div style={{
            textAlign: 'right'
          }}>
                <div style={{
              fontSize: 11,
              fontWeight: 700,
              color: prioColor[z.priority],
              textTransform: 'uppercase'
            }}>{z.priority} priority</div>
                <div style={{
              fontSize: 11,
              color: 'rgba(0,0,0,.6)',
              marginTop: 2
            }}>⏰ {z.scheduled}</div>
              </div>
            </div>
            <div style={{
          display: 'flex',
          gap: 14,
          fontSize: 12.5
        }}>
              <span style={{
            color: 'rgba(0,0,0,.8)'
          }}>💊 {z.chem}</span>
              <span style={{
            color: 'rgba(0,0,0,.8)'
          }}>💧 {z.dose}g/L</span>
              <span style={{
            color: 'rgba(0,0,0,.8)'
          }}>🤖 DroneBot Alpha</span>
            </div>
            {z.sel && <div style={{
          display: 'flex',
          gap: 9,
          marginTop: 12
        }}>
                <button className="rob-btn ghost" style={{
            flex: 1
          }} onClick={e => {
            e.stopPropagation();
            toast('Schedule updated ✅');
          }}>✏️ Edit</button>
                <button className="rob-btn primary" style={{
            flex: 2
          }} onClick={e => {
            e.stopPropagation();
            startSpray();
          }} disabled={running}>
                  {running ? '🚁 In Progress...' : '🚀 Start Spray Now'}
                </button>
              </div>}
          </div>)}

        {/* Chemical Tank Status */}
        <div className="rob-card" style={{
        padding: 20,
        marginTop: 8
      }}>
          <div style={{
          fontSize: 14,
          fontWeight: 700,
          color: 'rgba(0,0,0,.8)',
          marginBottom: 14,
          textTransform: 'uppercase',
          letterSpacing: '.7px'
        }}>Chemical Tank Status</div>
          {[{
          name: 'Mancozeb 75% WP',
          level: 72,
          cap: '10L'
        }, {
          name: 'Copper Oxychloride',
          level: 45,
          cap: '5L'
        }, {
          name: 'Water Tank',
          level: 88,
          cap: '40L'
        }].map(t => <div key={t.name} style={{
          marginBottom: 13
        }}>
              <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 13,
            marginBottom: 5
          }}>
                <span style={{
              color: 'rgba(0,0,0,.9)',
              fontWeight: 600
            }}>{t.name}</span>
                <span style={{
              color: t.level > 50 ? 'var(--g4)' : t.level > 25 ? '#ffd700' : '#ff4444',
              fontWeight: 700
            }}>{t.level}% ({t.cap})</span>
              </div>
              <div className="rob-prog"><div className={`rob-prog-fill ${t.level > 50 ? 'green' : t.level > 25 ? 'yellow' : 'red'}`} style={{
              width: `${t.level}%`
            }} /></div>
            </div>)}
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   LIVE CAMERA FEED 📡
════════════════════════════════════════════════════════════════ */
