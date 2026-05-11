import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import { API, saveSession, clearSession, loadSession } from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody } from '../utils/helpers.jsx';
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary } from '../utils/constants.jsx';
import { cropName, formatCurrency, formatDateTime, formatNumber, formatPercent, reportValue, tx } from '../utils/localize.jsx';

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
import MyFarmPage from './MyFarmPage.jsx';
import CaseDetailPage from './CaseDetailPage.jsx';
import ChatPage from './ChatPage.jsx';
import BookingPage from './BookingPage.jsx';
import ExpertsPage from './ExpertsPage.jsx';
import FarmerDash from './FarmerDash.jsx';
import MyConsultPage from './MyConsultPage.jsx';
import AIReportPage from './AIReportPage.jsx';
import ConsultPage from './ConsultPage.jsx';
import HomePage from './HomePage.jsx';
import ExpertOnboarding from '../components/ExpertOnboarding.jsx';
import FarmerOnboarding from '../components/FarmerOnboarding.jsx';
import AuthModal from '../components/AuthModal.jsx';

export default function ExpertDash({
  user,
  nav,
  toast
}) {
  const { i18n } = useTranslation();
  const N = value => formatNumber(value, i18n);
  const P = value => formatPercent(value, i18n);
  const C = value => formatCurrency(value, i18n);
  const RV = value => reportValue(value, i18n);
  const D = value => formatDateTime(value, i18n, { day: '2-digit', month: 'short', year: 'numeric' });
  const [tab, setTab] = useState('urgent');
  const [avail, setAvail] = useState(user?.available !== false);
  const [cases, setCases] = useState(CONSULTATIONS);
  useEffect(() => {
    if (!user) return;
    API.get('/api/consultations').then(d => {
      if (d.consultations && d.consultations.length > 0) setCases(d.consultations.map(c => ({
        id: c._id,
        crop: cropName(c.cropId || c.cropName, i18n) || c.cropName,
        emoji: c.cropEmoji || '🌱',
        issue: `${RV(c.disease)} — ${P(c.confidence || 0)} ${tx(i18n, 'confidenceWord')}`,
        date: D(c.createdAt),
        expert: user?.name || 'Expert',
        status: c.status,
        statusLabel: c.status === 'completed' ? tx(i18n, 'completed') : c.status === 'expert_assigned' ? tx(i18n, 'expertAssigned') : c.status === 'pending' ? tx(i18n, 'pending') : tx(i18n, 'aiReady'),
        sev: c.severity || 1,
        conf: c.confidence || 0
      })));
    }).catch(() => {});
  }, [user, i18n.language]);

  // Poll for new cases every 15 seconds — real-time notification
  const [newCaseCount, setNewCaseCount] = useState(0);
  useEffect(() => {
    if (!user || user?.type !== 'expert') return;
    let prevCount = 0;
    const poll = setInterval(() => {
      API.get('/api/consultations').then(d => {
        const n = (d.consultations || []).length;
        if (prevCount > 0 && n > prevCount) setNewCaseCount(n - prevCount);
        prevCount = n;
      }).catch(() => {});
    }, 15000);
    return () => clearInterval(poll);
  }, [user]);
  return <div className="wrap">
      {newCaseCount > 0 && <div style={{
      background: 'var(--b3)',
      color: 'white',
      padding: '10px 18px',
      borderRadius: 10,
      marginBottom: 16,
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      fontWeight: 700,
      fontSize: 13.5
    }}>
          🔔 {N(newCaseCount)} {newCaseCount === 1 ? tx(i18n, 'newCaseArrived') : tx(i18n, 'newCasesArrived')}! <button style={{
        background: 'white',
        color: 'var(--b1)',
        border: 'none',
        borderRadius: 7,
        padding: '4px 12px',
        fontWeight: 800,
        cursor: 'pointer'
      }} onClick={() => setNewCaseCount(0)}>{tx(i18n, 'see')} →</button>
        </div>}
      {/* ── EXPERT HEADER CARD ── */}
      <div className="ed-head">
        <div style={{
        position: 'absolute',
        right: -30,
        top: -30,
        width: 180,
        height: 180,
        background: 'rgba(255,255,255,.05)',
        borderRadius: '50%'
      }} />
        <div style={{
        position: 'absolute',
        right: 80,
        bottom: -50,
        width: 120,
        height: 120,
        background: 'rgba(255,255,255,.03)',
        borderRadius: '50%'
      }} />
        <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        position: 'relative'
      }}>
          <div>
            <div style={{
            fontSize: 12,
            opacity: .76,
            marginBottom: 4,
            fontWeight: 600
          }}>👨‍⚕️ {tx(i18n, 'expertDashboard')}</div>
            <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 24,
            fontWeight: 900,
            color: 'white'
          }}>{tx(i18n, 'hello')}, Dr. {user?.name?.split(' ')?.slice(-1)?.[0] || 'Expert'}! 👋</div>
            <div style={{
            fontSize: 13.5,
            opacity: .8,
            color: 'white',
            marginTop: 3
          }}>{user?.spec || 'Plant Pathologist'}</div>
            <div style={{
            marginTop: 10,
            display: 'flex',
            alignItems: 'center',
            gap: 10
          }}>
              <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '6px 12px',
              background: 'rgba(255,255,255,.12)',
              borderRadius: 100,
              backdropFilter: 'blur(6px)'
            }}>
                <div style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: avail ? '#4ade80' : '#f87171',
                animation: avail ? 'ringPulse 2s infinite' : 'none'
              }} />
                <span style={{
                fontSize: 12.5,
                fontWeight: 700,
                color: 'white'
              }}>{avail ? tx(i18n, 'availableNow') : tx(i18n, 'busyOffline')}</span>
              </div>
              <button onClick={async () => {
              const nv = !avail;
              try {
                await API.patch('/api/experts/availability', {
                  available: nv
                });
                setAvail(nv);
                toast(nv ? '🟢 Online — cases milenge' : '🔴 Offline ho gaye');
              } catch (e) {
                toast(e.message, 'err');
              }
            }} className="btn btn-sm" style={{
              background: 'rgba(255,255,255,.18)',
              color: 'white',
              border: '1.5px solid rgba(255,255,255,.3)',
              borderRadius: 8,
              padding: '5px 12px',
              fontSize: 12,
              fontWeight: 700,
              cursor: 'pointer'
            }}>
                {avail ? `🔴 ${tx(i18n, 'goOffline')}` : `🟢 ${tx(i18n, 'goOnline')}`}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── MAIN 2-COL ── */}
      <div className="dash-2">
        {/* LEFT: Case Queue */}
        <div>
          <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 13
        }}>
            <div style={{
            fontSize: 16,
            fontWeight: 800,
            color: 'var(--b1)'
          }}>📋 {tx(i18n, 'caseQueue')}</div>
            <button className="btn btn-ghost btn-sm" style={{
            color: 'var(--b3)'
          }} onClick={() => nav('case-detail')}>{tx(i18n, 'allCases')} →</button>
          </div>
          <div style={{
          display: 'flex',
          gap: 5,
          marginBottom: 14
        }}>
            {[{
            k: 'urgent',
            l: `🚨 ${tx(i18n, 'urgent')}`,
            n: 3
          }, {
            k: 'pending',
            l: `📋 ${tx(i18n, 'pending')}`,
            n: 5
          }, {
            k: 'active',
            l: `🔄 ${tx(i18n, 'active')}`,
            n: 2
          }].map(t => <button key={t.k} onClick={() => setTab(t.k)} style={{
            flex: 1,
            padding: '8px 6px',
            borderRadius: 8,
            fontSize: 12,
            fontWeight: 700,
            border: `2px solid ${tab === t.k ? 'var(--b3)' : 'var(--br)'}`,
            background: tab === t.k ? 'var(--bp)' : 'none',
            color: tab === t.k ? 'var(--b3)' : 'var(--tx2)',
            cursor: 'pointer',
            fontFamily: "'Outfit',sans-serif",
            transition: 'all .18s'
          }}>
                {t.l}({N(t.n)})
              </button>)}
          </div>
          {cases.map((c, i) => <div key={c.id} className={`case-card${i === 0 ? ' urg' : i === 2 ? ' med' : ''}`} onClick={() => {
          rememberConsultationContext(c.id);
          nav('case-detail');
        }}>
              <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            marginBottom: 7
          }}>
                <div>
                <div className="case-id">#{String(c.id || '').slice(-6).toUpperCase()} • {String(c.date).split('•')[0].trim()}</div>
                  <div className="case-crop">{c.emoji} {cropName(c.crop, i18n) || c.crop}</div>
                </div>
                <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 4,
              alignItems: 'flex-end'
            }}>
                  <span className={`badge ${c.sev >= 3 ? 'bg-r' : c.sev === 2 ? 'bg-a' : 'bg-g'}`}>{tx(i18n, 'severityStage')} {N(c.sev)}/{N(5)}</span>
                  <span className="badge bg-b">AI {P(c.conf)}</span>
                </div>
              </div>
              <div className="case-issue">{RV(c.issue)}</div>
              <div className="case-meta-row">
                <span>👤 {tx(i18n, 'farmerLabel')} #{String(c.id || '').slice(-6).toUpperCase()}</span><span>•</span><span>📍 Pune, MH</span><span>•</span><span>🤖 AI {P(c.conf)} {tx(i18n, 'aiConfident')}</span>
              </div>
              <div style={{
            display: 'flex',
            gap: 7,
            marginTop: 11
          }}>
                <button className="btn btn-b btn-sm" style={{
              flex: 2
            }} onClick={e => {
              e.stopPropagation();
              rememberConsultationContext(c.id);
              nav('case-detail');
            }}>🔍 {tx(i18n, 'reviewCase')} →</button>
                <button className="btn btn-ghost btn-sm" style={{
              flex: 1,
              color: 'var(--b2)'
            }} onClick={e => {
              e.stopPropagation();
              rememberConsultationContext(c.id);
              nav('chat');
            }}>💬 {tx(i18n, 'chatAction')}</button>
              </div>
            </div>)}
        </div>

        {/* RIGHT: Performance + Availability + Earnings */}
        <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 18
      }}>
          {/* Performance */}
          <div className="card" style={{
          padding: 20
        }}>
            <div style={{
            fontSize: 15,
            fontWeight: 800,
            color: 'var(--b1)',
            marginBottom: 14
          }}>📊 {tx(i18n, 'weeklyPerformance')}</div>
            {[[tx(i18n, 'casesSolved'), N(24), '✅'], [tx(i18n, 'avgResponse'), `${N(38)} ${tx(i18n, 'minutesShort')}`, '⚡'], [tx(i18n, 'rating'), `${N(4.9)} ⭐`, '🏆'], [tx(i18n, 'accuracy'), P(96), '🎯'], [tx(i18n, 'repeatClients'), P(68), '🔄']].map(([l, v, i]) => <div key={l} style={{
            display: 'flex',
            justifyContent: 'space-between',
            padding: '8px 0',
            borderBottom: '1px solid var(--gp)'
          }}>
                <span style={{
              fontSize: 13,
              color: 'var(--tx2)',
              fontWeight: 600
            }}>{i} {l}</span>
                <span style={{
              fontSize: 14,
              fontWeight: 800,
              color: 'var(--b3)'
            }}>{v}</span>
              </div>)}
          </div>

          {/* Availability */}
          <div className="card" style={{
          padding: 20
        }}>
            <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 14
          }}>
              <div style={{
              fontSize: 15,
              fontWeight: 800,
              color: 'var(--b1)'
            }}>📅 {tx(i18n, 'todayAvailability')}</div>
              <button className="btn btn-b btn-sm" onClick={() => toast('Schedule editor — coming soon', 'inf')}>✏️ {tx(i18n, 'edit')}</button>
            </div>
            {[{
            t: `9 AM–${N(11)} AM`,
            s: `🟢 ${tx(i18n, 'available')}`,
            c: 'var(--g4)'
          }, {
            t: `${N(11)} AM–${N(2)} PM`,
            s: `🔴 ${tx(i18n, 'busy')}`,
            c: 'var(--r2)'
          }, {
            t: `${N(2)} PM–${N(5)} PM`,
            s: `🟢 ${tx(i18n, 'available')}`,
            c: 'var(--g4)'
          }, {
            t: `${N(5)} PM–${N(6)} PM`,
            s: `🟡 ${tx(i18n, 'limited')}`,
            c: 'var(--a2)'
          }].map(({
            t,
            s,
            c
          }) => <div key={t} style={{
            display: 'flex',
            justifyContent: 'space-between',
            padding: '8px 0',
            borderBottom: '1px solid var(--gp)',
            fontSize: 13,
            fontWeight: 600
          }}>
                <span style={{
              color: 'var(--tx2)'
            }}>{t}</span>
                <span style={{
              color: c
            }}>{s}</span>
              </div>)}
          </div>

          {/* Earnings Mini */}
          <div className="card" style={{
          padding: 20,
          background: 'linear-gradient(135deg,#eef5ff,#dbeafe)',
          border: '1.5px solid var(--bpb)'
        }}>
            <div style={{
            fontSize: 15,
            fontWeight: 800,
            color: 'var(--b1)',
            marginBottom: 13
          }}>💰 {tx(i18n, 'earningsSummary')}</div>
            {[[tx(i18n, 'today'), C(4800), `${N(6)} ${tx(i18n, 'casesWord')}`], [tx(i18n, 'thisMonth'), C(38400), `${N(48)} ${tx(i18n, 'casesWord')}`], [tx(i18n, 'netPayout'), C(28800), tx(i18n, 'afterDeductions')]].map(([l, v, s]) => <div key={l} style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '8px 0',
            borderBottom: '1px solid rgba(147,197,253,.4)'
          }}>
                <div>
                  <div style={{
                fontSize: 13,
                fontWeight: 700,
                color: 'var(--b1)'
              }}>{l}</div>
                  <div style={{
                fontSize: 11,
                color: 'var(--tx3)'
              }}>{s}</div>
                </div>
                <div style={{
              fontFamily: "'Baloo 2',cursive",
              fontSize: 18,
              fontWeight: 900,
              color: 'var(--b3)'
            }}>{v}</div>
              </div>)}
            <button className="btn btn-b btn-sm" style={{
            width: '100%',
            marginTop: 12
          }} onClick={() => toast('Withdrawal feature — coming soon', 'inf')}>
              💸 {tx(i18n, 'withdrawToBank')}
            </button>
          </div>
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   CONSULTATION PAGE
════════════════════════════════════════════════════════════════ */
