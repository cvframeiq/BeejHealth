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
import MyFarmPage from './MyFarmPage.jsx';
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
   CASE DETAIL (Expert)
════════════════════════════════════════════════════════════════ */
function CaseDetailPage({
  user,
  nav,
  toast
}) {
  const [cases, setCases] = useState([]);
  const [selCase, setSelCase] = useState(null);
  const [tab, setTab] = useState('overview');
  const [msgs, setMsgs] = useState([]);
  const [msgTxt, setMsgTxt] = useState('');
  const [sending, setSending] = useState(false);
  const [reportTxt, setReportTxt] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [loadingCases, setLoadingCases] = useState(true);
  const [pollRef, setPollRef] = useState(null);
  const selectCase = c => {
    setSelCase(c);
    rememberConsultationContext(c._id);
    setMsgs([]);
    if (pollRef) clearInterval(pollRef);
    const loadMsgs = () => {
      API.get('/api/consultations/' + c._id + '/messages').then(m => {
        if (m.messages) setMsgs(m.messages);
      }).catch(() => {});
    };
    loadMsgs();
    const iv = setInterval(loadMsgs, 5000);
    setPollRef(iv);
  };
  useEffect(() => {
    if (!user) return;
    API.get('/api/consultations').then(d => {
      const list = d.consultations || [];
      setCases(list);
      if (list.length > 0) {
        const preferredId = getConsultationContextId();
        const initialCase = preferredId ? list.find(c => c._id === preferredId) || list[0] : list[0];
        selectCase(initialCase);
      }
      setLoadingCases(false);
    }).catch(() => setLoadingCases(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  // Cleanup on unmount
  useEffect(() => () => {
    if (pollRef) clearInterval(pollRef);
  }, [pollRef]);
  const sendMsg = async () => {
    if (!msgTxt.trim() || !selCase) return;
    setSending(true);
    try {
      const res = await API.post('/api/consultations/' + selCase._id + '/messages', {
        text: msgTxt.trim()
      });
      if (res.message) setMsgs(p => [...p, res.message]);
      setMsgTxt('');
    } catch (e) {
      if (toast) toast(e.message, 'err');
    }
    setSending(false);
  };
  const submitReport = async () => {
    if (!reportTxt.trim() || !selCase) return;
    setSubmitting(true);
    try {
      await API.patch('/api/consultations/' + selCase._id + '/status', {
        status: 'completed',
        report: reportTxt.trim()
      });
      if (toast) toast('Report submit ho gayi! Farmer ko notify kiya. ✅');
      setSelCase(p => p ? {
        ...p,
        status: 'completed',
        report: reportTxt.trim()
      } : p);
      setReportTxt('');
    } catch (e) {
      if (toast) toast(e.message, 'err');
    }
    setSubmitting(false);
  };
  if (!user) return <div className="wrap" style={{
    textAlign: 'center',
    padding: '80px 20px'
  }}>
      <div style={{
      fontSize: 60,
      marginBottom: 16
    }}>🔒</div>
      <div style={{
      fontFamily: "'Baloo 2',cursive",
      fontSize: 22,
      fontWeight: 900,
      color: 'var(--g1)'
    }}>Login Karein</div>
      <div style={{
      fontSize: 14,
      color: 'var(--tx3)',
      marginTop: 8
    }}>Cases dekhne ke liye login zaroor hai</div>
    </div>;
  return <div className="wrap">
      <div style={{
      fontFamily: "'Baloo 2',cursive",
      fontSize: 26,
      fontWeight: 900,
      color: 'var(--g1)',
      marginBottom: 20
    }}>
        📋 {user?.type === 'expert' ? 'Assigned Cases' : 'My Consultations'}
      </div>
      {loadingCases && <div style={{
      textAlign: 'center',
      padding: 40,
      color: 'var(--tx3)'
    }}>⏳ Loading cases...</div>}
      {!loadingCases && cases.length === 0 && <div style={{
      textAlign: 'center',
      padding: '60px 20px',
      color: 'var(--tx3)'
    }}>
          <div style={{
        fontSize: 50,
        marginBottom: 12
      }}>📭</div>
          <div style={{
        fontSize: 16,
        fontWeight: 700
      }}>Abhi koi case nahi hai</div>
          {user?.type !== 'expert' && <button className="btn btn-g btn-md" style={{
        marginTop: 16
      }} onClick={() => nav('consultation')}>🔬 Nayi Consultation</button>}
        </div>}
      <div style={{
      display: 'grid',
      gridTemplateColumns: cases.length > 0 ? '300px 1fr' : '1fr',
      gap: 20
    }}>
        {/* Case List */}
        {cases.length > 0 && <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 10
      }}>
            {cases.map(c => <div key={c._id} className={`card${selCase?._id === c._id ? ' card-hov' : ' card-hov'}`} style={{
          padding: '14px 16px',
          cursor: 'pointer',
          borderLeft: selCase?._id === c._id ? '3px solid var(--g4)' : '3px solid transparent'
        }} onClick={() => selectCase(c)}>
                <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            marginBottom: 6
          }}>
                  <span style={{
              fontSize: 24
            }}>{c.cropEmoji || '🌱'}</span>
                  <div>
                    <div style={{
                fontSize: 14,
                fontWeight: 800,
                color: 'var(--tx)'
              }}>{c.cropName}</div>
                    <div style={{
                fontSize: 11,
                color: 'var(--tx3)'
              }}>{new Date(c.createdAt).toLocaleDateString('en-IN')}</div>
                  </div>
                </div>
                <div style={{
            fontSize: 12,
            color: 'var(--tx2)',
            marginBottom: 4
          }}>{c.disease}</div>
                <span className={`badge ${c.status === 'completed' ? 'bg-g' : c.status === 'expert_assigned' ? 'bg-b' : 'bg-a'}`} style={{
            fontSize: 10
          }}>
                  {c.status === 'completed' ? '✅ Completed' : c.status === 'expert_assigned' ? '🔵 Expert Assigned' : '🟠 Pending'}
                </span>
              </div>)}
          </div>}

        {/* Case Detail */}
        {selCase && <div className="card" style={{
        padding: 24
      }}>
            <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          marginBottom: 20,
          paddingBottom: 16,
          borderBottom: '1.5px solid var(--br)'
        }}>
              <span style={{
            fontSize: 36
          }}>{selCase.cropEmoji || '🌱'}</span>
              <div>
                <div style={{
              fontFamily: "'Baloo 2',cursive",
              fontSize: 20,
              fontWeight: 900,
              color: 'var(--g1)'
            }}>{selCase.cropName} — {selCase.disease}</div>
                <div style={{
              fontSize: 13,
              color: 'var(--tx3)'
            }}>
                  Expert: {selCase.expertName} · {new Date(selCase.createdAt).toLocaleDateString('en-IN')}
                </div>
              </div>
            </div>

            <div style={{
          display: 'flex',
          gap: 8,
          marginBottom: 20
        }}>
              {['overview', 'chat', 'report'].map(t => <button key={t} onClick={() => setTab(t)} className="btn btn-sm" style={{
            background: tab === t ? 'var(--g4)' : 'var(--gp)',
            color: tab === t ? 'white' : 'var(--g2)',
            border: 'none'
          }}>
                  {t === 'overview' ? '📊 Overview' : t === 'chat' ? '💬 Chat' : '📄 Report'}
                </button>)}
            </div>

            {tab === 'overview' && <div>
                <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr 1fr',
            gap: 12,
            marginBottom: 16
          }}>
                  {[['AI Confidence', selCase.confidence + '%', 'var(--b3)'], ['Severity', 'Stage ' + selCase.severity + '/5', selCase.severity >= 3 ? 'var(--r2)' : selCase.severity === 2 ? 'var(--a2)' : 'var(--g4)'], ['Status', selCase.status, selCase.status === 'completed' ? 'var(--g4)' : 'var(--a2)']].map(([l, v, c]) => <div key={l} style={{
              background: 'var(--gb)',
              borderRadius: 10,
              padding: '12px 14px',
              textAlign: 'center'
            }}>
                      <div style={{
                fontFamily: "'Baloo 2',cursive",
                fontSize: 20,
                fontWeight: 900,
                color: c
              }}>{v}</div>
                      <div style={{
                fontSize: 11,
                color: 'var(--tx3)',
                marginTop: 3
              }}>{l}</div>
                    </div>)}
                </div>
                <div style={{
            background: 'var(--gp)',
            borderRadius: 10,
            padding: '14px 16px',
            marginBottom: 12
          }}>
                  <div style={{
              fontSize: 13,
              fontWeight: 700,
              color: 'var(--g1)',
              marginBottom: 6
            }}>📋 Answers from Farmer:</div>
                  {Object.entries(selCase.answers || {}).map(([k, v]) => <div key={k} style={{
              fontSize: 12,
              color: 'var(--tx2)',
              padding: '3px 0'
            }}>
                      <strong>{k}:</strong> {v?.label || String(v)}
                    </div>)}
                </div>
                {selCase.report && <div style={{
            background: 'var(--tp)',
            borderRadius: 10,
            padding: '14px 16px',
            border: '1.5px solid var(--t3)'
          }}>
                    <div style={{
              fontSize: 13,
              fontWeight: 700,
              color: 'var(--t1)',
              marginBottom: 6
            }}>✅ Expert Report:</div>
                    <div style={{
              fontSize: 13,
              color: 'var(--tx)',
              lineHeight: 1.6
            }}>{selCase.report}</div>
                  </div>}
              </div>}

            {tab === 'chat' && <div>
                <div style={{
            height: 300,
            overflowY: 'auto',
            marginBottom: 14,
            padding: '8px 0'
          }}>
                  {msgs.length === 0 && <div style={{
              textAlign: 'center',
              color: 'var(--tx4)',
              padding: 30,
              fontSize: 13
            }}>
                    Abhi koi message nahi. Pehla message bhejo!
                  </div>}
                  {msgs.map((m, i) => (() => {
              const fm = formatChatMessage(m, 'en-IN');
              const mine = fm.from === user?.type;
              return <div key={m._id || i} style={{
                display: 'flex',
                justifyContent: mine ? 'flex-end' : 'flex-start',
                marginBottom: 10
              }}>
                          <div style={{
                  maxWidth: '75%',
                  background: mine ? 'var(--g4)' : 'white',
                  color: mine ? 'white' : 'var(--tx)',
                  borderRadius: 12,
                  padding: '10px 14px',
                  border: mine ? 'none' : '1.5px solid var(--br)',
                  fontSize: 13,
                  lineHeight: 1.5
                }}>
                        <div style={{
                    fontSize: 10,
                    opacity: .7,
                    marginBottom: 4,
                    fontWeight: 600
                  }}>
                          {fm.senderName} · {new Date(m.createdAt).toLocaleTimeString('en-IN', {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                        </div>
                        <ChatMessageBody msg={fm} />
                      </div>
                        </div>;
            })())}
                </div>
                <div style={{
            display: 'flex',
            gap: 10
          }}>
                  <input className="finp" style={{
              flex: 1
            }} value={msgTxt} onChange={e => setMsgTxt(e.target.value)} placeholder="Message type karo..." onKeyDown={e => e.key === 'Enter' && sendMsg()} />
                  <button className="btn btn-g btn-md" onClick={sendMsg} disabled={sending || !msgTxt.trim()}>
                    {sending ? '..' : 'Bhejo →'}
                  </button>
                </div>
              </div>}

            {tab === 'report' && <div>
                {selCase.report ? <div style={{
            background: 'var(--gp)',
            borderRadius: 12,
            padding: '16px 18px',
            marginBottom: 16
          }}>
                    <div style={{
              fontWeight: 800,
              color: 'var(--g1)',
              marginBottom: 8
            }}>✅ Submitted Report:</div>
                    <div style={{
              fontSize: 13,
              color: 'var(--tx)',
              lineHeight: 1.6,
              whiteSpace: 'pre-wrap'
            }}>{selCase.report}</div>
                    
                    <div style={{
              marginTop: 16,
              paddingTop: 16,
              borderTop: '1px solid var(--br)'
            }}>
                      <div style={{
                fontWeight: 800,
                color: 'var(--g1)',
                marginBottom: 8
              }}>📸 Uploaded Photos:</div>
                      <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                gap: 10
              }}>
                        {(selCase.photoUrls && selCase.photoUrls.length > 0 ? selCase.photoUrls : ['https://placehold.co/400x400/e6f4ea/1e7e42?text=Sample+Upload']).map((url, i) => <img key={i} src={url} alt={`Photo ${i + 1}`} style={{
                  width: '100%',
                  aspectRatio: '4/3',
                  objectFit: 'cover',
                  borderRadius: 8,
                  border: '1px solid var(--br)'
                }} crossOrigin="anonymous" />)}
                      </div>
                    </div>
                  </div> : user?.type === 'expert' ? <div>
                      <div style={{
              fontSize: 14,
              fontWeight: 700,
              color: 'var(--tx)',
              marginBottom: 8
            }}>
                        📝 Farmer ke liye report likhein:
                      </div>
                      <textarea className="ftxt" rows={6} value={reportTxt} onChange={e => setReportTxt(e.target.value)} placeholder="Disease analysis, treatment plan, medicines, follow-up instructions..." />
                      <button className="btn btn-g btn-full" style={{
              marginTop: 12
            }} onClick={submitReport} disabled={submitting || !reportTxt.trim()}>
                        {submitting ? 'Submitting...' : '✅ Report Submit Karo'}
                      </button>
                    </div> : <div style={{
            textAlign: 'center',
            padding: '40px 20px',
            color: 'var(--tx3)'
          }}>
                      <div style={{
              fontSize: 40,
              marginBottom: 12
            }}>⏳</div>
                      Expert abhi report likh raha hai. Thodi der mein check karein.
                    </div>}
              </div>}
          </div>}
      </div>
    </div>;
}
