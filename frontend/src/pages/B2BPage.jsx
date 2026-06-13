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
   5. B2B DATA INTELLIGENCE 💼
════════════════════════════════════════════════════════════════ */
function B2BPage({
  nav,
  toast
}) {
  const [lead, setLead] = useState({
    company: '',
    contact: '',
    mobile: '',
    email: '',
    type: '',
    msg: ''
  });
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const submitLead = async () => {
    if (!lead.company || !lead.mobile) {
      toast('Company aur mobile zaroor bharo', 'err');
      return;
    }
    setSubmitting(true);
    try {
      await API.post('/api/consultations', {
        cropId: 'b2b',
        cropName: 'B2B Lead: ' + lead.company,
        cropEmoji: '💼',
        method: 'manual',
        disease: 'B2B Enquiry — ' + lead.type,
        confidence: 100,
        severity: 1,
        answers: {
          company: lead.company,
          contact: lead.contact,
          mobile: lead.mobile,
          email: lead.email,
          type: lead.type,
          msg: lead.msg
        }
      });
      setSubmitted(true);
      toast('B2B enquiry submit ho gayi! Team 24 ghante mein contact karegi. ✅');
    } catch (e) {
      toast('Submit fail — dobara try karo', 'err');
    }
    setSubmitting(false);
  };
  const districts = ['Pune', 'Nashik', 'Aurangabad', 'Nagpur', 'Solapur', 'Kolhapur', 'Satara', 'Sangli'];
  const diseases = ['Early Blight', 'Late Blight', 'Powdery Mildew', 'Leaf Rust', 'Bacterial Wilt', 'Downy Mildew', 'Fusarium', 'Anthracnose'];
  const heatData = Array.from({
    length: 56
  }, () => Math.floor(Math.random() * 5));
  const heatColors = ['#eaf7ef', '#b8d9c2', '#7dd4a0', '#4dbd7a', '#1e7e42'];
  return <div className="wrap-md">
      <div style={{
      display: 'flex',
      gap: 10,
      alignItems: 'center',
      marginBottom: 5
    }}>
        <div style={{
        padding: '4px 12px',
        background: 'var(--bp)',
        borderRadius: 100,
        fontSize: 12,
        fontWeight: 700,
        color: 'var(--b2)'
      }}>💼 B2B Portal</div>
        <div style={{
        fontSize: 12,
        color: 'var(--tx3)'
      }}>Agri companies & government access</div>
      </div>
      <div style={{
      fontFamily: "'Baloo 2',cursive",
      fontSize: 26,
      fontWeight: 900,
      color: 'var(--b1)',
      marginBottom: 18
    }}>Disease Intelligence Dashboard</div>

      {/* KPI Row */}
      <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(4,1fr)',
      gap: 14,
      marginBottom: 24
    }}>
        {[{
        n: '47,320',
        l: 'Active Farmers',
        i: '👨‍🌾'
      }, {
        n: '1,24,580',
        l: 'Disease Cases',
        i: '🦠'
      }, {
        n: '94.2%',
        l: 'AI Accuracy',
        i: '🎯'
      }, {
        n: '58',
        l: 'Districts Covered',
        i: '📍'
      }].map(s => <div key={s.l} className="b2b-stat">
            <div style={{
          fontSize: 24,
          marginBottom: 6
        }}>{s.i}</div>
            <div className="b2b-n">{s.n}</div>
            <div className="b2b-l">{s.l}</div>
          </div>)}
      </div>

      <div className="dash-2">
        <div>
          {/* Disease Heatmap */}
          <div className="card" style={{
          padding: 20,
          marginBottom: 18
        }}>
            <div style={{
            fontSize: 15,
            fontWeight: 800,
            color: 'var(--b1)',
            marginBottom: 5
          }}>🗺️ District Disease Heatmap</div>
            <div style={{
            fontSize: 12,
            color: 'var(--tx3)',
            marginBottom: 13
          }}>Maharashtra — Last 30 days</div>
            <div style={{
            display: 'flex',
            gap: 6,
            marginBottom: 8,
            flexWrap: 'wrap'
          }}>
              {districts.map(d => <span key={d} style={{
              fontSize: 11,
              padding: '2px 8px',
              background: 'var(--bp)',
              borderRadius: 100,
              fontWeight: 600,
              color: 'var(--b2)'
            }}>{d}</span>)}
            </div>
            <div className="heatmap-grid">
              {heatData.map((v, i) => <div key={i} className="hm-cell" style={{
              background: heatColors[v]
            }} title={`Cases: ${v * 23}`} />)}
            </div>
            <div style={{
            display: 'flex',
            gap: 8,
            alignItems: 'center',
            marginTop: 8
          }}>
              {heatColors.map((c, i) => <div key={i} style={{
              width: 16,
              height: 10,
              background: c,
              borderRadius: 2
            }} />)}
              <span style={{
              fontSize: 11,
              color: 'var(--tx3)'
            }}>Low → High cases</span>
            </div>
          </div>

          {/* Top Diseases */}
          <div className="card" style={{
          padding: 20
        }}>
            <div style={{
            fontSize: 15,
            fontWeight: 800,
            color: 'var(--b1)',
            marginBottom: 14
          }}>📊 Top Diseases This Season</div>
            {diseases.slice(0, 5).map((d, i) => {
            const pct = Math.floor(85 - i * 12);
            return <div key={d} className="disease-bar-row">
                  <div style={{
                width: 120,
                fontSize: 12.5,
                fontWeight: 600,
                color: 'var(--tx)',
                flexShrink: 0
              }}>{d}</div>
                  <div className="disease-bar-track"><div className="disease-bar-fill" style={{
                  width: `${pct}%`
                }} /></div>
                  <div style={{
                fontSize: 12,
                fontWeight: 800,
                color: 'var(--b3)',
                width: 40,
                textAlign: 'right'
              }}>{pct}%</div>
                </div>;
          })}
          </div>
        </div>

        <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 18
      }}>
          {/* Revenue opportunity */}
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
          }}>💰 Data API Pricing</div>
            {[['Basic Plan', '₹15,000/mo', 'District-level data'], ['Pro Plan', '₹45,000/mo', 'Real-time + crop-wise'], ['Enterprise', 'Custom', 'Full API + white-label']].map(([p, pr, d]) => <div key={p} style={{
            padding: '11px 13px',
            background: 'white',
            borderRadius: 10,
            marginBottom: 9,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
                <div>
                  <div style={{
                fontSize: 13.5,
                fontWeight: 700,
                color: 'var(--b1)'
              }}>{p}</div>
                  <div style={{
                fontSize: 11,
                color: 'var(--tx3)'
              }}>{d}</div>
                </div>
                <div style={{
              fontFamily: "'Baloo 2',cursive",
              fontSize: 16,
              fontWeight: 900,
              color: 'var(--b3)'
            }}>{pr}</div>
              </div>)}
            <button className="btn btn-b btn-sm" style={{
            width: '100%',
            marginTop: 4
          }} onClick={() => toast('Sales team se contact kiya jayega! 📧', 'inf')}>
              📧 Contact Sales Team
            </button>
          </div>

          {/* Recent Alerts */}
          <div className="card" style={{
          padding: 20
        }}>
            <div style={{
            fontSize: 15,
            fontWeight: 800,
            color: 'var(--b1)',
            marginBottom: 13
          }}>🚨 Recent Outbreak Alerts</div>
            {[{
            d: 'Late Blight',
            loc: 'Nashik (5 blocks)',
            n: 124,
            sev: 'High'
          }, {
            d: 'Early Blight',
            loc: 'Pune (3 blocks)',
            n: 87,
            sev: 'Med'
          }, {
            d: 'Powdery Mildew',
            loc: 'Satara',
            n: 43,
            sev: 'Low'
          }].map(a => <div key={a.d} style={{
            padding: '10px 0',
            borderBottom: '1px solid var(--gp)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
                <div>
                  <div style={{
                fontSize: 13,
                fontWeight: 700,
                color: 'var(--tx)'
              }}>{a.d}</div>
                  <div style={{
                fontSize: 11.5,
                color: 'var(--tx3)'
              }}>📍 {a.loc}</div>
                </div>
                <div style={{
              textAlign: 'right'
            }}>
                  <div style={{
                fontSize: 13,
                fontWeight: 800,
                color: 'var(--b3)'
              }}>{a.n} cases</div>
                  <span className={`badge ${a.sev === 'High' ? 'bg-r' : a.sev === 'Med' ? 'bg-a' : 'bg-g'}`} style={{
                fontSize: 10
              }}>{a.sev}</span>
                </div>
              </div>)}
          </div>
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   6. INPUT MARKETPLACE 📦
════════════════════════════════════════════════════════════════ */
