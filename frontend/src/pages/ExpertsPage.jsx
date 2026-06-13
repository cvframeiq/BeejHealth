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
import CaseDetailPage from './CaseDetailPage.jsx';
import ChatPage from './ChatPage.jsx';
import BookingPage from './BookingPage.jsx';
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
   EXPERT DASHBOARD
════════════════════════════════════════════════════════════════ */
/* ════════════════════════════════════════════════════════════════
   EXPERTS PAGE
════════════════════════════════════════════════════════════════ */
function ExpertsPage({
  user,
  nav,
  toast
}) {
  const [avail, setAvail] = useState(false);
  const [spec, setSpec] = useState('');
  const [apiExperts, setApiExperts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [profileExpert, setProfileExpert] = useState(null);
  useEffect(() => {
    fetch('/api/experts').then(r => r.json()).then(d => {
      if (d.experts && d.experts.length > 0) setApiExperts(d.experts);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);
  const baseList = apiExperts.length > 0 ? apiExperts.map(e => ({
    id: e._id,
    name: e.name,
    spec: e.spec || 'Agricultural Expert',
    exp: '5+ yrs',
    langs: (e.langs || 'Hindi').split(','),
    price: e.fee || 500,
    rating: e.rating || 4.5,
    reviews: e.totalCases || 0,
    online: e.available || false,
    emoji: '👨‍🔬',
    crops: e.crops || 'All Crops',
    cases: e.totalCases || 0,
    response: '45 min',
    success: 95
  })) : EXPERTS;
  const list = baseList.filter(e => {
    if (avail && !e.online) return false;
    if (spec && e.spec !== spec) return false;
    return true;
  });
  return <div className="wrap">
      <div style={{
      textAlign: 'center',
      marginBottom: 32
    }}>
        <div style={{
        fontFamily: "'Baloo 2',cursive",
        fontSize: 32,
        fontWeight: 900,
        color: 'var(--g1)'
      }}>👨‍⚕️ Certified Agricultural Experts</div>
        <div style={{
        fontSize: 15,
        color: 'var(--tx2)',
        marginTop: 6
      }}>Sahi specialist dhundho — disease, soil ya crop ke liye</div>
      </div>
      <div className="exp-filters">
        <select className="flt-sel" value={spec} onChange={e => setSpec(e.target.value)}>
          <option value="">All Specializations</option>
          <option>Plant Pathologist</option><option>Horticulture Expert</option>
          <option>Soil Scientist</option><option>Crop Scientist</option>
        </select>
        {['Crop Type', 'Language', 'Min Rating'].map(f => <select key={f} className="flt-sel"><option>{f}</option></select>)}
        <div style={{
        marginLeft: 'auto',
        display: 'flex',
        alignItems: 'center',
        gap: 9,
        fontSize: 13.5,
        fontWeight: 600,
        color: 'var(--tx2)'
      }}>
          <label className="sw"><input type="checkbox" checked={avail} onChange={e => setAvail(e.target.checked)} /><span className="sw-sl" /></label>
          Available Now Only
        </div>
      </div>
      <div className="experts-grid">
        {list.map(e => <div key={e.id} className="exp-card">
            <div style={{
          display: 'flex',
          gap: 13,
          marginBottom: 14
        }}>
              <div className="exp-av">{e.emoji}{e.online && <div className="on-dot" />}</div>
              <div style={{
            flex: 1
          }}>
                <div className="exp-nm">{e.name}</div>
                <div className="exp-sp">{e.spec}</div>
                <div className="exp-rat">⭐ {e.rating} <span style={{
                color: 'var(--tx3)',
                fontWeight: 400
              }}>({e.reviews})</span></div>
              </div>
            </div>
            <div className="exp-det">⏱️ {e.exp} experience</div>
            <div className="exp-det">🗣️ {e.langs.join(', ')}</div>
            <div className="exp-det">🌾 {e.crops}</div>
            <div className="exp-det">✅ {e.cases} cases • ⚡ {e.response} avg</div>
            <div className="exp-pr">₹{e.price} <span>/ consultation</span></div>
            <div style={{
          display: 'flex',
          gap: 9
        }}>
              <button className="btn btn-out btn-sm" style={{
            flex: 1
          }} onClick={() => setProfileExpert(e)}>👤 Profile</button>
              <button className="btn btn-g btn-sm" style={{
            flex: 2
          }} onClick={() => {
            if (!user) {
              toast('Pehle login karein!', 'err');
              return;
            }
            localStorage.setItem('bh_sel_expert', JSON.stringify({
              id: e.id,
              name: e.name,
              spec: e.spec,
              fee: e.price,
              rating: e.rating
            }));
            nav('booking');
          }}>✅ Select Expert</button>
            </div>
          </div>)}
      </div>
      <div style={{
      textAlign: 'center',
      marginTop: 30
    }}>
        <button className="btn btn-out btn-md">Load More Experts</button>
      </div>

      {/* Expert Profile Modal */}
      {profileExpert && <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0,0,0,.5)',
      zIndex: 999,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 20
    }} onClick={() => setProfileExpert(null)}>
          <div style={{
        background: 'white',
        borderRadius: 'var(--rad)',
        padding: 28,
        maxWidth: 420,
        width: '100%',
        boxShadow: 'var(--sh2)'
      }} onClick={e => e.stopPropagation()}>
            <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: 18
        }}>
              <div style={{
            display: 'flex',
            gap: 13,
            alignItems: 'center'
          }}>
                <div style={{
              width: 56,
              height: 56,
              borderRadius: '50%',
              background: 'linear-gradient(135deg,var(--g5),var(--g6))',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 26,
              position: 'relative'
            }}>
                  {profileExpert.emoji || '👨‍🔬'}
                  {profileExpert.online && <div className="on-dot" />}
                </div>
                <div>
                  <div style={{
                fontSize: 18,
                fontWeight: 900,
                color: 'var(--tx)'
              }}>{profileExpert.name}</div>
                  <div style={{
                fontSize: 13,
                color: 'var(--g3)',
                fontWeight: 700
              }}>{profileExpert.spec}</div>
                  <div style={{
                fontSize: 12.5,
                color: 'var(--tx3)'
              }}>⭐ {profileExpert.rating} • {profileExpert.reviews || profileExpert.cases || 0} cases</div>
                </div>
              </div>
              <button style={{
            background: 'none',
            border: 'none',
            fontSize: 20,
            cursor: 'pointer',
            color: 'var(--tx3)'
          }} onClick={() => setProfileExpert(null)}>✕</button>
            </div>
            <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 10,
          marginBottom: 16
        }}>
              {[['💰 Fee', `₹${profileExpert.price || profileExpert.fee || 500}/consult`], ['⏱️ Experience', profileExpert.exp || '5+ yrs'], ['🗣️ Languages', (profileExpert.langs || ['Hindi']).join?.(', ') || profileExpert.langs], ['✅ Success Rate', `${profileExpert.success || 95}%`]].map(([k, v]) => <div key={k} style={{
            padding: '10px 12px',
            background: 'var(--gp)',
            borderRadius: 9
          }}>
                  <div style={{
              fontSize: 11,
              color: 'var(--tx3)',
              fontWeight: 700
            }}>{k}</div>
                  <div style={{
              fontSize: 13.5,
              color: 'var(--tx)',
              fontWeight: 800,
              marginTop: 2
            }}>{v}</div>
                </div>)}
            </div>
            <div style={{
          padding: '12px 14px',
          background: 'var(--gp)',
          borderRadius: 9,
          fontSize: 13,
          color: 'var(--tx2)',
          lineHeight: 1.65,
          marginBottom: 16
        }}>
              {profileExpert.bio || `${profileExpert.name} ek certified agricultural expert hain jo ${profileExpert.spec || 'crop diseases'} mein specialization rakhte hain. Aapki fasal ki problems ke liye expert guidance milegi.`}
            </div>
            <div style={{
          display: 'flex',
          gap: 10
        }}>
              <button className="btn btn-out btn-md" style={{
            flex: 1
          }} onClick={() => setProfileExpert(null)}>Wapas</button>
              <button className="btn btn-g btn-md" style={{
            flex: 2
          }} onClick={() => {
            if (!user) {
              toast('Pehle login karein!', 'err');
              return;
            }
            localStorage.setItem('bh_sel_expert', JSON.stringify({
              id: profileExpert.id,
              name: profileExpert.name,
              spec: profileExpert.spec,
              fee: profileExpert.price || profileExpert.fee || 500,
              rating: profileExpert.rating
            }));
            setProfileExpert(null);
            nav('booking');
          }}>✅ Select Expert →</button>
            </div>
          </div>
        </div>}
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   BOOKING PAGE
════════════════════════════════════════════════════════════════ */
