import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import { API, saveSession, clearSession, loadSession } from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody } from '../utils/helpers.jsx';
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary } from '../utils/constants.jsx';

import BeejHealthApp from './BeejHealthApp.jsx';
import RobotAnalyticsPage from '../pages/RobotAnalyticsPage.jsx';
import RobotMaintenancePage from '../pages/RobotMaintenancePage.jsx';
import RobotControlPage from '../pages/RobotControlPage.jsx';
import RobotMapPage from '../pages/RobotMapPage.jsx';
import RobotCameraPage from '../pages/RobotCameraPage.jsx';
import RobotSprayPage from '../pages/RobotSprayPage.jsx';
import RobotDashboard from '../pages/RobotDashboard.jsx';
import GovtMapPage from '../pages/GovtMapPage.jsx';
import InsurancePage from '../pages/InsurancePage.jsx';
import MarketplacePage from '../pages/MarketplacePage.jsx';
import B2BPage from '../pages/B2BPage.jsx';
import SoilSensorPage from '../pages/SoilSensorPage.jsx';
import ForecastPage from '../pages/ForecastPage.jsx';
import SatellitePage from '../pages/SatellitePage.jsx';
import VoiceInputPage from '../pages/VoiceInputPage.jsx';
import SettingsPage from '../pages/SettingsPage.jsx';
import EarningsPage from '../pages/EarningsPage.jsx';
import ProfilePage from '../pages/ProfilePage.jsx';
import SupportPage from '../pages/SupportPage.jsx';
import NotifPage from '../pages/NotifPage.jsx';
import MyFarmPage from '../pages/MyFarmPage.jsx';
import CaseDetailPage from '../pages/CaseDetailPage.jsx';
import ChatPage from '../pages/ChatPage.jsx';
import BookingPage from '../pages/BookingPage.jsx';
import ExpertsPage from '../pages/ExpertsPage.jsx';
import FarmerDash from '../pages/FarmerDash.jsx';
import ExpertDash from '../pages/ExpertDash.jsx';
import MyConsultPage from '../pages/MyConsultPage.jsx';
import AIReportPage from '../pages/AIReportPage.jsx';
import ConsultPage from '../pages/ConsultPage.jsx';
import HomePage from '../pages/HomePage.jsx';
import ExpertOnboarding from './ExpertOnboarding.jsx';
import AuthModal from './AuthModal.jsx';

export default /* ════════════════════════════════════════════════════════════════
   FARMER ONBOARDING
════════════════════════════════════════════════════════════════ */
function FarmerOnboarding({
  user,
  onDone,
  setUser
}) {
  const [step, setStep] = useState(1);
  const [sel, setSel] = useState({
    district: user?.district || 'Pune',
    crops: ['tomato', 'wheat'],
    notifs: {
      disease: true,
      weather: true,
      market: true,
      expert: true
    }
  });
  const [saving, setSaving] = useState(false);
  const steps = 4;
  const finishOnboarding = async () => {
    setSaving(true);
    try {
      const res = await API.patch('/api/auth/profile', {
        district: sel.district,
        crops: sel.crops,
        langs: 'Hindi'
      });
      if (res.user) {
        saveSession(localStorage.getItem('bh_token'), res.user);
        if (setUser) setUser(res.user);
      }
    } catch (e) {
      console.warn('Onboarding save:', e.message);
    }
    setSaving(false);
    onDone();
  };
  return <div className="ob-wrap">
      <div className="ob-box">
        <div className="ob-head">
          <div style={{
          fontSize: 12,
          opacity: .76,
          marginBottom: 5
        }}>🌱 BeejHealth Setup — {step}/{steps}</div>
          <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 24,
          fontWeight: 900
        }}>Swagat Hai, {user?.name?.split(' ')?.[0] || 'Kisan'}! 🎉</div>
          <div style={{
          fontSize: 13.5,
          opacity: .8,
          marginTop: 3
        }}>Sirf 2 minute — platform taiyar karte hain</div>
        </div>
        <div style={{
        padding: '0 30px 4px'
      }}>
          <div className="ob-prog" style={{
          margin: '0',
          paddingTop: 18,
          paddingBottom: 4
        }}>
            {Array.from({
            length: steps
          }, (_, i) => <div key={i} className={`ob-step${i < step ? ' done' : ''}`} />)}
          </div>
        </div>
        <div className="ob-body">
          {step === 1 && <div className="slide-up">
            <div className="ob-sec-t">📍 Aapka Location</div>
            <div className="ob-sec-p">Sahi experts aur disease alerts ke liye</div>
            <div className="frow">
              <div className="fgrp"><label className="flbl">District</label>
                <select className="fsel" value={sel.district} onChange={e => setSel(p => ({
                ...p,
                district: e.target.value
              }))}>
                  {DISTRICTS.map(d => <option key={d}>{d}</option>)}
                </select>
              </div>
              <div className="fgrp"><label className="flbl">Taluka</label>
                <select className="fsel">{TALUKAS.map(t => <option key={t}>{t}</option>)}</select>
              </div>
            </div>
            <button style={{
            width: '100%',
            padding: '12px',
            background: 'var(--gp)',
            border: '2px dashed var(--br2)',
            borderRadius: 10,
            fontSize: 13.5,
            fontWeight: 600,
            color: 'var(--g2)',
            cursor: 'pointer',
            marginBottom: 6
          }}>
              📍 GPS Se Location Use Karo
            </button>
          </div>}
          {step === 2 && <div className="slide-up">
            <div className="ob-sec-t">🌾 Aapki Fasalein</div>
            <div className="ob-sec-p">Kaunsi crops ughate hain? (Multiple select)</div>
            <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4,1fr)',
            gap: 9
          }}>
              {CROPS.slice(0, 8).map(c => <div key={c.id} onClick={() => setSel(p => ({
              ...p,
              crops: p.crops.includes(c.id) ? p.crops.filter(x => x !== c.id) : [...p.crops, c.id]
            }))} style={{
              padding: '11px 6px',
              borderRadius: 10,
              border: `2px solid ${sel.crops.includes(c.id) ? 'var(--g4)' : 'var(--br)'}`,
              background: sel.crops.includes(c.id) ? 'var(--gp)' : 'white',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all .18s'
            }}>
                  <div style={{
                fontSize: 26,
                marginBottom: 5
              }}>{c.emoji}</div>
                  <div style={{
                fontSize: 11,
                fontWeight: 700,
                color: 'var(--tx)'
              }}>{c.name.split(' ')[0]}</div>
                </div>)}
            </div>
          </div>}
          {step === 3 && <div className="slide-up">
            <div className="ob-sec-t">🔔 Notification Preferences</div>
            <div className="ob-sec-p">Kaunsi alerts chahiye?</div>
            {[{
            k: 'disease',
            i: '🦠',
            l: 'Disease Alerts',
            s: 'Nearby outbreak mein'
          }, {
            k: 'weather',
            i: '🌦️',
            l: 'Weather Warnings',
            s: 'Spray timing ke liye'
          }, {
            k: 'market',
            i: '📈',
            l: 'Market Price Alerts',
            s: 'Jab crop ka bhav badhe'
          }, {
            k: 'expert',
            i: '💬',
            l: 'Expert Replies',
            s: 'Consultation updates'
          }].map(n => <div key={n.k} style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '12px 0',
            borderBottom: '1px solid var(--gp)'
          }}>
                <div style={{
              display: 'flex',
              gap: 11,
              alignItems: 'center'
            }}>
                  <span style={{
                fontSize: 20
              }}>{n.i}</span>
                  <div>
                    <div style={{
                  fontSize: 14,
                  fontWeight: 700,
                  color: 'var(--tx)'
                }}>{n.l}</div>
                    <div style={{
                  fontSize: 12,
                  color: 'var(--tx3)'
                }}>{n.s}</div>
                  </div>
                </div>
                <label className="sw">
                  <input type="checkbox" checked={sel.notifs[n.k]} onChange={e => setSel(p => ({
                ...p,
                notifs: {
                  ...p.notifs,
                  [n.k]: e.target.checked
                }
              }))} />
                  <span className="sw-sl" />
                </label>
              </div>)}
          </div>}
          {step === 4 && <div className="slide-up" style={{
          textAlign: 'center'
        }}>
            <div style={{
            fontSize: 68,
            animation: 'bounce 1.2s infinite',
            marginBottom: 14
          }}>🎉</div>
            <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 24,
            fontWeight: 900,
            color: 'var(--g1)',
            marginBottom: 7
          }}>Sab Ready Hai!</div>
            <div style={{
            fontSize: 14,
            color: 'var(--tx2)',
            lineHeight: 1.75,
            marginBottom: 20
          }}>Aapka BeejHealth account fully setup ho gaya.</div>
            <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 10
          }}>
              {[['🔬', 'AI Diagnosis', 'Photo se instant disease'], ['👨‍⚕️', '200+ Experts', 'Certified specialists'], ['📊', 'Farm Analytics', 'Complete tracking'], ['🌦️', 'Smart Alerts', 'Weather + disease']].map(([ic, t, d]) => <div key={t} style={{
              padding: 13,
              background: 'var(--gp)',
              borderRadius: 10,
              textAlign: 'left'
            }}>
                  <div style={{
                fontSize: 22,
                marginBottom: 5
              }}>{ic}</div>
                  <div style={{
                fontSize: 13,
                fontWeight: 700,
                color: 'var(--g1)'
              }}>{t}</div>
                  <div style={{
                fontSize: 11,
                color: 'var(--tx3)',
                marginTop: 2
              }}>{d}</div>
                </div>)}
            </div>
          </div>}
          <div style={{
          display: 'flex',
          gap: 9,
          marginTop: 22
        }}>
            {step > 1 && step < 4 && <button className="btn btn-out btn-md" style={{
            flex: 1
          }} onClick={() => setStep(p => p - 1)}>← Wapas</button>}
            <button className="btn btn-g btn-md" style={{
            flex: 2
          }} onClick={() => step < 4 ? setStep(p => p + 1) : finishOnboarding()} disabled={saving}>
              {step === 4 ? '🚀 Dashboard Par Jao' : 'Aage Badho →'}
            </button>
          </div>
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   EXPERT ONBOARDING
════════════════════════════════════════════════════════════════ */
