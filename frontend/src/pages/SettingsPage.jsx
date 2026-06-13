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
   SETTINGS PAGE
════════════════════════════════════════════════════════════════ */
function SettingsPage({
  user,
  setUser,
  nav,
  toast
}) {
  // Profile fields
  const [nm, setNm] = useState(user?.name || '');
  const [em, setEm] = useState(user?.email || '');
  const [dt, setDt] = useState(user?.district || '');
  const [vl, setVl] = useState(user?.village || '');
  const [sp, setSp] = useState(user?.spec || '');
  const [fee, setFee] = useState(user?.fee || '');
  const [lang, setLang] = useState(user?.langs || 'Hindi');
  const [savingProfile, setSavingProfile] = useState(false);

  // Password fields
  const [oldPwd, setOldPwd] = useState('');
  const [newPwd, setNewPwd] = useState('');
  const [cnfPwd, setCnfPwd] = useState('');
  const [savingPwd, setSavingPwd] = useState(false);

  // Notification prefs
  const [notifs, setNotifs] = useState({
    disease: true,
    weather: true,
    market: true,
    expert: true
  });
  const [activeTab, setActiveTab] = useState('profile');
  const isEx = user?.type === 'expert';
  const saveProfile = async () => {
    if (!nm.trim()) {
      toast('Naam required hai', 'err');
      return;
    }
    setSavingProfile(true);
    try {
      const payload = {
        name: nm.trim(),
        email: em.trim(),
        district: dt,
        village: vl,
        langs: lang
      };
      if (isEx) {
        payload.spec = sp;
        payload.fee = Number(fee) || 0;
      }
      const res = await API.patch('/api/auth/profile', payload);
      if (res.user) {
        saveSession(localStorage.getItem('bh_token'), res.user);
        if (setUser) setUser(res.user);
      }
      toast('Profile save ho gayi! ✅');
    } catch (e) {
      toast(e.message, 'err');
    }
    setSavingProfile(false);
  };
  const savePassword = async () => {
    if (!oldPwd) {
      toast('Purana password daalein', 'err');
      return;
    }
    if (newPwd.length < 8) {
      toast('Naya password 8+ characters ka hona chahiye', 'err');
      return;
    }
    if (newPwd !== cnfPwd) {
      toast('Passwords match nahi kar rahe', 'err');
      return;
    }
    setSavingPwd(true);
    try {
      await API.patch('/api/auth/password', {
        oldPassword: oldPwd,
        newPassword: newPwd
      });
      setOldPwd('');
      setNewPwd('');
      setCnfPwd('');
      toast('Password successfully change ho gaya! ✅');
    } catch (e) {
      toast(e.message, 'err');
    }
    setSavingPwd(false);
  };
  const doLogout = () => {
    clearSession();
    localStorage.removeItem('bh_latest_consult');
    localStorage.removeItem('bh_chat_consult');
    localStorage.removeItem('bh_latest_crop');
    localStorage.removeItem('bh_view_consult');
    localStorage.removeItem('bh_sel_expert');
    if (setUser) setUser(null);
    nav('home');
    toast('Aap logout ho gaye 👋', 'inf');
  };
  const tabs = [{
    id: 'profile',
    l: '👤 Profile'
  }, {
    id: 'password',
    l: '🔑 Password'
  }, {
    id: 'notifs',
    l: '🔔 Notifications'
  }, {
    id: 'account',
    l: '⚙️ Account'
  }];
  return <div className="wrap-sm">
      <div style={{
      fontFamily: "'Baloo 2',cursive",
      fontSize: 26,
      fontWeight: 900,
      color: 'var(--g1)',
      marginBottom: 22
    }}>⚙️ Settings</div>

      {/* Tab bar */}
      <div style={{
      display: 'flex',
      gap: 6,
      marginBottom: 24,
      flexWrap: 'wrap'
    }}>
        {tabs.map(t => <button key={t.id} onClick={() => setActiveTab(t.id)} className="btn btn-sm" style={{
        background: activeTab === t.id ? 'var(--g4)' : 'white',
        color: activeTab === t.id ? 'white' : 'var(--tx2)',
        border: `1.5px solid ${activeTab === t.id ? 'var(--g4)' : 'var(--br)'}`,
        fontWeight: 700
      }}>
            {t.l}
          </button>)}
      </div>

      {/* ── PROFILE TAB ── */}
      {activeTab === 'profile' && <div className="card" style={{
      padding: 24
    }}>
          <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 18
      }}>👤 Profile Information</div>

          {/* Avatar */}
          <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        marginBottom: 22,
        padding: '14px 16px',
        background: 'var(--gp)',
        borderRadius: 12
      }}>
            <div style={{
          width: 56,
          height: 56,
          borderRadius: '50%',
          background: isEx ? 'var(--b3)' : 'var(--g4)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          fontSize: 20,
          fontWeight: 900,
          flexShrink: 0
        }}>
              {(nm || user?.name || 'U').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()}
            </div>
            <div>
              <div style={{
            fontSize: 15,
            fontWeight: 800,
            color: 'var(--tx)'
          }}>{nm || user?.name}</div>
              <div style={{
            fontSize: 12,
            color: 'var(--tx3)'
          }}>{isEx ? '👨‍⚕️ Agricultural Expert' : '🌾 Farmer'} • {user?.mobile}</div>
            </div>
          </div>

          <div className="frow">
            <div className="fgrp">
              <label className="flbl">Naam *</label>
              <input className="finp" value={nm} onChange={e => setNm(e.target.value)} placeholder="Aapka poora naam" />
            </div>
            <div className="fgrp">
              <label className="flbl">Email</label>
              <input className="finp" type="email" value={em} onChange={e => setEm(e.target.value)} placeholder="email@example.com" />
            </div>
          </div>
          <div className="frow">
            <div className="fgrp">
              <label className="flbl">District</label>
              <input className="finp" value={dt} onChange={e => setDt(e.target.value)} placeholder="Aapka district" />
            </div>
            <div className="fgrp">
              <label className="flbl">Village / Gaon</label>
              <input className="finp" value={vl} onChange={e => setVl(e.target.value)} placeholder="Gaon ka naam" />
            </div>
          </div>
          {isEx && <div className="frow">
              <div className="fgrp">
                <label className="flbl">Specialization</label>
                <select className="fsel" value={sp} onChange={e => setSp(e.target.value)}>
                  <option value="">Select...</option>
                  <option>Plant Pathologist</option>
                  <option>Horticulture Expert</option>
                  <option>Soil Scientist</option>
                  <option>Crop Scientist</option>
                </select>
              </div>
              <div className="fgrp">
                <label className="flbl">Consultation Fee (₹)</label>
                <input className="finp" type="number" value={fee} onChange={e => setFee(e.target.value)} placeholder="e.g. 500" />
              </div>
            </div>}
          <div className="fgrp">
            <label className="flbl">Preferred Language</label>
            <select className="fsel" value={lang} onChange={e => setLang(e.target.value)}>
              {['Hindi', 'English', 'Marathi', 'Punjabi', 'Gujarati', 'Tamil', 'Telugu'].map(l => <option key={l}>{l}</option>)}
            </select>
          </div>
          <button className="btn btn-g btn-full" style={{
        marginTop: 8
      }} onClick={saveProfile} disabled={savingProfile}>
            {savingProfile ? <><div className="spin" />Saving...</> : '💾 Profile Save Karo'}
          </button>
        </div>}

      {/* ── PASSWORD TAB ── */}
      {activeTab === 'password' && <div className="card" style={{
      padding: 24
    }}>
          <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 18
      }}>🔑 Password Change Karo</div>
          <div className="fgrp">
            <label className="flbl">Purana Password *</label>
            <input className="finp" type="password" value={oldPwd} onChange={e => setOldPwd(e.target.value)} placeholder="Current password" />
          </div>
          <div className="fgrp">
            <label className="flbl">Naya Password * (8+ characters)</label>
            <input className="finp" type="password" value={newPwd} onChange={e => setNewPwd(e.target.value)} placeholder="New password" />
          </div>
          <div className="fgrp">
            <label className="flbl">Naya Password Confirm *</label>
            <input className="finp" type="password" value={cnfPwd} onChange={e => setCnfPwd(e.target.value)} placeholder="Confirm new password" />
          </div>
          {newPwd && cnfPwd && newPwd !== cnfPwd && <div className="ferr" style={{
        marginBottom: 12
      }}>⚠️ Passwords match nahi kar rahe</div>}
          {newPwd && newPwd.length < 8 && <div className="ferr" style={{
        marginBottom: 12
      }}>⚠️ Password 8+ characters ka hona chahiye</div>}
          <button className="btn btn-g btn-full" onClick={savePassword} disabled={savingPwd || !oldPwd || !newPwd || !cnfPwd}>
            {savingPwd ? <><div className="spin" />Changing...</> : '🔑 Password Change Karo'}
          </button>
        </div>}

      {/* ── NOTIFICATIONS TAB ── */}
      {activeTab === 'notifs' && <div className="card" style={{
      padding: 24
    }}>
          <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 18
      }}>🔔 Notification Preferences</div>
          {[['disease', '🦠 Disease Alerts', 'Aapke area mein outbreak alerts'], ['weather', '🌦️ Weather Warnings', 'Spray timing aur mausam alerts'], ['market', '📈 Market Price Updates', 'Jab bhav mein badlaav aaye'], ['expert', '💬 Expert Replies', 'Consultation aur report updates']].map(([k, l, s]) => <div key={k} className="sett-row" style={{
        padding: '12px 0',
        borderBottom: '1px solid var(--gp)'
      }}>
              <div>
                <div className="sett-lbl" style={{
            fontSize: 14,
            fontWeight: 700
          }}>{l}</div>
                <div style={{
            fontSize: 12,
            color: 'var(--tx3)',
            marginTop: 2
          }}>{s}</div>
              </div>
              <label className="sw">
                <input type="checkbox" checked={notifs[k]} onChange={e => setNotifs(p => ({
            ...p,
            [k]: e.target.checked
          }))} />
                <span className="sw-sl" />
              </label>
            </div>)}
          <button className="btn btn-g btn-full" style={{
        marginTop: 16
      }} onClick={() => toast('Notification preferences save ho gayi ✅')}>
            💾 Save Preferences
          </button>
        </div>}

      {/* ── ACCOUNT TAB ── */}
      {activeTab === 'account' && <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: 14
    }}>
          <div className="card" style={{
        padding: 20
      }}>
            <div style={{
          fontSize: 14,
          fontWeight: 800,
          color: 'var(--g1)',
          marginBottom: 12
        }}>📋 Account Info</div>
            {[['Mobile', user?.mobile || '—'], ['Account Type', isEx ? '👨‍⚕️ Expert' : '🌾 Farmer'], ['Member Since', user?.createdAt ? new Date(user.createdAt).toLocaleDateString('en-IN', {
          day: '2-digit',
          month: 'short',
          year: 'numeric'
        }) : '—'], ['Status', user?.verified ? '✅ Verified' : '⏳ Pending']].map(([k, v]) => <div key={k} style={{
          display: 'flex',
          justifyContent: 'space-between',
          padding: '9px 0',
          borderBottom: '1px solid var(--gp)',
          fontSize: 13
        }}>
                <span style={{
            color: 'var(--tx3)',
            fontWeight: 600
          }}>{k}</span>
                <span style={{
            color: 'var(--tx)',
            fontWeight: 700
          }}>{v}</span>
              </div>)}
          </div>

          <div className="card" style={{
        padding: 20
      }}>
            <div style={{
          fontSize: 14,
          fontWeight: 800,
          color: 'var(--g1)',
          marginBottom: 12
        }}>🗑️ Danger Zone</div>
            <div style={{
          fontSize: 13,
          color: 'var(--tx3)',
          marginBottom: 14
        }}>
              Yeh actions reversible nahi hain. Dhyan se proceed karein.
            </div>
            <button className="btn btn-out btn-md" style={{
          width: '100%',
          marginBottom: 10,
          color: 'var(--r2)',
          borderColor: 'var(--r2)'
        }} onClick={() => toast('Data export 24 ghante mein email pe milega', 'inf')}>
              📤 My Data Export Karo
            </button>
            <button className="btn btn-red btn-md" style={{
          width: '100%'
        }} onClick={doLogout}>
              🚪 Logout
            </button>
          </div>

          <div style={{
        textAlign: 'center',
        fontSize: 12,
        color: 'var(--tx4)',
        padding: '8px 0'
      }}>
            BeejHealth v3.0 • Made with 💚 for Indian Farmers
          </div>
        </div>}
    </div>;
}
