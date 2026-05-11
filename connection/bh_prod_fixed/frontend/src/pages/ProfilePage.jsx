import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import { API, saveSession, clearSession, loadSession } from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody } from '../utils/helpers.jsx';
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary } from '../utils/constants.jsx';
import { localizeCropLabel, tx } from '../utils/localize.jsx';

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
   PROFILE PAGE
════════════════════════════════════════════════════════════════ */
function ProfilePage({
  user,
  nav,
  toast,
  setUser
}) {
  const { i18n } = useTranslation();
  const isEx = user?.type === 'expert';
  const [editing, setEditing] = useState(false);
  const [nm, setNm] = useState(user?.name || '');
  const [em, setEm] = useState(user?.email || '');
  const [dt, setDt] = useState(user?.district || '');
  const [vl, setVl] = useState(user?.village || '');
  const [bio, setBio] = useState(user?.bio || '');
  const [saving, setSaving] = useState(false);
  const saveProfile = async () => {
    if (!nm.trim()) {
      toast('Naam required hai', 'err');
      return;
    }
    setSaving(true);
    try {
      const res = await API.patch('/api/auth/profile', {
        name: nm.trim(),
        email: em.trim(),
        district: dt,
        village: vl,
        bio
      });
      if (res.user) {
        saveSession(localStorage.getItem('bh_token'), res.user);
        if (setUser) setUser(res.user);
      }
      setEditing(false);
      toast('Profile update ho gayi! ✅');
    } catch (e) {
      toast(e.message, 'err');
    }
    setSaving(false);
  };
  const cancelEdit = () => {
    setNm(user?.name || '');
    setEm(user?.email || '');
    setDt(user?.district || '');
    setVl(user?.village || '');
    setBio(user?.bio || '');
    setEditing(false);
  };
  return <div className="wrap-md">
      {/* Header */}
      <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 24
    }}>
        <div style={{
        fontFamily: "'Baloo 2',cursive",
        fontSize: 26,
        fontWeight: 900,
        color: 'var(--g1)'
      }}>👤 {tx(i18n, 'profile')}</div>
        {!editing && <button className="btn btn-g btn-md" onClick={() => setEditing(true)}>✏️ {tx(i18n, 'editProfile')}</button>}
      </div>

      {/* Avatar Card */}
      <div className="card" style={{
      padding: 24,
      marginBottom: 20
    }}>
        <div style={{
        display: 'flex',
        gap: 20,
        alignItems: 'center',
        marginBottom: 20
      }}>
          <div style={{
          width: 72,
          height: 72,
          borderRadius: '50%',
          background: isEx ? 'linear-gradient(135deg,var(--b3),var(--b4))' : 'linear-gradient(135deg,var(--g4),var(--g5))',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white',
          fontSize: 26,
          fontWeight: 900,
          flexShrink: 0,
          boxShadow: '0 4px 16px rgba(0,0,0,.15)'
        }}>
            {(nm || user?.name || 'U').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()}
          </div>
          <div>
            <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 22,
            fontWeight: 900,
            color: 'var(--tx)'
          }}>{user?.name}</div>
            <div style={{
            fontSize: 13,
            color: isEx ? 'var(--b3)' : 'var(--g4)',
            fontWeight: 700,
            marginTop: 3
          }}>{isEx ? `👨‍⚕️ ${tx(i18n, 'expert')}` : `🌾 ${tx(i18n, 'farmer')}`}</div>
            <div style={{
            display: 'flex',
            gap: 8,
            marginTop: 8,
            flexWrap: 'wrap'
          }}>
              <span className="badge bg-g">✅ {tx(i18n, 'verified')}</span>
              {user?.district && <span className="badge" style={{
              background: 'var(--gp)',
              color: 'var(--g2)'
            }}>📍 {user?.district}</span>}
              {isEx && user?.spec && <span className="badge bg-b">{user?.spec}</span>}
            </div>
          </div>
        </div>

        {/* View Mode */}
        {!editing && <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 12
      }}>
            {[['📱 Mobile', user?.mobile || '—'], ['📧 Email', user?.email || 'Not set'], ['📍 District', user?.district || 'Not set'], ['🏘️ Village', user?.village || 'Not set'], ...(isEx ? [['🎓 Specialization', user?.spec || '—'], ['💰 Fee', '₹' + (user?.fee || 0) + ' / consultation']] : [['🌾 Crops', (user?.crops || []).join(', ') || 'Not set'], ['🗣️ Languages', user?.langs || 'Hindi']])].map(([k, v]) => <div key={k} style={{
          background: 'var(--gb)',
          borderRadius: 10,
          padding: '12px 14px'
        }}>
                <div style={{
            fontSize: 11,
            color: 'var(--tx3)',
            fontWeight: 600,
            marginBottom: 3
          }}>{k}</div>
                <div style={{
            fontSize: 13,
            color: 'var(--tx)',
            fontWeight: 700
          }}>{v}</div>
              </div>)}
          </div>}

        {/* Edit Mode */}
        {editing && <div>
            <div style={{
          fontSize: 14,
          fontWeight: 800,
          color: 'var(--g1)',
          marginBottom: 16
        }}>✏️ {tx(i18n, 'profileEdit')}</div>
            <div className="frow">
              <div className="fgrp">
                <label className="flbl">{tx(i18n, 'name')} *</label>
                <input className="finp" value={nm} onChange={e => setNm(e.target.value)} placeholder="Poora naam" />
              </div>
              <div className="fgrp">
                <label className="flbl">{tx(i18n, 'email')}</label>
                <input className="finp" type="email" value={em} onChange={e => setEm(e.target.value)} placeholder="email@example.com" />
              </div>
            </div>
            <div className="frow">
              <div className="fgrp">
                <label className="flbl">{tx(i18n, 'district')}</label>
                <input className="finp" value={dt} onChange={e => setDt(e.target.value)} placeholder="Aapka district" />
              </div>
              <div className="fgrp">
                <label className="flbl">{tx(i18n, 'village')}</label>
                <input className="finp" value={vl} onChange={e => setVl(e.target.value)} placeholder="Gaon ka naam" />
              </div>
            </div>
            <div className="fgrp">
              <label className="flbl">{tx(i18n, 'bio')}</label>
              <textarea className="ftxt" rows={3} value={bio} onChange={e => setBio(e.target.value)} placeholder="Apne baare mein kuch likhein..." />
            </div>
            <div style={{
          display: 'flex',
          gap: 10,
          marginTop: 4
        }}>
              <button className="btn btn-out btn-md" style={{
            flex: 1
          }} onClick={cancelEdit} disabled={saving}>✕ {tx(i18n, 'cancel')}</button>
              <button className="btn btn-g btn-md" style={{
            flex: 2
          }} onClick={saveProfile} disabled={saving}>
                {saving ? <><div className="spin" />{tx(i18n, 'saving')}</> : `💾 ${tx(i18n, 'saveChanges')}`}
              </button>
            </div>
          </div>}
      </div>

      {/* Stats */}
      <div className="card" style={{
      padding: 20,
      marginBottom: 20
    }}>
        <div style={{
        fontSize: 14,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 14
      }}>📊 {tx(i18n, 'activityStats')}</div>
        <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3,1fr)',
        gap: 12
      }}>
          {(isEx ? [['Total Cases', user?.totalCases || 0, 'var(--b3)'], ['Rating', user?.rating ? user?.rating.toFixed(1) + '⭐' : 'N/A', 'var(--a2)'], ['Status', user?.available ? '🟢 Online' : '🔴 Offline', user?.available ? 'var(--g4)' : 'var(--r2)']] : [[tx(i18n, 'consultations'), '—', 'var(--g4)'], [tx(i18n, 'crops'), user?.crops?.length || 0, 'var(--a2)'], [tx(i18n, 'memberSince'), user?.createdAt ? new Date(user.createdAt).getFullYear() : '—', 'var(--b3)']]).map(([l, v, c]) => <div key={l} style={{
          background: 'var(--gb)',
          borderRadius: 10,
          padding: '14px 12px',
          textAlign: 'center'
        }}>
              <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 22,
            fontWeight: 900,
            color: c
          }}>{v}</div>
              <div style={{
            fontSize: 11,
            color: 'var(--tx3)',
            marginTop: 3,
            fontWeight: 600
          }}>{l}</div>
            </div>)}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card" style={{
      padding: 20
    }}>
        <div style={{
        fontSize: 14,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 14
      }}>⚡ {tx(i18n, 'quickActions')}</div>
        <div style={{
        display: 'flex',
        gap: 10,
        flexWrap: 'wrap'
      }}>
          <button className="btn btn-ghost btn-md" onClick={() => nav('settings')}>⚙️ {tx(i18n, 'settings')}</button>
          <button className="btn btn-ghost btn-md" onClick={() => nav('notifications')}>🔔 {tx(i18n, 'notifications')}</button>
          {isEx && <button className="btn btn-ghost btn-md" onClick={() => nav('earnings')}>💰 Earnings</button>}
          {!isEx && <button className="btn btn-ghost btn-md" onClick={() => nav('my-farm')}>🌾 {tx(i18n, 'myFarm')}</button>}
          <button className="btn btn-red btn-md" onClick={() => {
          clearSession();
          ['bh_latest_consult', 'bh_chat_consult', 'bh_latest_crop', 'bh_view_consult', 'bh_sel_expert'].forEach(k => localStorage.removeItem(k));
          if (setUser) setUser(null);
          nav('home');
          toast('Logout successful 👋', 'inf');
        }}>🚪 {tx(i18n, 'logout')}</button>
        </div>
      </div>
    </div>;
}
