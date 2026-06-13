import React, { useState, useEffect, useRef, useCallback } from "react";
import '../styles/App.css';
import { useToasts } from '../utils/useToasts.jsx';
import { useTranslation } from 'react-i18next';
import { tx } from '../utils/localize.jsx';
import { API, saveSession, clearSession, loadSession } from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody } from '../utils/helpers.jsx';
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary } from '../utils/constants.jsx';

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
import FarmerOnboarding from './FarmerOnboarding.jsx';
import AuthModal from './AuthModal.jsx';

export default /* ════════════════════════════════════════════════════════════════
   ROOT APP
════════════════════════════════════════════════════════════════ */
function BeejHealthApp() {
  const {
    t,
    i18n
  } = useTranslation();
  const {
    user: savedUser
  } = loadSession();
  const [page, setPage] = useState(savedUser ? savedUser.type === 'expert' ? 'expert-dashboard' : 'farmer-dashboard' : 'home');
  const [user, setUser] = useState(savedUser || null);
  const [showAuth, setShowAuth] = useState(false);
  const [authMode, setAuthMode] = useState('login');
  const [onboarding, setOnboarding] = useState(false);
  const [selCrop, setSelCrop] = useState(null);
  const [qAnswers, setQAnswers] = useState({});
  const [showDD, setShowDD] = useState(false);
  const {
    toasts,
    add: toast
  } = useToasts();
  const isEx = user?.type === 'expert';
  const [unreadCount, setUnreadCount] = useState(0);
  useEffect(() => {
    if (!user) return;
    const checkUnread = () => {
      API.get('/api/notifications').then(d => {
        if (d.notifications) setUnreadCount(d.notifications.filter(n => !n.read).length);
      }).catch(() => {});
    };
    checkUnread();
    const iv = setInterval(checkUnread, 30000);
    return () => clearInterval(iv);
  }, [user]);
  const nav = useCallback(pg => {
    const locked = ['farmer-dashboard', 'expert-dashboard', 'my-consultations', 'my-farm', 'chat', 'notifications', 'profile', 'settings', 'booking', 'case-detail', 'ai-report', 'earnings', 'voice', 'satellite', 'forecast', 'soil-sensors', 'insurance', 'marketplace', 'robot-dashboard', 'robot-spray', 'robot-camera', 'robot-map', 'robot-control', 'robot-maintenance', 'robot-analytics'];
    if (locked.includes(pg) && !user) {
      setShowAuth(true);
      setAuthMode('login');
      return;
    }
    setPage(pg);
    window.scrollTo(0, 0);
  }, [user]);
  const handleAuthDone = u => {
    setUser(u);
    setShowAuth(false);
    if (u.fresh) {
      setOnboarding(true);
    } else {
      setPage(u.type === 'expert' ? 'expert-dashboard' : 'farmer-dashboard');
      toast(`Swagat Hai, ${u.name.split(' ')[0]}! 🌱`);
      // Reset consultation state on new login
      setSelCrop(null);
      setQAnswers({});
    }
  };
  const handleObDone = () => {
    setUser(p => ({
      ...p,
      fresh: false
    }));
    setOnboarding(false);
    setPage(isEx ? 'expert-dashboard' : 'farmer-dashboard');
    toast('Setup complete! 🎉');
  };

  // Onboarding screens override
  useEffect(() => {
    document.documentElement.lang = i18n.language || 'en';
  }, [i18n.language]);
  if (onboarding && user) {
    return <>{isEx ? <ExpertOnboarding user={user} onDone={handleObDone} setUser={setUser} /> : <FarmerOnboarding user={user} onDone={handleObDone} setUser={setUser} />}</>;
  }

  // Nav config
  const publicLinks = t('nav.publicLinks', {
    returnObjects: true
  }).map(link => ({
    id: link.id,
    l: link.label
  }));
  const navText = {
    en: { home: 'Home', consult: 'Consult', forecast: 'Forecast', market: 'Market', robots: 'Robots', experts: 'Experts', dashboard: 'Dashboard', cases: 'Cases', earnings: 'Earnings', b2b: 'B2B' },
    hi: { home: 'होम', consult: 'सलाह', forecast: 'पूर्वानुमान', market: 'बाज़ार', robots: 'रोबोट', experts: 'विशेषज्ञ', dashboard: 'डैशबोर्ड', cases: 'केस', earnings: 'कमाई', b2b: 'B2B' },
    mr: { home: 'होम', consult: 'सल्ला', forecast: 'अंदाज', market: 'बाजार', robots: 'रोबोट', experts: 'तज्ञ', dashboard: 'डॅशबोर्ड', cases: 'केसेस', earnings: 'कमाई', b2b: 'B2B' }
  };
  const nt = navText[(i18n.language || 'en').split('-')[0]] || navText.en;
  const cleanNavText = {
    en: { home: 'Home', consult: 'Consult', forecast: 'Forecast', market: 'Market', robots: 'Robots', experts: 'Experts', dashboard: 'Dashboard', cases: 'Cases', earnings: 'Earnings', b2b: 'B2B' },
    hi: { home: 'होम', consult: 'सलाह', forecast: 'पूर्वानुमान', market: 'बाज़ार', robots: 'रोबोट', experts: 'विशेषज्ञ', dashboard: 'डैशबोर्ड', cases: 'केस', earnings: 'कमाई', b2b: 'B2B' },
    mr: { home: 'होम', consult: 'सल्ला', forecast: 'अंदाज', market: 'बाजार', robots: 'रोबोट', experts: 'तज्ज्ञ', dashboard: 'डॅशबोर्ड', cases: 'केसेस', earnings: 'कमाई', b2b: 'B2B' }
  };
  const cleanNt = cleanNavText[(i18n.language || 'en').split('-')[0]] || cleanNavText.en;
  const farmerLinks = [{
    id: 'farmer-dashboard',
    l: '🏠 Home'
  }, {
    id: 'consultation',
    l: '🔬 Consult'
  }, {
    id: 'forecast',
    l: '📊 Forecast'
  }, {
    id: 'marketplace',
    l: '📦 Market'
  }, {
    id: 'robot-dashboard',
    l: '🤖 Robots'
  }, {
    id: 'experts',
    l: '👨‍⚕️ Experts'
  }];
  const expertLinks = [{
    id: 'expert-dashboard',
    l: '🏠 Dashboard'
  }, {
    id: 'case-detail',
    l: '📋 Cases'
  }, {
    id: 'earnings',
    l: '💰 Earnings'
  }, {
    id: 'robot-dashboard',
    l: '🤖 Robots'
  }, {
    id: 'b2b',
    l: '💼 B2B'
  }];
  const localizedNavLabel = id => ({
    'farmer-dashboard': `🏠 ${cleanNt.home}`,
    consultation: `🔬 ${cleanNt.consult}`,
    forecast: `📊 ${cleanNt.forecast}`,
    marketplace: `📦 ${cleanNt.market}`,
    'robot-dashboard': `🤖 ${cleanNt.robots}`,
    experts: `👨‍⚕️ ${cleanNt.experts}`,
    'expert-dashboard': `🏠 ${cleanNt.dashboard}`,
    'case-detail': `📋 ${cleanNt.cases}`,
    earnings: `💰 ${cleanNt.earnings}`,
    b2b: `💼 ${cleanNt.b2b}`
  })[id];
  const links = (user ? isEx ? expertLinks : farmerLinks : publicLinks).map(link => ({
    ...link,
    l: localizedNavLabel(link.id) || link.l
  }));
  const renderPage = () => {
    try {
      switch (page) {
        case 'home':
          return <HomePage nav={nav} setAuth={setShowAuth} setAuthMode={setAuthMode} user={user} />;
        case 'farmer-dashboard':
          return <FarmerDash user={user} nav={nav} toast={toast} />;
        case 'expert-dashboard':
          return <ExpertDash user={user} nav={nav} toast={toast} />;
        case 'consultation':
          return <ConsultPage user={user} nav={nav} toast={toast} selCrop={selCrop} setSelCrop={setSelCrop} qAnswers={qAnswers} setQAnswers={setQAnswers} />;
        case 'ai-report':
          return <AIReportPage selCrop={selCrop || CROPS[0]} nav={nav} toast={toast} qAnswers={qAnswers} viewConsultId={localStorage.getItem('bh_view_consult') || localStorage.getItem('bh_latest_consult')} />;
        case 'my-consultations':
          return <MyConsultPage user={user} nav={nav} toast={toast} />;
        case 'experts':
          return <ExpertsPage user={user} nav={nav} toast={toast} />;
        case 'chat':
          return <ChatPage user={user} nav={nav} toast={toast} />;
        case 'my-farm':
          return <MyFarmPage user={user} nav={nav} toast={toast} />;
        case 'notifications':
          return <NotifPage nav={nav} user={user} />;
        case 'earnings':
          return <EarningsPage user={user} nav={nav} toast={toast} />;
        case 'voice':
          return <VoiceInputPage user={user} nav={nav} toast={toast} />;
        case 'satellite':
          return <SatellitePage user={user} nav={nav} toast={toast} />;
        case 'forecast':
          return <ForecastPage user={user} nav={nav} toast={toast} />;
        case 'soil-sensors':
          return <SoilSensorPage user={user} nav={nav} toast={toast} />;
        case 'b2b':
          return <B2BPage nav={nav} toast={toast} />;
        case 'marketplace':
          return <MarketplacePage user={user} nav={nav} toast={toast} />;
        case 'insurance':
          return <InsurancePage user={user} nav={nav} toast={toast} />;
        case 'govt-map':
          return <GovtMapPage nav={nav} toast={toast} />;
        case 'robot-dashboard':
          return <RobotDashboard user={user} nav={nav} toast={toast} />;
        case 'robot-spray':
          return <RobotSprayPage nav={nav} toast={toast} />;
        case 'robot-camera':
          return <RobotCameraPage nav={nav} toast={toast} />;
        case 'robot-map':
          return <RobotMapPage nav={nav} toast={toast} />;
        case 'robot-control':
          return <RobotControlPage nav={nav} toast={toast} />;
        case 'robot-maintenance':
          return <RobotMaintenancePage nav={nav} toast={toast} />;
        case 'robot-analytics':
          return <RobotAnalyticsPage nav={nav} toast={toast} />;
        case 'support':
          return <SupportPage toast={toast} />;
        case 'profile':
          return <ProfilePage user={user} nav={nav} toast={toast} setUser={setUser} />;
        case 'settings':
          return <SettingsPage user={user} setUser={setUser} nav={nav} toast={toast} />;
        case 'case-detail':
          return <CaseDetailPage user={user} nav={nav} toast={toast} />;
        case 'booking':
          return <BookingPage user={user} nav={nav} toast={toast} />;
        default:
          return <HomePage nav={nav} setAuth={setShowAuth} setAuthMode={setAuthMode} user={user} />;
      }
    } catch (err) {
      console.error('Page render error:', err);
      return <div style={{
        textAlign: 'center',
        padding: '60px 20px',
        fontFamily: 'sans-serif'
      }}>
          <div style={{
          fontSize: 48,
          marginBottom: 16
        }}>🌿</div>
          <div style={{
          fontSize: 18,
          fontWeight: 700,
          color: '#166534',
          marginBottom: 8
        }}>Page load mein error</div>
          <div style={{
          fontSize: 13,
          color: '#6b7280',
          marginBottom: 20
        }}>{err?.message}</div>
          <button onClick={() => setPage('home')} style={{
          padding: '10px 24px',
          background: '#16a34a',
          color: 'var(--tx)',
          border: 'none',
          borderRadius: 8,
          fontSize: 14,
          fontWeight: 600,
          cursor: 'pointer'
        }}>
            🏠 Home Par Jao
          </button>
        </div>;
    }
  };
  return <>
      
      <div className="shell" onClick={() => setShowDD(false)}>

        {/* NAVBAR */}
        <nav className="nav">
          <div className="nav-logo" onClick={() => setPage(user ? isEx ? 'expert-dashboard' : 'farmer-dashboard' : 'home')}>
            <div className="nav-logo-mark">🌱</div>
            <span className="nav-logo-txt">BeejHealth</span>
          </div>
          <div className="nav-links">
            {links.map(l => <button key={l.id} className={`nav-a${page === l.id ? ' on' : ''}`} onClick={() => nav(l.id)}>{l.l}</button>)}
          </div>
          <div className="nav-right">
            <select className="nav-lang" aria-label={t('common.language')} value={i18n.language || 'en'} onChange={e => i18n.changeLanguage(e.target.value)} onClick={e => e.stopPropagation()}>
              <option value="en">{t('common.languages.en')}</option>
              <option value="hi">{t('common.languages.hi')}</option>
              <option value="mr">{t('common.languages.mr')}</option>
            </select>
            {user ? <>
                <div className="nav-bell" onClick={() => {
              nav('notifications');
              setUnreadCount(0);
            }}>
                  🔔{unreadCount > 0 && <div className="nav-bell-dot" />}
                </div>
                <div style={{
              position: 'relative'
            }}>
                  <div className={`nav-av${isEx ? ' ex-av' : ''}`} onClick={e => {
                e.stopPropagation();
                setShowDD(v => !v);
              }}>
                    {user?.initials}
                  </div>
                  {showDD && <div className="dd-menu" onClick={e => e.stopPropagation()}>
                      <div className="dd-head">
                        <div className="dd-name">{user?.name || ''}</div>
                        <div className="dd-sub">{isEx ? `👨‍⚕️ ${tx(i18n, 'expert')}` : `🌾 ${tx(i18n, 'farmer')}`} • ✅ {tx(i18n, 'verified')}</div>
                      </div>
                      {[
                        ['👤', tx(i18n, 'profile'), 'profile'],
                        ['📋', isEx ? tx(i18n, 'allCases') : tx(i18n, 'consultations'), isEx ? 'case-detail' : 'my-consultations'],
                        [isEx ? '💰' : '🗺️', isEx ? cleanNt.earnings : tx(i18n, 'myFarm'), isEx ? 'earnings' : 'my-farm'],
                        ['🔔', tx(i18n, 'notifications'), 'notifications'],
                        ['⚙️', tx(i18n, 'settings'), 'settings']
                      ].map(([ic, l, p]) => <div key={l} className="dd-row" onClick={() => {
                  nav(p);
                  setShowDD(false);
                }}><span>{ic}</span>{l}</div>)}
                      <div className="dd-div" />
                      <div className="dd-row red-row" onClick={() => {
                  clearSession();
                  localStorage.removeItem('bh_latest_consult');
                  localStorage.removeItem('bh_chat_consult');
                  localStorage.removeItem('bh_latest_crop');
                  localStorage.removeItem('bh_view_consult');
                  localStorage.removeItem('bh_sel_expert');
                  setUser(null);
                  setPage('home');
                  setShowDD(false);
                  toast('Aap successfully logout ho gaye 👋', 'inf');
                }}>🚪 {tx(i18n, 'logout')}</div>
                    </div>}
                </div>
              </> : <>
                <button className="nav-btn-login" onClick={() => {
              setAuthMode('login');
              setShowAuth(true);
            }}>{t('nav.auth.login')}</button>
                <button className="nav-btn-reg" onClick={() => {
              setAuthMode('register');
              setShowAuth(true);
            }}>{t('nav.auth.register')}</button>
              </>}
          </div>
        </nav>

        {/* MAIN CONTENT */}
        <main className={page === 'chat' ? 'pg-chat' : 'pg'}>
          {renderPage()}
        </main>
      </div>

      {/* AUTH MODAL */}
      {showAuth && <AuthModal mode={authMode} setMode={setAuthMode} onClose={() => setShowAuth(false)} onDone={handleAuthDone} initType={isEx ? 'expert' : 'farmer'} />}

      {/* TOASTS */}
      <div className="toast-wrap">
        {toasts.map(t => <div key={t.id} className={`toast${t.type === 'err' ? ' err' : t.type === 'inf' ? ' inf' : t.type === 'warn' ? ' warn' : ''}`}>
            <span style={{
          fontSize: 17
        }}>{t.type === 'err' ? '❌' : t.type === 'inf' ? 'ℹ️' : t.type === 'warn' ? '⚠️' : '✅'}</span>
            {t.msg}
          </div>)}
      </div>
    </>;
}
