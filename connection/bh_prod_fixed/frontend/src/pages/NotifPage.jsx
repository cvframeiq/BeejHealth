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
   NOTIFICATIONS PAGE
════════════════════════════════════════════════════════════════ */
function NotifPage({
  nav,
  user
}) {
  const [notifs, setNotifs] = useState(NOTIFICATIONS);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!user) return;
    setLoading(true);
    API.get('/api/notifications').then(d => {
      if (d.notifications && d.notifications.length > 0) {
        setNotifs(d.notifications.map(n => ({
          id: n._id,
          type: n.type || 'info',
          icon: n.icon || '🔔',
          col: n.type === 'welcome' ? '#eaf7ef' : n.type === 'consultation' ? '#e3f2fd' : n.type === 'message' ? '#fff8e1' : n.type === 'report_ready' ? '#eaf7ef' : '#f3e5f5',
          title: n.title,
          desc: n.body,
          time: new Date(n.createdAt).toLocaleDateString('en-IN', {
            day: '2-digit',
            month: 'short'
          }),
          unread: !n.read
        })));
      }
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [user]);
  return <div className="wrap-sm">
      <div style={{
      fontFamily: "'Baloo 2',cursive",
      fontSize: 26,
      fontWeight: 900,
      color: 'var(--g1)',
      marginBottom: 5
    }}>🔔 Notifications</div>
      <div style={{
      fontSize: 14,
      color: 'var(--tx2)',
      marginBottom: 22
    }}>Disease alerts, weather warnings, expert replies</div>
      {loading && <div style={{
      textAlign: 'center',
      padding: 20,
      fontSize: 13,
      color: 'var(--tx3)'
    }}>⏳ Loading...</div>}
      <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 16
    }}>
        <div style={{
        fontSize: 13,
        color: 'var(--tx3)'
      }}>{notifs.filter(n => n.unread).length} unread</div>
        <button className="btn btn-ghost btn-sm" onClick={async () => {
        setNotifs(p => p.map(n => ({
          ...n,
          unread: false
        })));
        try {
          await API.patch('/api/notifications/read-all');
        } catch (e) {
          console.warn(e);
        }
      }}>✓ Sab Read Karo</button>
      </div>
      {notifs.map(n => <div key={n.id} className={`notif-item${n.unread ? ' unread' : ''}`}>
          <div className="notif-icon" style={{
        background: n.col
      }}>{n.icon}</div>
          <div style={{
        flex: 1
      }}>
            <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
              <div className="notif-t">{n.title}</div>
              {n.unread && <div style={{
            width: 7,
            height: 7,
            background: 'var(--g4)',
            borderRadius: '50%',
            flexShrink: 0
          }} />}
            </div>
            <div className="notif-d">{n.desc}</div>
            <div className="notif-time">{n.time}</div>
          </div>
        </div>)}
      {notifs.length === 0 && <div style={{
      textAlign: 'center',
      padding: '40px 20px',
      color: 'var(--tx4)',
      fontSize: 14
    }}>
        🔔 Koi notification nahi hai
      </div>}
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   SUPPORT PAGE
════════════════════════════════════════════════════════════════ */
