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
   BOOKING PAGE
════════════════════════════════════════════════════════════════ */
function BookingPage({
  user,
  nav,
  toast
}) {
  const [type, setType] = useState(() => {
    const saved = localStorage.getItem('bh_booking_mode');
    return ['chat', 'audio', 'video'].includes(saved) ? saved : 'chat';
  });
  const [selExpert] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('bh_sel_expert') || 'null');
    } catch {
      return null;
    }
  });
  const [booking, setBooking] = useState(false);
  useEffect(() => {
    localStorage.removeItem('bh_booking_mode');
  }, []);
  const confirmBooking = async () => {
    if (!user) {
      if (toast) toast('Pehle login karein!', 'err');
      return;
    }
    setBooking(true);
    try {
      const consultId = localStorage.getItem('bh_latest_consult');
      if (consultId && selExpert) {
        await API.patch('/api/consultations/' + consultId + '/status', {
          status: 'expert_assigned',
          report: null,
          expertId: selExpert.id,
          expertName: selExpert.name
        });
      }
      if (toast) toast(type === 'chat' ? 'Free chat started ✅' : 'Payment successful! Expert jald hi contact karega. ✅');
      setTimeout(() => nav('my-consultations'), 1500);
    } catch (e) {
      if (toast) toast(e.message, 'err');
    }
    setBooking(false);
  };
  const types = [{
    id: 'chat',
    l: '💬 Chat (Free)',
    p: 0,
    t: 'Instant replies',
    d: 'Text chat without any payment'
  }, {
    id: 'audio',
    l: '📞 Audio Call (Paid)',
    p: 600,
    t: '15–30 min',
    d: 'Direct call expert se'
  }, {
    id: 'video',
    l: '📹 Video Call (Paid)',
    p: 1200,
    t: '30–45 min',
    d: 'Live field guidance'
  }];
  const sel = types.find(t => t.id === type);
  const total = sel.p ? Math.round(sel.p * 1.18 * 1.05) : 0;
  return <div className="wrap-sm">
      <button className="btn btn-ghost btn-sm" style={{
      marginBottom: 18
    }} onClick={() => nav('experts')}>← Wapas</button>
      <div style={{
      fontFamily: "'Baloo 2',cursive",
      fontSize: 26,
      fontWeight: 900,
      color: 'var(--g1)',
      marginBottom: 5
    }}>📅 Consultation Book Karo</div>
      <div style={{
      fontSize: 14,
      color: 'var(--tx2)',
      marginBottom: 24
    }}>
        {selExpert?.name ? `${selExpert.name} ke saath service select karein` : 'Chat free hai; audio/video calls paid hain'}
      </div>
      <div className="book-types">
        {types.map(t => <div key={t.id} className={`book-type${type === t.id ? ' sel' : ''}`} onClick={() => setType(t.id)}>
            <div style={{
          fontSize: 17,
          fontWeight: 800,
          color: 'var(--g1)',
          marginBottom: 3
        }}>{t.l}</div>
            <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 21,
          fontWeight: 900,
          color: t.p === 0 ? 'var(--g3)' : 'var(--g4)',
          marginBottom: 3
        }}>{t.p === 0 ? 'Free' : `₹${t.p}`}</div>
            <div style={{
          fontSize: 12,
          color: 'var(--tx3)',
          marginBottom: 3
        }}>⏱️ {t.t}</div>
            <div style={{
          fontSize: 12.5,
          color: 'var(--tx2)'
        }}>{t.d}</div>
          </div>)}
      </div>
      <div className="card" style={{
      padding: 22,
      marginBottom: 18
    }}>
        <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 14
      }}>💰 Payment Summary</div>
        {sel.p === 0 ? <div style={{
        padding: '12px 14px',
        background: 'var(--gp)',
        borderRadius: 10,
        fontSize: 13,
        color: 'var(--g2)',
        fontWeight: 700
      }}>
            💬 Chat free hai. Koi payment required nahi hai.
          </div> : <>
            {[['Consultation Fee', `₹${sel.p}`], ['Platform Fee (5%)', `₹${Math.round(sel.p * .05)}`], ['GST (18%)', `₹${Math.round(sel.p * .18)}`]].map(([k, v]) => <div key={k} style={{
          display: 'flex',
          justifyContent: 'space-between',
          padding: '9px 0',
          borderBottom: '1px solid var(--gp)',
          fontSize: 13.5,
          color: 'var(--tx2)'
        }}>
                <span>{k}</span><span style={{
            fontWeight: 700,
            color: 'var(--tx)'
          }}>{v}</span>
              </div>)}
            <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          padding: '13px 0',
          fontSize: 16,
          fontWeight: 900,
          color: 'var(--g1)'
        }}>
              <span>TOTAL</span><span style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 22
          }}>₹{total}</span>
            </div>
            <div style={{
          fontSize: 12,
          color: 'var(--tx3)'
        }}>🔒 Razorpay secured • 24hr refund policy</div>
          </>}
      </div>
      {sel.p > 0 && <div className="card" style={{
      padding: 20,
      marginBottom: 18
    }}>
          <div style={{
        fontSize: 14,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 13
      }}>💳 Payment Method</div>
          {['UPI (Google Pay, PhonePe, Paytm)', 'Debit / Credit Card', 'Net Banking', 'Kisan Credit Card'].map((m, i) => <div key={m} style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '9px 0',
        borderBottom: '1px solid var(--gp)',
        cursor: 'pointer'
      }}>
              <div style={{
          width: 17,
          height: 17,
          borderRadius: '50%',
          border: '2px solid var(--g4)',
          background: i === 0 ? 'var(--g4)' : 'none'
        }} />
              <span style={{
          fontSize: 13.5,
          fontWeight: 600,
          color: 'var(--tx)'
        }}>{m}</span>
            </div>)}
        </div>}
      <button className="btn btn-g" style={{
      width: '100%',
      padding: '14px',
      fontSize: 16,
      borderRadius: 12
    }} onClick={() => {
      const consultContextId = getConsultationContextId();
      if (consultContextId) rememberConsultationContext(consultContextId);
      if (type === 'chat') {
        toast('Free chat started ✅');
        nav('chat');
        return;
      }
      toast(`Payment ₹${total} successful! Consultation booked ✅`);
      nav('chat');
    }}>
        {type === 'chat' ? '💬 Start Free Chat →' : `🔐 Secure Pay ₹${total} →`}
      </button>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   CHAT PAGE
════════════════════════════════════════════════════════════════ */
