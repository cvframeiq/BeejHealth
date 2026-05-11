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
import FarmerOnboarding from './FarmerOnboarding.jsx';
import AuthModal from './AuthModal.jsx';

export default /* ════════════════════════════════════════════════════════════════
   EXPERT ONBOARDING
════════════════════════════════════════════════════════════════ */
function ExpertOnboarding({
  user,
  onDone,
  setUser
}) {
  const finishExpertOnboarding = async data => {
    try {
      const res = await API.patch('/api/auth/profile', {
        spec: data.spec || 'Agricultural Expert',
        fee: Number(data.fee) || 500,
        university: data.university || '',
        langs: data.langs || 'Hindi',
        available: true
      });
      if (res.user) {
        saveSession(localStorage.getItem('bh_token'), res.user);
        if (setUser) setUser(res.user);
      }
    } catch (e) {
      console.warn('Expert onboarding save:', e.message);
    }
    onDone();
  };
  const [uploaded, setUploaded] = useState({
    id: false,
    degree: false,
    exp: false
  });
  return <div className="ob-wrap">
      <div className="ob-box">
        <div className="ob-head ex">
          <div style={{
          fontSize: 12,
          opacity: .76,
          marginBottom: 5
        }}>👨‍⚕️ Expert Verification</div>
          <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 22,
          fontWeight: 900
        }}>Namaskar, {user?.name || 'Expert'}! 👋</div>
          <div style={{
          fontSize: 13,
          opacity: .8,
          marginTop: 3
        }}>Verification complete karo — 2–3 din mein approval</div>
        </div>
        <div className="ob-body">
          <div style={{
          background: 'white',
          border: '1.5px solid var(--bpb)',
          borderRadius: 'var(--rad)',
          padding: 20
        }}>
            <div style={{
            fontSize: 15,
            fontWeight: 800,
            color: 'var(--b1)',
            marginBottom: 14
          }}>🔐 Verification Checklist</div>
            {[{
            k: 'mobile',
            l: 'Mobile Verified',
            s: 'OTP se complete',
            done: true
          }, {
            k: 'id',
            l: 'Government ID',
            s: 'Aadhaar / PAN upload',
            done: uploaded.id
          }, {
            k: 'degree',
            l: 'Degree Certificate',
            s: 'University certificate',
            done: uploaded.degree
          }, {
            k: 'exp',
            l: 'Experience Letter',
            s: 'Employer / registration',
            done: uploaded.exp
          }].map(item => <div key={item.k} className="ver-item">
                <div className="ver-ic" style={{
              background: item.done ? 'var(--gp)' : 'var(--bp)'
            }}>{item.done ? '✅' : '📄'}</div>
                <div style={{
              flex: 1
            }}>
                  <div style={{
                fontSize: 14,
                fontWeight: 700,
                color: 'var(--tx)'
              }}>{item.l}</div>
                  <div style={{
                fontSize: 12,
                color: 'var(--tx3)'
              }}>{item.s}</div>
                </div>
                {!item.done && <button className="btn btn-b btn-sm" onClick={() => setUploaded(p => ({
              ...p,
              [item.k]: true
            }))}>📤 Upload</button>}
              </div>)}
          </div>
          <div style={{
          padding: 13,
          background: 'var(--bp)',
          borderRadius: 10,
          margin: '16px 0',
          fontSize: 13,
          color: 'var(--b1)',
          lineHeight: 1.65
        }}>
            <strong>ℹ️ Process:</strong> Documents review 2–3 business days. Tab tak platform explore kar sakte hain.
          </div>
          <div style={{
          display: 'flex',
          gap: 9
        }}>
            <button className="btn btn-out-b btn-md" style={{
            flex: 1
          }} onClick={() => finishExpertOnboarding({})}>Baad Mein</button>
            <button className="btn btn-b btn-md" style={{
            flex: 2
          }} onClick={() => finishExpertOnboarding({})}>📋 Dashboard Dekho →</button>
          </div>
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   HOME PAGE
════════════════════════════════════════════════════════════════ */
