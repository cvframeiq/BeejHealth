import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import { API, saveSession, clearSession, loadSession } from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody } from '../utils/helpers.jsx';
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary } from '../utils/constants.jsx';
import { cropName, formatDateTime, formatNumber, formatPercent, reportValue, tx } from '../utils/localize.jsx';

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
import ExpertsPage from './ExpertsPage.jsx';
import FarmerDash from './FarmerDash.jsx';
import ExpertDash from './ExpertDash.jsx';
import AIReportPage from './AIReportPage.jsx';
import ConsultPage from './ConsultPage.jsx';
import HomePage from './HomePage.jsx';
import ExpertOnboarding from '../components/ExpertOnboarding.jsx';
import FarmerOnboarding from '../components/FarmerOnboarding.jsx';
import AuthModal from '../components/AuthModal.jsx';

export default /* ════════════════════════════════════════════════════════════════
   MY CONSULTATIONS
════════════════════════════════════════════════════════════════ */
function MyConsultPage({
  user,
  nav,
  toast
}) {
  const { i18n } = useTranslation();
  const [filter, setFilter] = useState('all');
  const [apiConsults, setApiConsults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [ratingConsult, setRatingConsult] = useState(null);
  const [myRating, setMyRating] = useState(0);
  const [ratingSubmitted, setRatingSubmitted] = useState(false);

  // Status helpers
  const stColor = s => s === 'completed' ? 'bg-g' : s === 'expert' ? 'bg-b' : s === 'pending' ? 'bg-a' : 'bg-p';
  const stIcon = s => s === 'completed' ? '✅' : s === 'expert' ? '👨‍⚕️' : s === 'pending' ? '⏳' : '🔵';
  const N = value => formatNumber(value, i18n);
  const P = value => formatPercent(value, i18n);
  const RV = value => reportValue(value, i18n);
  const statusText = status => status === 'completed' ? tx(i18n, 'completed') : status === 'expert' ? tx(i18n, 'expertAssigned') : status === 'pending' ? tx(i18n, 'pending') : tx(i18n, 'aiReady');
  const submitRating = async () => {
    if (!myRating || !ratingConsult?.expertId) return;
    try {
      await API.post('/api/experts/' + ratingConsult.expertId + '/rate', {
        rating: myRating,
        consultationId: ratingConsult._id
      });
      setRatingSubmitted(true);
      toast('Rating submit ho gayi! ⭐ Shukriya.');
      setTimeout(() => {
        setRatingConsult(null);
        setMyRating(0);
        setRatingSubmitted(false);
      }, 2000);
    } catch (e) {
      toast('Rating fail hua', 'err');
    }
  };
  useEffect(() => {
    if (!user) return;
    setLoading(true);
    API.get('/api/consultations').then(d => {
      if (d.consultations) setApiConsults(d.consultations.map(c => ({
        _id: c._id,
        id: c._id,
        crop: cropName(c.cropId || c.cropName, i18n) || c.cropName,
        emoji: c.cropEmoji || '🌱',
        issue: `${RV(c.disease || 'Analysis')} ${tx(i18n, 'diseaseDetected')} (${P(c.confidence || 0)} ${tx(i18n, 'confidenceWord')})`,
        date: formatDateTime(c.createdAt, i18n, {
          day: '2-digit',
          month: 'short',
          year: 'numeric'
        }),
        expert: c.expertName || 'Auto-assign',
        expertId: c.expertId || null,
        expertName: c.expertName || 'Expert',
        status: c.status === 'completed' ? 'completed' : c.status === 'expert_assigned' ? 'expert' : c.status === 'pending' ? 'pending' : 'ai',
        statusLabel: c.status === 'completed' ? tx(i18n, 'completed') : c.status === 'expert_assigned' ? tx(i18n, 'expertAssigned') : c.status === 'pending' ? tx(i18n, 'pending') : tx(i18n, 'aiReady'),
        sev: c.severity || 1,
        conf: c.confidence || 0,
        rated: c.rated || false
      })));
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [user]);
  if (!user) return <div className="wrap" style={{
    textAlign: 'center',
    padding: '80px 28px'
  }}>
      <div style={{
      fontSize: 60,
      marginBottom: 16
    }}>🔒</div>
      <div style={{
      fontSize: 22,
      fontWeight: 800,
      color: 'var(--g1)',
      marginBottom: 8
    }}>{tx(i18n, 'login')}</div>
      <div style={{
      fontSize: 15,
      color: 'var(--tx2)',
      marginBottom: 24
    }}>{tx(i18n, 'loginRequired')}</div>
      <button className="btn btn-g btn-lg" onClick={() => nav('home')}>{tx(i18n, 'login')}</button>
    </div>;
  const displayConsults = apiConsults.length > 0 ? apiConsults : [];
  const filtered = filter === 'all' ? displayConsults : displayConsults.filter(c => c.status === filter);
  return <div className="wrap">
      {/* Rating Modal */}
      {ratingConsult && !ratingSubmitted && <div className="overlay" onClick={() => setRatingConsult(null)}>
          <div className="modal" style={{
        maxWidth: 380,
        padding: 28
      }} onClick={e => e.stopPropagation()}>
            <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 20,
          fontWeight: 900,
          color: 'var(--g1)',
          marginBottom: 8
        }}>⭐ Expert Rate Karein</div>
            <div style={{
          fontSize: 13,
          color: 'var(--tx3)',
          marginBottom: 18
        }}>Dr. {ratingConsult?.expertName || 'Expert'} ke liye aapka feedback</div>
            <div style={{
          display: 'flex',
          gap: 12,
          justifyContent: 'center',
          marginBottom: 20
        }}>
              {[1, 2, 3, 4, 5].map(s => <button key={s} onClick={() => setMyRating(s)} style={{
            fontSize: 28,
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            opacity: s <= myRating ? 1 : 0.3,
            transform: s <= myRating ? 'scale(1.2)' : 'scale(1)',
            transition: 'all .15s'
          }}>⭐</button>)}
            </div>
            <button className="btn btn-g btn-full" onClick={submitRating} disabled={!myRating}>
              {myRating ? `${myRating} Star Dein` : 'Star select karein'}
            </button>
          </div>
        </div>}
      {ratingConsult && ratingSubmitted && <div className="overlay">
          <div className="modal" style={{
        maxWidth: 300,
        padding: 28,
        textAlign: 'center'
      }}>
            <div style={{
          fontSize: 50,
          marginBottom: 8
        }}>🎉</div>
            <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 18,
          fontWeight: 900,
          color: 'var(--g1)'
        }}>Shukriya!</div>
            <div style={{
          fontSize: 13,
          color: 'var(--tx3)',
          marginTop: 4
        }}>Aapka feedback expert ke liye valuable hai.</div>
          </div>
        </div>}

      {loading && <div style={{
      textAlign: 'center',
      padding: 40,
      fontSize: 14,
      color: 'var(--tx3)'
      }}>⏳ {tx(i18n, 'loadingConsultations')}</div>}
      <div style={{
      marginBottom: 26
    }}>
        <div style={{
        fontFamily: "'Baloo 2',cursive",
        fontSize: 28,
        fontWeight: 900,
        color: 'var(--g1)'
      }}>📋 {tx(i18n, 'consultations')}</div>
        <div style={{
        fontSize: 15,
        color: 'var(--tx2)',
        marginTop: 5
      }}>{tx(i18n, 'diagnosisHistory')}</div>
      </div>

      {/* Filter tabs */}
      <div style={{
      display: 'flex',
      gap: 8,
      marginBottom: 22,
      flexWrap: 'wrap'
    }}>
        {['all', 'ai', 'expert', 'pending', 'completed'].map(f => <button key={f} onClick={() => setFilter(f)} style={{
        padding: '7px 17px',
        borderRadius: 100,
        fontSize: 13,
        fontWeight: 700,
        border: `2px solid ${filter === f ? 'var(--g4)' : 'var(--br)'}`,
        background: filter === f ? 'var(--gp)' : 'white',
        color: filter === f ? 'var(--g3)' : 'var(--tx2)',
        cursor: 'pointer',
        transition: 'all .18s'
      }}>
            {f === 'all' ? tx(i18n, 'all') : f === 'ai' ? `🔵 ${tx(i18n, 'aiReady')}` : f === 'expert' ? `🟢 ${tx(i18n, 'expertAssigned')}` : f === 'pending' ? `🟠 ${tx(i18n, 'pending')}` : `⚫ ${tx(i18n, 'completed')}`}
          </button>)}
      </div>

      {/* Empty state */}
      {!loading && filtered.length === 0 && <div style={{
      textAlign: 'center',
      padding: '60px 20px',
      color: 'var(--tx4)'
    }}>
          <div style={{
        fontSize: 48,
        marginBottom: 12
      }}>🌿</div>
          <div style={{
        fontSize: 16,
        fontWeight: 700,
        color: 'var(--tx2)',
        marginBottom: 8
      }}>{tx(i18n, 'noConsultations')}</div>
          <div style={{
        fontSize: 13,
        color: 'var(--tx3)',
        marginBottom: 20
      }}>{tx(i18n, 'startFirstConsultation')}</div>
          <button className="btn btn-g btn-md" onClick={() => nav('consultation')}>🔬 {tx(i18n, 'startConsultation')}</button>
        </div>}

      {/* Consultation grid */}
      <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill,minmax(310px,1fr))',
      gap: 18
    }}>
        {filtered.map(c => <div key={c.id} className="card card-hov" style={{
        overflow: 'hidden',
        cursor: 'pointer'
      }} onClick={() => {
        rememberConsultationContext(c.id);
        nav('ai-report');
      }}>
            <div style={{
          height: 75,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 40,
          background: 'linear-gradient(135deg,var(--gp),var(--gpb))'
        }}>{c.emoji}</div>
            <div className="cons-body">
              <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            marginBottom: 5
          }}>
                <div className="cons-nm">{c.crop}</div>
                <span className={`badge ${stColor(c.status)}`}>{stIcon(c.status)} {c.statusLabel || statusText(c.status)}</span>
              </div>
              <div className="cons-issue">{c.issue}</div>
              <div className="cons-meta">📅 {c.date}</div>
              <div className="cons-meta">👨‍⚕️ {c.expert}</div>
              <div style={{
            display: 'flex',
            gap: 6,
            margin: '7px 0'
          }}>
                <span className="badge bg-b">AI: {P(c.conf)}</span>
                <span className={`badge ${c.sev >= 3 ? 'bg-r' : c.sev === 2 ? 'bg-a' : 'bg-g'}`}>{tx(i18n, 'severityStage')}: {N(c.sev)}/{N(5)}</span>
              </div>
              <div className="cons-acts">
                <button className="ca-rep" onClick={e => {
              e.stopPropagation();
              rememberConsultationContext(c.id);
              nav('ai-report');
            }}>📄 {tx(i18n, 'reportAction')}</button>
                <button className="ca-chat" onClick={e => {
              e.stopPropagation();
              rememberConsultationContext(c.id);
              nav('chat');
            }}>💬 {tx(i18n, 'chatAction')} →</button>
                {c.status === 'completed' && c.expertId && <button className="ca-rep" style={{
              background: 'var(--ap)',
              color: 'var(--a1)'
            }} onClick={e => {
              e.stopPropagation();
              setRatingConsult(c);
            }}>⭐ {tx(i18n, 'rate')}</button>}
              </div>
            </div>
          </div>)}
      </div>
    </div>;
}
