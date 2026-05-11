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

export default function EarningsPage({
  user,
  nav,
  toast
}) {
  const [stats, setStats] = useState({
    total: 0,
    pending: 0,
    completed: 0,
    thisMonth: 0,
    cases: []
  });
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    if (!user || user?.type !== 'expert') return;
    API.get('/api/earnings').then(d => {
      setStats({
        total: d.total || 0,
        pending: d.pending || 0,
        completed: d.completed || 0,
        thisMonth: d.thisMonth || 0,
        totalCases: d.totalCases || 0,
        cases: d.recentCases || [],
        feePerCase: d.feePerCase || 500
      });
      setLoading(false);
    }).catch(() => {
      // Fallback to consultations API
      API.get('/api/consultations').then(d => {
        const cases = d.consultations || [];
        const completed = cases.filter(c => c.status === 'completed');
        const now = new Date();
        const thisMonthCases = completed.filter(c => {
          const dd = new Date(c.createdAt);
          return dd.getMonth() === now.getMonth() && dd.getFullYear() === now.getFullYear();
        });
        const fee = user?.fee || 500;
        setStats({
          total: completed.length * fee,
          pending: cases.filter(c => c.status !== 'completed').length * fee,
          completed: completed.length,
          thisMonth: thisMonthCases.length * fee,
          totalCases: cases.length,
          cases: cases.slice(0, 10),
          feePerCase: fee
        });
        setLoading(false);
      }).catch(() => setLoading(false));
    });
  }, [user]);
  return <div className="wrap-sm">
      <button className="btn btn-ghost btn-sm" style={{
      marginBottom: 18
    }} onClick={() => nav('expert-dashboard')}>← Back</button>
      <div style={{
      fontFamily: "'Baloo 2',cursive",
      fontSize: 26,
      fontWeight: 900,
      color: 'var(--b1)',
      marginBottom: 5
    }}>💰 Earnings & Payouts</div>
      <div style={{
      fontSize: 14,
      color: 'var(--tx2)',
      marginBottom: 24
    }}>Aapki kamaai ka complete breakdown</div>

      {/* Summary Cards */}
      <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(3,1fr)',
      gap: 12,
      marginBottom: 22
    }}>
        {[{
        l: 'Is Mahine',
        v: `₹${(stats.thisMonth || 0).toLocaleString('en-IN')}`,
        s: `${stats.completed || 0} completed cases`,
        c: 'var(--g3)'
      }, {
        l: 'Pending',
        v: `₹${(stats.pending || 0).toLocaleString('en-IN')}`,
        s: `${(stats.totalCases || 0) - (stats.completed || 0)} cases pending`,
        c: 'var(--a2)'
      }, {
        l: 'Total Earned',
        v: `₹${(stats.total || 0).toLocaleString('en-IN')}`,
        s: `₹${stats.feePerCase || 500}/case`,
        c: 'var(--b3)'
      }].map(({
        l,
        v,
        s,
        c
      }) => <div key={l} style={{
        padding: 18,
        borderRadius: 'var(--rad)',
        background: 'white',
        border: '1.5px solid var(--br)',
        textAlign: 'center',
        boxShadow: 'var(--sh)'
      }}>
            <div style={{
          fontSize: 11,
          fontWeight: 700,
          color: 'var(--tx3)',
          textTransform: 'uppercase',
          letterSpacing: '.6px',
          marginBottom: 7
        }}>{l}</div>
            <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 24,
          fontWeight: 900,
          color: c
        }}>{v}</div>
            <div style={{
          fontSize: 11.5,
          color: 'var(--tx3)',
          marginTop: 4
        }}>{s}</div>
          </div>)}
      </div>

      {/* Payment Breakdown */}
      <div className="card" style={{
      padding: 22,
      marginBottom: 18
    }}>
        <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--b1)',
        marginBottom: 14
      }}>📊 Is Mahine Ka Breakdown</div>
        {[['Gross Earnings', `₹${(stats.total || 0).toLocaleString('en-IN')}`, 'var(--tx)'], ['Platform Fee (15%)', `-₹${Math.round((stats.total || 0) * 0.15).toLocaleString('en-IN')}`, 'var(--r2)'], ['TDS (10%)', `-₹${Math.round((stats.total || 0) * 0.10).toLocaleString('en-IN')}`, 'var(--r2)']].map(([l, v, c]) => <div key={l} style={{
        display: 'flex',
        justifyContent: 'space-between',
        padding: '10px 0',
        borderBottom: '1px solid var(--gp)'
      }}>
            <span style={{
          fontSize: 13.5,
          color: 'var(--tx2)',
          fontWeight: 600
        }}>{l}</span>
            <span style={{
          fontSize: 14.5,
          fontWeight: 800,
          color: c
        }}>{v}</span>
          </div>)}
        <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        padding: '14px 0',
        fontSize: 16,
        fontWeight: 900
      }}>
          <span style={{
          color: 'var(--g2)'
        }}>✅ Net Payout</span>
          <span style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 22,
          color: 'var(--g3)'
        }}>₹{Math.round((stats.total || 0) * 0.75).toLocaleString('en-IN')}</span>
        </div>
        <div style={{
        padding: '9px 14px',
        background: 'var(--ap)',
        borderRadius: 8,
        fontSize: 12.5,
        color: 'var(--a1)',
        fontWeight: 600
      }}>
          ⏳ Pending: ₹6,400 (3 cases settled — processing)
        </div>
      </div>

      {/* Consultation Type Breakdown */}
      <div className="card" style={{
      padding: 22,
      marginBottom: 18
    }}>
        <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--b1)',
        marginBottom: 14
      }}>📋 Consultation Breakdown</div>
        {[{
        t: '💬 Text Reports',
        n: 28,
        amt: '₹22,400'
      }, {
        t: '📞 Voice Calls',
        n: 12,
        amt: '₹9,600'
      }, {
        t: '📹 Video Calls',
        n: 8,
        amt: '₹9,600'
      }, {
        t: '🏠 Field Visits',
        n: 0,
        amt: '₹0'
      }].map(({
        t,
        n,
        amt
      }) => <div key={t} style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '9px 0',
        borderBottom: '1px solid var(--gp)'
      }}>
            <div>
              <div style={{
            fontSize: 13.5,
            fontWeight: 700,
            color: 'var(--tx)'
          }}>{t}</div>
              <div style={{
            fontSize: 11.5,
            color: 'var(--tx3)'
          }}>{n} cases</div>
            </div>
            <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 17,
          fontWeight: 900,
          color: n > 0 ? 'var(--b3)' : 'var(--tx4)'
        }}>{amt}</div>
          </div>)}
      </div>

      {/* Withdraw + Invoice */}
      <div style={{
      display: 'flex',
      gap: 11
    }}>
        <button className="btn btn-b btn-lg" style={{
        flex: 2
      }} onClick={() => toast('Bank withdrawal initiated! 2-3 din mein credit hoga', 'inf')}>
          💸 Bank Mein Withdraw Karo
        </button>
        <button className="btn btn-out-b btn-lg" style={{
        flex: 1
      }} onClick={() => toast('GST Invoice download ho raha hai...', 'inf')}>
          🧾 Invoice
        </button>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   SETTINGS PAGE
════════════════════════════════════════════════════════════════ */
