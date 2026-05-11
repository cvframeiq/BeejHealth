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
   SUPPORT PAGE
════════════════════════════════════════════════════════════════ */
function SupportPage({
  toast
}) {
  const [ticket, setTicket] = useState({
    name: '',
    mobile: '',
    issue: '',
    desc: ''
  });
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const submitTicket = async () => {
    if (!ticket.issue.trim() || !ticket.desc.trim()) {
      toast('Topic aur message zaroor bharo', 'err');
      return;
    }
    setSubmitting(true);
    try {
      const res = await API.post('/api/support', {
        name: ticket.name,
        mobile: ticket.mobile,
        issue: ticket.issue,
        desc: ticket.desc
      });
      setSubmitted(true);
      toast(`Support ticket #${res.ticketId?.slice(-6).toUpperCase() || 'XXXXXX'} submit ho gaya! 24 ghante mein reply milegi. ✅`);
    } catch (e) {
      toast(e.message || 'Submit fail hua', 'err');
    }
    setSubmitting(false);
  };
  const [openFaq, setOpenFaq] = useState(null);
  const faqs = [{
    q: 'Crop photos kaise upload karein?',
    a: 'Crop Consultation page par jao, "Photo Upload" select karo, aur photo lo ya gallery se choose karo.'
  }, {
    q: 'AI report kab milegi?',
    a: 'Photo upload ke baad 30–60 seconds mein AI analysis complete hoti hai.'
  }, {
    q: 'Expert se kaise connect karein?',
    a: 'Experts page par specialist dhundho, "Select Expert" click karo aur consultation book karo.'
  }, {
    q: 'Payment refund policy kya hai?',
    a: 'Consultation ke 24 ghante ke andar refund request ki ja sakti hai. support@beejhealth.com contact karein.'
  }, {
    q: 'App offline kaam karta hai kya?',
    a: 'Basic features aur last AI report offline accessible hain. Full functionality ke liye internet chahiye.'
  }];
  return <div className="wrap-md">
      <div style={{
      fontFamily: "'Baloo 2',cursive",
      fontSize: 26,
      fontWeight: 900,
      color: 'var(--g1)',
      marginBottom: 5
    }}>🆘 Support Center</div>
      <div style={{
      fontSize: 14,
      color: 'var(--tx2)',
      marginBottom: 28
    }}>Kisi bhi sawaal ya issue ke liye — hum yahan hain</div>
      <div style={{
      display: 'grid',
      gridTemplateColumns: '1fr 300px',
      gap: 22
    }}>
        <div className="card" style={{
        padding: 26
      }}>
          <div style={{
          fontSize: 16,
          fontWeight: 800,
          color: 'var(--g1)',
          marginBottom: 20
        }}>📝 Message Bhejein</div>
          <div className="frow">
            <div className="fgrp"><label className="flbl">Naam</label><input className="finp" placeholder="Aapka naam" value={ticket.name} onChange={e => setTicket(p => ({
              ...p,
              name: e.target.value
            }))} /></div>
            <div className="fgrp"><label className="flbl">Mobile</label><input className="finp" placeholder="Mobile number" value={ticket.mobile} onChange={e => setTicket(p => ({
              ...p,
              mobile: e.target.value
            }))} /></div>
          </div>
          <div className="fgrp">
            <label className="flbl">Topic</label>
            <select className="fsel" value={ticket.issue} onChange={e => setTicket(p => ({
            ...p,
            issue: e.target.value
          }))}>
              <option value="">Select topic</option>
              <option>Technical Issue</option><option>Payment Problem</option><option>Expert Related</option><option>AI Report Query</option><option>Other</option>
            </select>
          </div>
          <div className="fgrp">
            <label className="flbl">Message</label>
            <textarea className="ftxt" rows={4} placeholder="Apna issue ya sawaal describe karein..." value={ticket.desc} onChange={e => setTicket(p => ({
            ...p,
            desc: e.target.value
          }))} />
          </div>
          <button className="btn btn-g btn-full" onClick={submitTicket} disabled={submitting}>
            {submitting ? <><div className="spin" />Bhej raha hoon...</> : '📨 Submit Request'}
          </button>
          {submitted && <div style={{
          marginTop: 14,
          padding: '13px 16px',
          background: 'var(--gp)',
          borderRadius: 10,
          fontSize: 13,
          color: 'var(--g2)',
          fontWeight: 700,
          textAlign: 'center'
        }}>✅ Ticket #{Math.floor(Math.random() * 90000 + 10000)} submit ho gaya! 24 ghante mein reply milegi.</div>}
        </div>
        <div>
          <div className="card" style={{
          padding: 20,
          marginBottom: 18
        }}>
            <div style={{
            fontSize: 14,
            fontWeight: 800,
            color: 'var(--g1)',
            marginBottom: 14
          }}>📞 Contact Info</div>
            {[['📧', 'Email', 'support@beejhealth.com'], ['📞', 'Call', '+91 123 456 7890'], ['💬', 'WhatsApp', '+91 98765 43210'], ['⏰', 'Hours', 'Mon–Sat: 9AM–6PM']].map(([i, l, v]) => <div key={l} style={{
            display: 'flex',
            gap: 11,
            padding: '11px 0',
            borderBottom: '1px solid var(--gp)'
          }}>
                <div style={{
              width: 34,
              height: 34,
              background: 'var(--gp)',
              borderRadius: 9,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              fontSize: 15
            }}>{i}</div>
                <div><div style={{
                fontSize: 11,
                fontWeight: 700,
                color: 'var(--tx3)',
                textTransform: 'uppercase',
                letterSpacing: .5
              }}>{l}</div><div style={{
                fontSize: 13.5,
                fontWeight: 600,
                color: 'var(--tx)'
              }}>{v}</div></div>
              </div>)}
          </div>
          <div className="card" style={{
          padding: 20
        }}>
            <div style={{
            fontSize: 14,
            fontWeight: 800,
            color: 'var(--g1)',
            marginBottom: 13
          }}>❓ FAQs</div>
            {faqs.map((f, i) => <div key={i} style={{
            borderBottom: '1px solid var(--gp)',
            cursor: 'pointer'
          }} onClick={() => setOpenFaq(openFaq === i ? null : i)}>
                <div style={{
              padding: '11px 0',
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 13,
              fontWeight: 700,
              color: 'var(--tx)'
            }}>
                  <span>{f.q}</span><span style={{
                fontSize: 11,
                marginLeft: 8
              }}>{openFaq === i ? '▲' : '▼'}</span>
                </div>
                {openFaq === i && <div style={{
              fontSize: 13,
              color: 'var(--tx2)',
              paddingBottom: 11,
              lineHeight: 1.65
            }}>{f.a}</div>}
              </div>)}
          </div>
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   PROFILE PAGE
════════════════════════════════════════════════════════════════ */
