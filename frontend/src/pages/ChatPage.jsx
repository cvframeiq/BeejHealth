import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import { API, saveSession, clearSession, loadSession } from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody, fileToUploadDataUrl, uploadPhotoAsset } from '../utils/helpers.jsx';
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
   CHAT PAGE
════════════════════════════════════════════════════════════════ */
function ChatPage({
  user,
  nav,
  toast
}) {
  const [msgs, setMsgs] = useState([]);
  const [txt, setTxt] = useState('');
  const [consultation, setConsultation] = useState(null);
  const [consultId, setConsultId] = useState(null);
  const [peerTyping, setPeerTyping] = useState(false);
  const [sendingImage, setSendingImage] = useState(false);
  const endRef = useRef(null);
  const chatImageInputRef = useRef(null);
  const typingTimerRef = useRef(null);
  const typingActiveRef = useRef(false);
  const isEx = user?.type === 'expert';
  useEffect(() => endRef.current?.scrollIntoView({
    behavior: 'smooth'
  }), [msgs]);
  const stopTyping = async () => {
    clearTimeout(typingTimerRef.current);
    typingTimerRef.current = null;
    if (!consultId || !typingActiveRef.current) return;
    typingActiveRef.current = false;
    try {
      await API.post('/api/consultations/' + consultId + '/typing', {
        isTyping: false
      });
    } catch {}
  };
  const setTypingState = async isTyping => {
    if (!consultId) return;
    if (typingActiveRef.current === isTyping) return;
    typingActiveRef.current = isTyping;
    try {
      await API.post('/api/consultations/' + consultId + '/typing', {
        isTyping
      });
    } catch {}
  };
  const loadThread = async id => {
    if (!id) return;
    try {
      const m = await API.get('/api/consultations/' + id + '/messages');
      if (m.messages) {
        setMsgs(m.messages.map(msg => formatChatMessage(msg, 'en-IN')));
      }
      setPeerTyping(Boolean(m.typing?.length));
    } catch {}
  };
  useEffect(() => {
    if (!user) return;
    let alive = true;
    const storedId = getConsultationContextId();
    if (storedId) {
      setConsultId(storedId);
      loadThread(storedId);
    }
    API.get('/api/consultations').then(d => {
      if (!alive) return;
      const list = d.consultations || [];
      if (!list.length) return;
      const matched = storedId ? list.find(c => c._id === storedId) : null;
      const selected = matched || list[0];
      setConsultation(selected);
      setConsultId(selected._id);
      rememberConsultationContext(selected._id);
      if (!storedId || selected._id !== storedId) {
        loadThread(selected._id);
      }
    }).catch(() => {});
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  // Auto-refresh messages every 5 seconds when consultId available
  useEffect(() => {
    if (!consultId) return;
    const poll = setInterval(() => {
      API.get('/api/consultations/' + consultId + '/messages').then(m => {
        if (m.messages && m.messages.length > 0) {
          const next = m.messages.map(msg => formatChatMessage(msg, 'en-IN'));
          setMsgs(prev => {
            if (prev.length === next.length && prev[prev.length - 1]?.id === next[next.length - 1]?.id) return prev;
            return next;
          });
        }
        setPeerTyping(Boolean(m.typing?.length));
      }).catch(() => {});
    }, 5000);
    return () => clearInterval(poll);
  }, [consultId]);
  useEffect(() => () => {
    void stopTyping();
  }, [consultId]);
  const handleTxtChange = e => {
    const value = e.target.value;
    setTxt(value);
    if (!consultId) return;
    void setTypingState(true);
    clearTimeout(typingTimerRef.current);
    typingTimerRef.current = setTimeout(() => {
      void setTypingState(false);
    }, 1200);
  };
  const send = async () => {
    if (!txt.trim() || !consultId) return;
    const msgTxt = txt.trim();
    setTxt('');
    await stopTyping();
    const nm = {
      id: Date.now(),
      from: isEx ? 'expert' : 'farmer',
      text: msgTxt,
      senderName: user?.name || 'You',
      time: new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
      })
    };
    setMsgs(p => [...p, nm]);
    try {
      await API.post('/api/consultations/' + consultId + '/messages', {
        text: msgTxt
      });
    } catch (e) {
      console.warn('Message send warn:', e.message);
    }
  };
  const sendImage = async file => {
    if (!file || !consultId) return;
    setSendingImage(true);
    await stopTyping();
    try {
      const uploadAsset = await fileToUploadDataUrl(file);
      const upload = await uploadPhotoAsset({
        base64: uploadAsset.dataUrl,
        type: uploadAsset.contentType || file.type || 'image/jpeg',
        consultationId: consultId,
        index: msgs.length,
        totalExpected: 1
      });
      const res = await API.post('/api/consultations/' + consultId + '/messages', {
        messageType: 'image',
        photoId: upload.photoId,
        imageUrl: upload.url,
        text: ''
      });
      if (res?.message) {
        setMsgs(prev => [...prev, formatChatMessage(res.message, 'en-IN')]);
      } else {
        await loadThread(consultId);
      }
    } catch (e) {
      console.warn('Image send warn:', e.message);
      toast?.(`Image upload fail: ${e.message}`, 'err');
    } finally {
      setSendingImage(false);
      if (chatImageInputRef.current) chatImageInputRef.current.value = '';
    }
  };
  const openImagePicker = () => chatImageInputRef.current?.click();
  const shortId = consultation?._id ? String(consultation._id).slice(-6).toUpperCase() : consultId ? String(consultId).slice(-6).toUpperCase() : '------';
  const chatTitle = consultation?.expertName || (isEx ? 'Assigned Consultation' : 'Consultation Chat');
  const chatSub = consultation ? `${consultation.cropName || 'Crop'} • ${consultation.disease || 'Messages'} • Chat Free` : 'Loading consultation...';
  return <div className="chat-wrap">
      <div className="chat-hd">
        <button className="btn btn-ghost btn-sm" onClick={() => nav(isEx ? 'expert-dashboard' : 'farmer-dashboard')}>←</button>
        <div style={{
        width: 40,
        height: 40,
        borderRadius: '50%',
        background: 'linear-gradient(135deg,var(--g5),var(--g6))',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 19
      }}>{consultation?.cropEmoji || '👨‍🔬'}</div>
        <div>
          <div style={{
          fontSize: 15,
          fontWeight: 800,
          color: 'var(--tx)'
        }}>{chatTitle}</div>
          <div style={{
          fontSize: 12,
          color: 'var(--g4)',
          fontWeight: 600
        }}>{chatSub}</div>
        </div>
        <div style={{
        marginLeft: 'auto',
        display: 'flex',
        gap: 8
      }}>
          <button className="btn btn-ghost btn-sm" onClick={() => {
          localStorage.setItem('bh_booking_mode', 'audio');
          nav('booking');
        }}>📞</button>
          <button className="btn btn-ghost btn-sm" onClick={() => {
          localStorage.setItem('bh_booking_mode', 'video');
          nav('booking');
        }}>📹</button>
          <button className="btn btn-ghost btn-sm" onClick={() => nav('ai-report')}>📄 Report</button>
        </div>
      </div>
      <div style={{
      padding: '8px 22px',
      background: 'var(--gp)',
      fontSize: 12.5,
      color: 'var(--g2)',
      textAlign: 'center',
      fontWeight: 700,
      borderBottom: '1px solid var(--br)',
      display: 'flex',
      justifyContent: 'center',
      flexWrap: 'wrap',
      gap: 10
    }}>
        <span>{consultation?.cropEmoji || '💬'} Case #{shortId}</span>
        <span>Chat (Free)</span>
        <span>Audio Call (Paid)</span>
        <span>Video Call (Paid)</span>
      </div>
      {peerTyping && <div style={{
      padding: '8px 22px',
      background: 'var(--ap)',
      fontSize: 12.5,
      color: 'var(--a1)',
      textAlign: 'left',
      fontWeight: 700,
      borderBottom: '1px solid var(--br)'
    }}>
          ⌨️ {isEx ? 'Farmer' : consultation?.expertName || 'Expert'} is typing...
        </div>}
      <div className="chat-msgs">
        {consultation && <div style={{
        textAlign: 'center',
        fontSize: 11.5,
        color: 'var(--tx3)',
        margin: '6px 0'
      }}>{new Date(consultation.createdAt).toLocaleString('en-IN', {
          dateStyle: 'medium',
          timeStyle: 'short'
        })}</div>}
        {msgs.map(m => <div key={m.id} className={`chat-msg${(isEx ? m.from === 'expert' : m.from === 'farmer') ? ' mine' : ' theirs'}`}>
            <div className="msg-bbl"><ChatMessageBody msg={m} /></div>
            <div className="msg-time">{m.time}{(isEx ? m.from === 'expert' : m.from === 'farmer') && ' ✓✓'}</div>
          </div>)}
        <div ref={endRef} />
      </div>
      <div className="chat-input-bar">
        <input ref={chatImageInputRef} type="file" accept="image/*" capture="environment" style={{
        display: 'none'
      }} onChange={async e => {
        const file = e.target.files?.[0];
        e.target.value = '';
        if (file) await sendImage(file);
      }} />
        <button style={{
        fontSize: 20,
        background: 'none',
        border: 'none',
        padding: '0 3px',
        cursor: 'pointer'
      }} onClick={openImagePicker} disabled={sendingImage} title="Image bhejo">📷</button>
        <input className="chat-inp" placeholder="Message type karo..." value={txt} onChange={handleTxtChange} onBlur={() => void stopTyping()} onKeyDown={e => e.key === 'Enter' && send()} />
        <button className="chat-send" onClick={send} disabled={!txt.trim() || sendingImage}>➤</button>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   CASE DETAIL (Expert)
════════════════════════════════════════════════════════════════ */
