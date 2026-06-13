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
import MyConsultPage from './MyConsultPage.jsx';
import AIReportPage from './AIReportPage.jsx';
import ConsultPage from './ConsultPage.jsx';
import HomePage from './HomePage.jsx';
import ExpertOnboarding from '../components/ExpertOnboarding.jsx';
import FarmerOnboarding from '../components/FarmerOnboarding.jsx';
import AuthModal from '../components/AuthModal.jsx';

export default function VoiceInputPage({
  user,
  nav,
  toast
}) {
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const recognitionRef = useRef(null);
  // Fix: use a ref to hold the latest transcript so onend closure isn't stale
  const transcriptRef = useRef('');
  const startListening = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      toast('Aapka browser voice input support nahi karta. Chrome use karein.', 'err');
      return;
    }
    transcriptRef.current = '';
    const rec = new SR();
    rec.lang = 'hi-IN';
    rec.continuous = false;
    rec.interimResults = true;
    rec.onstart = () => setListening(true);
    rec.onresult = e => {
      const t = Array.from(e.results).map(r => r[0].transcript).join('');
      transcriptRef.current = t;
      setTranscript(t);
    };
    rec.onerror = e => {
      toast('Voice error: ' + e.error, 'err');
      setListening(false);
    };
    rec.onend = () => {
      setListening(false);
      const latest = transcriptRef.current;
      if (latest.trim()) processVoice(latest);
    };
    recognitionRef.current = rec;
    rec.start();
  };
  const stopListening = () => {
    if (recognitionRef.current) recognitionRef.current.stop();
    setListening(false);
  };
  const processVoice = async text => {
    setProcessing(true);
    // Detect crop and disease keywords from Hindi transcript
    const crops = {
      'tamatar': 'tomato',
      'gehu': 'wheat',
      'aalu': 'potato',
      'kapas': 'cotton',
      'naariyal': 'coconut',
      'makka': 'corn',
      'aam': 'mango'
    };
    const diseases = {
      'daag': 'spots',
      'peela': 'yellow',
      'murjhana': 'wilt',
      'sukh': 'dry',
      'rot': 'spots'
    };
    let detectedCrop = 'tomato',
      detectedIssue = 'spots';
    Object.entries(crops).forEach(([hi, en]) => {
      if (text.toLowerCase().includes(hi)) detectedCrop = en;
    });
    Object.entries(diseases).forEach(([hi, en]) => {
      if (text.toLowerCase().includes(hi)) detectedIssue = en;
    });
    await new Promise(r => setTimeout(r, 1200));
    setResult({
      crop: detectedCrop,
      issue: detectedIssue,
      text,
      confidence: 82
    });
    setProcessing(false);
    toast('Voice input process ho gaya! ✅');
  };
  const SAMPLE_TRANSCRIPTS = ['Mere tamatar ke patte peele ho rahe hain aur neeche gir rahe hain', 'Gehun ki fasal mein lal rang ke daag aa rahe hain patto pe', 'Alu ke paudhon mein kaale rang ke dhabb dikhe hain'];
  const analyze = async () => {
    if (!transcript) return;
    setProcessing(true);
    await new Promise(r => setTimeout(r, 1800));
    setProcessing(false);
    setResult({
      crop: 'Tomato 🍅',
      disease: 'Early Blight',
      conf: 87,
      action: 'Mancozeb 75% WP spray karein — 2.5g/L'
    });
  };
  return <div className="wrap-sm">
      <button className="btn btn-ghost btn-sm" style={{
      marginBottom: 18
    }} onClick={() => nav('consultation')}>← Wapas</button>
      <div style={{
      fontFamily: "'Baloo 2',cursive",
      fontSize: 26,
      fontWeight: 900,
      color: 'var(--g1)',
      marginBottom: 5
    }}>🎤 Voice Se Bolo</div>
      <div style={{
      fontSize: 14,
      color: 'var(--tx2)',
      marginBottom: 28
    }}>Hindi mein apni fasal ki problem batao — AI samjhega</div>

      {/* Mic Button */}
      <div style={{
      textAlign: 'center',
      marginBottom: 32
    }}>
        <div style={{
        marginBottom: 18,
        fontSize: 13,
        color: 'var(--tx3)',
        fontWeight: 600
      }}>
          {listening ? '🔴 Sun raha hoon...' : transcript ? '✅ Suna gaya' : 'Mic dabao aur bolna shuru karo'}
        </div>
        <button className={`voice-btn${listening ? ' listening' : ' idle'}`} onClick={startListening} disabled={processing}>
          {listening ? '🎙️' : '🎤'}
        </button>
        {listening && <div style={{
        display: 'flex',
        justifyContent: 'center',
        marginTop: 18
      }}>
            <div className="voice-wave">
              {[28, 14, 36, 20, 32, 16, 30].map((h, i) => <div key={i} className="vw-bar" style={{
            height: h,
            animationDelay: `${i * 0.12}s`
          }} />)}
            </div>
          </div>}
      </div>

      {/* Transcript Box */}
      {(transcript || listening) && <div className="card" style={{
      padding: 20,
      marginBottom: 18
    }} key="trans">
          <div style={{
        fontSize: 11,
        fontWeight: 700,
        color: 'var(--tx3)',
        textTransform: 'uppercase',
        letterSpacing: '.6px',
        marginBottom: 9
      }}>📝 Aapne Kaha:</div>
          <div style={{
        fontSize: 15,
        lineHeight: 1.7,
        fontStyle: listening ? 'italic' : 'normal',
        color: listening ? 'var(--tx3)' : 'var(--tx)'
      }}>
            {listening ? '...' : `"${transcript}"`}
          </div>
          {transcript && !listening && <div style={{
        display: 'flex',
        gap: 9,
        marginTop: 14
      }}>
              <button className="btn btn-ghost btn-sm" style={{
          flex: 1
        }} onClick={() => {
          setTranscript('');
          setResult(null);
        }}>🔄 Dobara Bolo</button>
              <button className="btn btn-g btn-sm" style={{
          flex: 2
        }} onClick={analyze} disabled={processing}>
                {processing ? <><div className="spin" />Analyze ho raha hai...</> : '🤖 AI Se Analyze Karao →'}
              </button>
            </div>}
        </div>}

      {/* AI Result */}
      {result && <div className="card" style={{
      padding: 22,
      border: '2px solid var(--g4)'
    }} key="result">
          <div style={{
        fontSize: 13,
        fontWeight: 700,
        color: 'var(--g3)',
        marginBottom: 12
      }}>✅ AI Analysis Complete</div>
          <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 10,
        marginBottom: 14
      }}>
            {[['Crop', result.crop], ['Disease', result.disease], ['Confidence', `${result.conf}%`], ['Severity', 'Stage 2/5']].map(([k, v]) => <div key={k} style={{
          padding: 12,
          background: 'var(--gp)',
          borderRadius: 10
        }}>
                <div style={{
            fontSize: 11,
            color: 'var(--tx3)',
            fontWeight: 700,
            marginBottom: 3
          }}>{k}</div>
                <div style={{
            fontSize: 14,
            fontWeight: 800,
            color: 'var(--g1)'
          }}>{v}</div>
              </div>)}
          </div>
          <div style={{
        padding: 13,
        background: 'var(--ap)',
        borderRadius: 10,
        fontSize: 13.5,
        color: 'var(--a1)',
        fontWeight: 600,
        marginBottom: 14
      }}>
            💊 {result.action}
          </div>
          <div style={{
        display: 'flex',
        gap: 9
      }}>
            <button className="btn btn-out btn-sm" style={{
          flex: 1
        }} onClick={() => nav('ai-report')}>📄 Full Report</button>
            <button className="btn btn-g btn-sm" style={{
          flex: 2
        }} onClick={() => nav('experts')}>👨‍⚕️ Expert Se Confirm →</button>
          </div>
        </div>}

      {/* Language Support */}
      <div style={{
      marginTop: 24,
      padding: 16,
      background: 'var(--gp)',
      borderRadius: 'var(--rad)',
      fontSize: 13,
      color: 'var(--g2)',
      fontWeight: 600
    }}>
        🗣️ Supported Languages: Hindi • Marathi • Punjabi • Gujarati • Telugu • Tamil
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   2. SATELLITE FIELD MONITOR 🛰️
════════════════════════════════════════════════════════════════ */
