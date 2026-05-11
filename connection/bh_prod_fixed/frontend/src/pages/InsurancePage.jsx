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
import MyConsultPage from './MyConsultPage.jsx';
import AIReportPage from './AIReportPage.jsx';
import ConsultPage from './ConsultPage.jsx';
import HomePage from './HomePage.jsx';
import ExpertOnboarding from '../components/ExpertOnboarding.jsx';
import FarmerOnboarding from '../components/FarmerOnboarding.jsx';
import AuthModal from '../components/AuthModal.jsx';

export default /* ════════════════════════════════════════════════════════════════
   7. INSURANCE CLAIM 🏦
════════════════════════════════════════════════════════════════ */
function InsurancePage({
  user,
  nav,
  toast
}) {
  const [applying, setApplying] = useState(false);
  const [applied, setApplied] = useState(false);
  const [form, setForm] = useState({
    scheme: 'PMFBY',
    crop: '',
    area: '',
    bank: '',
    accountNo: '',
    aadhaar: ''
  });
  const applyInsurance = async () => {
    if (!form.crop || !form.area) {
      toast('Crop aur area fill karo', 'err');
      return;
    }
    setApplying(true);
    try {
      await API.post('/api/consultations', {
        cropId: 'insurance',
        cropName: 'Insurance: ' + form.crop,
        cropEmoji: '🏦',
        method: 'manual',
        disease: 'PMFBY Application',
        confidence: 100,
        severity: 1,
        answers: {
          ...form,
          userName: user?.name,
          district: user?.district
        }
      });
      setApplied(true);
      toast('Insurance application submit ho gayi! Krishi vibhag se confirmation aayegi. ✅');
    } catch (e) {
      toast('Submit fail hua', 'err');
    }
    setApplying(false);
  };
  const SCHEMES = [{
    id: 'pmfby',
    name: 'PMFBY — Pradhan Mantri Fasal Bima Yojana',
    premium: '2% (Kharif)',
    cover: 'Full crop loss',
    apply: 'pm-kisan.gov.in'
  }, {
    id: 'wbcis',
    name: 'WBCIS — Weather Based Crop Insurance',
    premium: '2% (Kharif)',
    cover: 'Rainfall based',
    apply: 'aicofindia.com'
  }, {
    id: 'unified',
    name: 'Unified Package Insurance Scheme',
    premium: 'Nominal',
    cover: 'Life + Assets + Crop',
    apply: 'agricoop.nic.in'
  }, {
    id: 'coconut',
    name: 'Coconut Palm Insurance Scheme',
    premium: '₹14.83/palm/year',
    cover: 'Per palm damage',
    apply: 'cpcri.res.in'
  }];
  const [step, setStep] = useState(1);
  const [claimData, setClaimData] = useState({
    crop: '',
    date: '',
    damage: '',
    area: ''
  });
  const steps = [{
    n: 1,
    l: 'AI Verification',
    s: 'Disease confirm karein',
    done: step > 1,
    active: step === 1
  }, {
    n: 2,
    l: 'Photo Evidence',
    s: '3-5 photos upload',
    done: step > 2,
    active: step === 2
  }, {
    n: 3,
    l: 'Field Assessment',
    s: 'GPS location + area',
    done: step > 3,
    active: step === 3
  }, {
    n: 4,
    l: 'Insurance Co.',
    s: 'Claim submit karein',
    done: step > 4,
    active: step === 4
  }, {
    n: 5,
    l: 'Payout',
    s: 'Bank transfer',
    done: step > 5,
    active: step === 5
  }];
  return <div className="wrap-sm">
      <div style={{
      fontFamily: "'Baloo 2',cursive",
      fontSize: 26,
      fontWeight: 900,
      color: 'var(--g1)',
      marginBottom: 5
    }}>🏦 Insurance Claim</div>
      <div style={{
      fontSize: 13,
      color: 'var(--tx2)',
      marginBottom: 20
    }}>PMFBY — Pradhan Mantri Fasal Bima Yojana • AI-verified claims</div>

      {/* Claim Status Header */}
      <div className="claim-status" style={{
      background: step >= 4 ? 'linear-gradient(135deg,var(--gp),var(--gpb))' : 'linear-gradient(135deg,var(--ap),#fff8e1)',
      border: `2px solid ${step >= 4 ? 'var(--g4)' : 'var(--a2)'}`
    }}>
        <div style={{
        fontSize: 36,
        marginBottom: 8
      }}>{step >= 5 ? '🎉' : step >= 4 ? '⏳' : '📋'}</div>
        <div style={{
        fontFamily: "'Baloo 2',cursive",
        fontSize: 20,
        fontWeight: 900,
        color: step >= 4 ? 'var(--g2)' : 'var(--a1)'
      }}>
          {step >= 5 ? 'Claim Approved! ✅' : step >= 4 ? 'Review Under Process' : 'Claim Filing in Progress'}
        </div>
        <div style={{
        fontSize: 13,
        color: 'var(--tx2)',
        marginTop: 5
      }}>
          {step >= 5 ? '₹42,000 aapke account mein 3-5 din mein aayega' : step >= 4 ? 'Insurance company review kar rahi hai — 5-7 working days' : `Step ${step}/5 complete`}
        </div>
        {step >= 4 && <div style={{
        marginTop: 12,
        fontFamily: "'Baloo 2',cursive",
        fontSize: 28,
        fontWeight: 900,
        color: 'var(--g3)'
      }}>₹42,000</div>}
      </div>

      {/* Progress Steps */}
      <div style={{
      marginBottom: 22
    }}>
        {steps.map(s => <div key={s.n} className={`ins-step${s.active ? ' active' : s.done ? ' done' : ''}`}>
            <div className={`ins-step-num ${s.done ? 'done' : s.active ? 'active' : 'wait'}`}>
              {s.done ? '✓' : s.n}
            </div>
            <div style={{
          flex: 1
        }}>
              <div style={{
            fontSize: 14,
            fontWeight: 700,
            color: s.active ? 'var(--g2)' : s.done ? 'var(--g4)' : 'var(--tx)'
          }}>{s.l}</div>
              <div style={{
            fontSize: 12,
            color: 'var(--tx3)',
            marginTop: 2
          }}>{s.s}</div>
            </div>
            {s.active && <span className="badge bg-g" style={{
          fontSize: 11
        }}>Current</span>}
            {s.done && <span className="badge bg-g" style={{
          fontSize: 11
        }}>Done ✅</span>}
          </div>)}
      </div>

      {/* Active Step Content */}
      {step === 1 && <div className="card" style={{
      padding: 22,
      marginBottom: 18
    }}>
          <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 14
      }}>🤖 AI Disease Verification</div>
          <div style={{
        padding: 14,
        background: 'var(--gp)',
        borderRadius: 10,
        marginBottom: 14
      }}>
            <div style={{
          fontSize: 13.5,
          fontWeight: 700,
          color: 'var(--g2)',
          marginBottom: 4
        }}>✅ AI Report Available</div>
            <div style={{
          fontSize: 13,
          color: 'var(--tx2)'
        }}>🍅 Tomato — Early Blight • Conf: 94% • Severity: Stage 2/5 • Affected Area: 23%</div>
          </div>
          <div className="frow">
            <div className="fgrp"><label className="flbl">Crop</label>
              <select className="fsel" value={claimData.crop} onChange={e => setClaimData(p => ({
            ...p,
            crop: e.target.value
          }))}>
                <option value="">Select</option><option>Tomato</option><option>Wheat</option><option>Cotton</option>
              </select>
            </div>
            <div className="fgrp"><label className="flbl">Damage Date</label>
              <input className="finp" type="date" value={claimData.date} onChange={e => setClaimData(p => ({
            ...p,
            date: e.target.value
          }))} />
            </div>
          </div>
          <div className="frow">
            <div className="fgrp"><label className="flbl">Damage %</label>
              <select className="fsel" value={claimData.damage} onChange={e => setClaimData(p => ({
            ...p,
            damage: e.target.value
          }))}>
                <option value="">Select</option><option>10-25%</option><option>25-50%</option><option>50-75%</option><option>75%+</option>
              </select>
            </div>
            <div className="fgrp"><label className="flbl">Affected Area (Acres)</label>
              <input className="finp" type="number" placeholder="e.g. 1.5" value={claimData.area} onChange={e => setClaimData(p => ({
            ...p,
            area: e.target.value
          }))} />
            </div>
          </div>
          <button className="btn btn-g btn-full" onClick={() => {
        setStep(2);
        toast('Step 1 complete! ✅');
      }}>Next: Photo Upload →</button>
        </div>}
      {step === 2 && <div className="card" style={{
      padding: 22,
      marginBottom: 18
    }}>
          <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 14
      }}>📸 Photo Evidence Upload</div>
          <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3,1fr)',
        gap: 10,
        marginBottom: 14
      }}>
            {['Affected Area 1', 'Affected Area 2', 'Field Overview'].map((l, i) => <div key={l} onClick={() => toast(`${l} uploaded ✅`)} style={{
          height: 80,
          background: 'linear-gradient(135deg,var(--gp),var(--gpb))',
          borderRadius: 10,
          border: '2px dashed var(--br2)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          gap: 4
        }}>
                <div style={{
            fontSize: 20
          }}>📷</div>
                <div style={{
            fontSize: 10,
            fontWeight: 600,
            color: 'var(--tx3)',
            textAlign: 'center'
          }}>{l}</div>
              </div>)}
          </div>
          <div style={{
        padding: 12,
        background: 'var(--ap)',
        borderRadius: 9,
        fontSize: 12.5,
        color: 'var(--a1)',
        fontWeight: 600,
        marginBottom: 14
      }}>
            💡 Minimum 3 photos required: Affected leaves, field view, GPS timestamp
          </div>
          <div style={{
        display: 'flex',
        gap: 9
      }}>
            <button className="btn btn-ghost btn-md" style={{
          flex: 1
        }} onClick={() => setStep(1)}>← Wapas</button>
            <button className="btn btn-g btn-md" style={{
          flex: 2
        }} onClick={() => {
          setStep(3);
          toast('Photos uploaded! ✅');
        }}>Next: GPS Location →</button>
          </div>
        </div>}
      {step === 3 && <div className="card" style={{
      padding: 22,
      marginBottom: 18
    }}>
          <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 14
      }}>📍 Field GPS Assessment</div>
          <div style={{
        height: 120,
        background: 'linear-gradient(135deg,#c8e6c9,#a5d6a7)',
        borderRadius: 10,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 40,
        marginBottom: 14,
        cursor: 'pointer'
      }} onClick={() => toast('GPS location captured: 18.5912°N, 73.7389°E ✅')}>
            🗺️
          </div>
          <button className="btn btn-ghost btn-sm" style={{
        width: '100%',
        marginBottom: 14
      }} onClick={() => toast('GPS: 18.5912°N, 73.7389°E — Wagholi Farm ✅')}>📍 Current GPS Location Use Karo</button>
          <div style={{
        display: 'flex',
        gap: 9
      }}>
            <button className="btn btn-ghost btn-md" style={{
          flex: 1
        }} onClick={() => setStep(2)}>← Wapas</button>
            <button className="btn btn-g btn-md" style={{
          flex: 2
        }} onClick={() => {
          setStep(4);
          toast('GPS location saved! ✅');
        }}>Next: Submit Claim →</button>
          </div>
        </div>}
      {step === 4 && <div className="card" style={{
      padding: 22,
      marginBottom: 18
    }}>
          <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 14
      }}>📋 Claim Summary</div>
          {[['Policy Number', 'PMFBY-MH-2026-47821'], ['Crop', 'Tomato — Early Blight'], ['Damage', '25-50% (Stage 2/5)'], ['Affected Area', '1.5 Acres'], ['AI Confidence', '94% — Verified'], ['Estimated Payout', '₹38,000 – ₹48,000']].map(([k, v]) => <div key={k} style={{
        display: 'flex',
        justifyContent: 'space-between',
        padding: '9px 0',
        borderBottom: '1px solid var(--gp)',
        fontSize: 13
      }}>
              <span style={{
          color: 'var(--tx3)',
          fontWeight: 600
        }}>{k}</span>
              <span style={{
          fontWeight: 700,
          color: 'var(--tx)'
        }}>{v}</span>
            </div>)}
          <button className="btn btn-g btn-full" style={{
        marginTop: 16
      }} onClick={() => {
        setStep(5);
        toast('Claim submitted! Insurance company ko bhej diya gaya ✅');
      }}>
            ✅ Final Submit — Insurance Company Ko Bhejo
          </button>
        </div>}
      {step === 5 && <div style={{
      textAlign: 'center',
      padding: 20
    }}>
          <div style={{
        fontSize: 64,
        animation: 'bounce 1.2s infinite',
        marginBottom: 14
      }}>🎉</div>
          <div style={{
        fontFamily: "'Baloo 2',cursive",
        fontSize: 22,
        fontWeight: 900,
        color: 'var(--g2)',
        marginBottom: 8
      }}>Claim Approved!</div>
          <div style={{
        fontSize: 14,
        color: 'var(--tx2)',
        marginBottom: 20,
        lineHeight: 1.7
      }}>₹42,000 aapke Kisan Credit Card mein 3-5 working days mein transfer hoga.</div>
          <button className="btn btn-g btn-lg" onClick={() => nav('farmer-dashboard')}>🏠 Dashboard Par Jao</button>
        </div>}
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   8. GOVT DISEASE SURVEILLANCE MAP 📈
════════════════════════════════════════════════════════════════ */
