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
import ExpertOnboarding from './ExpertOnboarding.jsx';
import FarmerOnboarding from './FarmerOnboarding.jsx';

export default /* ════════════════════════════════════════════════════════════════
   AUTH MODAL
════════════════════════════════════════════════════════════════ */
function AuthModal({
  mode,
  setMode,
  onClose,
  onDone,
  initType = 'farmer'
}) {
  const [utype, setUtype] = useState(initType);
  const [lmethod, setLmethod] = useState('otp');
  const [otpStage, setOtpStage] = useState(false);
  const [regStep, setRegStep] = useState(1);
  const [busy, setBusy] = useState(false);
  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [f, setF] = useState({
    name: '',
    mobile: '',
    email: '',
    password: '',
    confirm: '',
    state: 'Maharashtra',
    district: '',
    taluka: '',
    village: '',
    soil: '',
    spec: '',
    fee: '',
    university: '',
    langs: ''
  });
  const [errs, setErrs] = useState({});
  const isEx = utype === 'expert';
  const thm = isEx ? 'var(--b3)' : 'var(--g4)';
  const cleanMobile = v => String(v ?? '').replace(/\D/g, '').slice(-10);
  const upd = (k, v) => setF(p => ({
    ...p,
    [k]: v
  }));
  const setE = (k, v) => setErrs(p => ({
    ...p,
    [k]: v
  }));
  const clr = () => setErrs({});
  const handleOtp = (i, v) => {
    if (!/^\d*$/.test(v)) return;
    const n = [...otp];
    n[i] = v;
    setOtp(n);
    if (v && i < 5) document.getElementById(`oc${i + 1}`)?.focus();
    if (!v && i > 0) document.getElementById(`oc${i - 1}`)?.focus();
  };
  const vLogin = () => {
    const e = {};
    if (cleanMobile(f.mobile).length < 10) e.mobile = 'Valid 10-digit number enter karein';
    if (lmethod === 'password' && !f.password) e.password = 'Password required';
    setErrs(e);
    return !Object.keys(e).length;
  };
  const vR1 = () => {
    const e = {};
    if (!f.name || f.name.length < 3) e.name = 'Naam 3+ characters ka hona chahiye';
    if (cleanMobile(f.mobile).length < 10) e.mobile = 'Valid mobile number';
    if (f.email && !/\S+@\S+\.\S+/.test(f.email)) e.email = 'Valid email';
    setErrs(e);
    return !Object.keys(e).length;
  };
  const vR2 = () => {
    const e = {};
    if (!f.district) e.district = 'District zaroor select karein';
    setErrs(e);
    return !Object.keys(e).length;
  };
  const vR3 = () => {
    const e = {};
    if (!f.password || f.password.length < 8) e.password = 'Password 8+ characters';
    if (f.password !== f.confirm) e.confirm = 'Passwords match nahi kar rahe';
    setErrs(e);
    return !Object.keys(e).length;
  };
  const doLogin = async () => {
    if (lmethod === 'otp' && !otpStage) {
      if (!vLogin()) return;
      setOtpStage(true);
      void API.post('/api/auth/send-otp', {
        mobile: cleanMobile(f.mobile)
      }).catch(e => setE('mobile', e.message));
      return;
    }
    if (lmethod === 'otp' && otp.join('').length < 6) {
      setE('otp', '6-digit OTP enter karein');
      return;
    }
    if (lmethod === 'password' && !vLogin()) return;
    setBusy(true);
    try {
      const res = await API.post('/api/auth/login', {
        mobile: cleanMobile(f.mobile),
        password: f.password,
        otp: otp.join(''),
        method: lmethod,
        type: utype
      });
      saveSession(res.token, res.user);
      onDone(res.user);
    } catch (e) {
      setE('mobile', e.message);
    }
    setBusy(false);
  };
  const doRegNext = async () => {
    if (regStep === 1 && !vR1()) return;
    if (regStep === 2 && !vR2()) return;
    if (regStep === 3) {
      if (!vR3()) return;
      setBusy(true);
      try {
        const res = await API.post('/api/auth/register', {
          name: f.name,
          mobile: cleanMobile(f.mobile),
          email: f.email,
          password: f.password,
          type: utype,
          state: f.state || 'Maharashtra',
          district: f.district,
          taluka: f.taluka,
          village: f.village,
          soil: f.soil,
          farmSize: Number(f.farmSize) || 0,
          irrigation: f.irrigation || '',
          spec: f.spec,
          fee: Number(f.fee) || 0,
          university: f.university,
          langs: f.langs || 'Hindi',
          expYrs: Number(f.expYrs) || 0,
          crops: ['tomato', 'wheat']
        });
        saveSession(res.token, res.user);
        onDone(res.user);
      } catch (e) {
        setE('password', e.message);
      }
      setBusy(false);
      return;
    }
    setRegStep(p => p + 1);
  };
  return <div className="overlay" onClick={onClose}>
      <div className="modal auth-modal" onClick={e => e.stopPropagation()}>
        <div className={`auth-head${isEx ? ' ex' : ''}`}>
          <button className="modal-close" onClick={onClose}>✕</button>
          <div style={{
          fontSize: 12,
          opacity: .78,
          marginBottom: 4
        }}>🌱 BeejHealth — {mode === 'login' ? 'Wapas Aayein' : 'Account Banayein'}</div>
          <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 22,
          fontWeight: 900
        }}>{mode === 'login' ? isEx ? 'Expert Login' : 'Farmer Login' : isEx ? 'Expert Register' : 'Farmer Register'}</div>
        </div>
        <div className="auth-body">
          {/* Type switcher */}
          <div className={`tabs-row${isEx ? ' ex-tabs' : ''}`} style={{
          marginBottom: 20
        }}>
            <button className={`tab-b${!isEx ? ' on' : ''}`} onClick={() => {
            setUtype('farmer');
            clr();
            setOtpStage(false);
            setRegStep(1);
          }}>🌾 Farmer</button>
            <button className={`tab-b${isEx ? ' on' : ''}`} onClick={() => {
            setUtype('expert');
            clr();
            setOtpStage(false);
            setRegStep(1);
          }}>👨‍⚕️ Expert</button>
          </div>

          {mode === 'login' ? <>
              {!otpStage ? <>
                  <div style={{
              display: 'flex',
              gap: 8,
              marginBottom: 18
            }}>
                    {['otp', 'password'].map(m => <button key={m} onClick={() => setLmethod(m)} style={{
                flex: 1,
                padding: '9px',
                borderRadius: 9,
                fontSize: 13,
                fontWeight: 700,
                border: `2px solid ${lmethod === m ? thm : 'var(--br)'}`,
                background: lmethod === m ? isEx ? 'var(--bp)' : 'var(--gp)' : 'none',
                color: lmethod === m ? thm : 'var(--tx2)',
                cursor: 'pointer',
                fontFamily: "'Outfit',sans-serif",
                transition: 'all .18s'
              }}>
                        {m === 'otp' ? '📱 OTP Login' : '🔑 Password'}
                      </button>)}
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Mobile Number</label>
                    <input className="finp" placeholder="10-digit mobile number" value={f.mobile} onChange={e => upd('mobile', e.target.value.replace(/\D/g, '').slice(-10))} maxLength={10} />
                    {errs.mobile && <div className="ferr">⚠️ {errs.mobile}</div>}
                  </div>
                  {lmethod === 'password' && <div className="fgrp">
                      <label className="flbl">Password</label>
                      <input className="finp" type="password" placeholder="Aapka password" value={f.password} onChange={e => upd('password', e.target.value)} />
                      {errs.password && <div className="ferr">⚠️ {errs.password}</div>}
                    </div>}
                  <button className="btn btn-g btn-full" style={{
              background: thm,
              boxShadow: `0 3px 14px ${isEx ? 'rgba(26,111,212,.26)' : 'rgba(30,126,66,.28)'}`
            }} onClick={doLogin}>
                    {busy ? <><div className="spin" />Processing...</> : lmethod === 'otp' ? 'OTP Bhejo →' : 'Login Karo →'}
                  </button>
                </> : <>
                  <div style={{
              textAlign: 'center',
              fontSize: 13.5,
              color: 'var(--tx2)',
              marginBottom: 4
            }}>
                    6-digit OTP bheja gaya: <strong>+91 {cleanMobile(f.mobile)}</strong>
                  </div>
                  <div style={{
              textAlign: 'center',
              fontSize: 12,
              color: 'var(--tx3)',
              marginBottom: 4
            }}>(Demo: koi bhi 6 digits chalenge)</div>
                  <div className="otp-row">
                    {otp.map((v, i) => <input key={i} id={`oc${i}`} className="otp-c" maxLength={1} value={v} onChange={e => handleOtp(i, e.target.value)} />)}
                  </div>
                  {errs.otp && <div className="ferr" style={{
              justifyContent: 'center',
              marginBottom: 8
            }}>⚠️ {errs.otp}</div>}
                  <div style={{
              textAlign: 'center',
              fontSize: 13,
              color: 'var(--tx2)',
              margin: '4px 0 14px'
            }}>
                    Code nahi mila? <span style={{
                color: thm,
                fontWeight: 700,
                cursor: 'pointer'
              }}>Resend</span>
                  </div>
                  <button className="btn btn-g btn-full" style={{
              background: thm
            }} onClick={doLogin}>
                    {busy ? <><div className="spin" />Login Ho Raha Hai...</> : '✅ Login Karo'}
                  </button>
                  <button style={{
              width: '100%',
              marginTop: 9,
              padding: '10px',
              border: 'none',
              background: 'none',
              color: 'var(--tx3)',
              fontSize: 13,
              cursor: 'pointer'
            }} onClick={() => setOtpStage(false)}>← Wapas Jao</button>
                </>}
              <div className="auth-or">OR</div>
              <button className="g-btn"><span style={{
              fontWeight: 900,
              fontSize: 16,
              color: '#4285f4'
            }}>G</span> Continue with Google</button>
              <div className="auth-sw">Account nahi hai? <span className={isEx ? 'ex-link' : ''} onClick={() => {
              setMode('register');
              clr();
              setRegStep(1);
            }}>Register Karein</span></div>
            </> : <>
              {/* Step indicator */}
              <div className="steps-row" style={{
            marginBottom: 22
          }}>
                {['Personal', 'Location', 'Password'].map((s, i) => <div key={i} style={{
              display: 'flex',
              alignItems: 'center',
              flex: 1
            }}>
                    <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center'
              }}>
                      <div className={`step-dot${regStep > i + 1 ? ' done' : regStep === i + 1 ? ' act' : ''}`} style={regStep === i + 1 ? {
                  borderColor: thm,
                  color: isEx ? 'var(--b3)' : 'var(--g3)'
                } : {}}>
                        {regStep > i + 1 ? '✓' : i + 1}
                      </div>
                      <div className="step-lbl">{s}</div>
                    </div>
                    {i < 2 && <div className={`step-ln${regStep > i + 1 ? ' done' : ''}`} style={regStep > i + 1 ? {
                background: thm
              } : {}} />}
                  </div>)}
              </div>

              {regStep === 1 && <div className="fade-in">
                  <div className="fgrp">
                    <label className="flbl">Poora Naam *</label>
                    <input className="finp" placeholder="Aapka poora naam" value={f.name} onChange={e => upd('name', e.target.value)} />
                    {errs.name && <div className="ferr">⚠️ {errs.name}</div>}
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Mobile Number *</label>
                    <input className="finp" placeholder="10-digit mobile" value={f.mobile} onChange={e => upd('mobile', e.target.value.replace(/\D/, '').slice(0, 10))} maxLength={10} />
                    {errs.mobile && <div className="ferr">⚠️ {errs.mobile}</div>}
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Email (Optional)</label>
                    <input className="finp" type="email" placeholder="Email address" value={f.email} onChange={e => upd('email', e.target.value)} />
                    {errs.email && <div className="ferr">⚠️ {errs.email}</div>}
                  </div>
                  {isEx && <>
                      <div className="frow">
                        <div className="fgrp">
                          <label className="flbl">Specialization *</label>
                          <select className="fsel" value={f.spec} onChange={e => upd('spec', e.target.value)}>
                            <option value="">Select</option>
                            <option>Plant Pathologist</option>
                            <option>Horticulture Expert</option>
                            <option>Soil Scientist</option>
                            <option>Crop Scientist</option>
                            <option>Agri Economist</option>
                          </select>
                        </div>
                        <div className="fgrp">
                          <label className="flbl">Fee (₹/session)</label>
                          <input className="finp" type="number" placeholder="e.g. 800" value={f.fee} onChange={e => upd('fee', e.target.value)} />
                        </div>
                      </div>
                      <div className="fgrp">
                        <label className="flbl">University / Institution</label>
                        <input className="finp" placeholder="e.g. IARI New Delhi" value={f.university} onChange={e => upd('university', e.target.value)} />
                      </div>
                    </>}
                </div>}

              {regStep === 2 && <div className="fade-in">
                  {!isEx ? <>
                      {/* STATE */}
                      <div className="fgrp">
                        <label className="flbl">State *</label>
                        <select className="fsel" value={f.state} onChange={e => {
                  upd('state', e.target.value);
                  upd('district', '');
                  upd('taluka', '');
                  upd('village', '');
                }}>
                          <option value="">-- State Select Karein --</option>
                          {INDIA_STATES.map(s => <option key={s} value={s}>{s}</option>)}
                        </select>
                      </div>
                      {/* DISTRICT + TALUKA */}
                      <div className="frow">
                        <div className="fgrp">
                          <label className="flbl">District *</label>
                          <select className="fsel" value={f.district} onChange={e => {
                    upd('district', e.target.value);
                    upd('taluka', '');
                    upd('village', '');
                  }} disabled={!f.state}>
                            <option value="">-- District --</option>
                            {getDistricts(f.state).map(d => <option key={d} value={d}>{d}</option>)}
                          </select>
                          {errs.district && <div className="ferr">⚠️ {errs.district}</div>}
                        </div>
                        <div className="fgrp">
                          <label className="flbl">Taluka / Block</label>
                          <select className="fsel" value={f.taluka} onChange={e => {
                    upd('taluka', e.target.value);
                    upd('village', '');
                  }} disabled={!f.district}>
                            <option value="">-- Taluka --</option>
                            {getStateTalukas(f.state, f.district).map(t => <option key={t} value={t}>{t}</option>)}
                          </select>
                        </div>
                      </div>
                      {/* VILLAGE + SOIL */}
                      <div className="frow">
                        <div className="fgrp">
                          <label className="flbl">Village / Gaon</label>
                          <input className="finp" placeholder="Aapke gaon ka naam" value={f.village} onChange={e => upd('village', e.target.value)} />
                        </div>
                        <div className="fgrp">
                          <label className="flbl">Soil Type</label>
                          <select className="fsel" value={f.soil} onChange={e => upd('soil', e.target.value)}>
                            <option value="">-- Soil Type --</option>
                            {SOILS.map(s => <option key={s} value={s}>{s}</option>)}
                          </select>
                        </div>
                      </div>
                      {/* FARM SIZE */}
                      <div className="frow">
                        <div className="fgrp">
                          <label className="flbl">Farm Size (Acres)</label>
                          <input className="finp" type="number" placeholder="e.g. 5" value={f.farmSize || ''} onChange={e => upd('farmSize', e.target.value)} />
                        </div>
                        <div className="fgrp">
                          <label className="flbl">Irrigation Source</label>
                          <select className="fsel" value={f.irrigation || ''} onChange={e => upd('irrigation', e.target.value)}>
                            <option value="">Select</option>
                            {['Borewell / Tube well', 'Canal / Nahr', 'Rainwater / Baarish', 'Drip Irrigation', 'River / Nadi', 'Tank / Taalaab', 'None / Barani'].map(o => <option key={o}>{o}</option>)}
                          </select>
                        </div>
                      </div>
                    </> : <>
                      {/* Expert State + District */}
                      <div className="frow">
                        <div className="fgrp">
                          <label className="flbl">State *</label>
                          <select className="fsel" value={f.state} onChange={e => {
                    upd('state', e.target.value);
                    upd('district', '');
                  }}>
                            <option value="">-- State --</option>
                            {INDIA_STATES.map(s => <option key={s} value={s}>{s}</option>)}
                          </select>
                        </div>
                        <div className="fgrp">
                          <label className="flbl">District *</label>
                          <select className="fsel" value={f.district} onChange={e => upd('district', e.target.value)} disabled={!f.state}>
                            <option value="">-- District --</option>
                            {getDistricts(f.state).map(d => <option key={d} value={d}>{d}</option>)}
                          </select>
                          {errs.district && <div className="ferr">⚠️ {errs.district}</div>}
                        </div>
                      </div>
                      <div className="fgrp">
                        <label className="flbl">Languages Spoken</label>
                        <input className="finp" placeholder="e.g. Hindi, English, Marathi, Punjabi" value={f.langs} onChange={e => upd('langs', e.target.value)} />
                      </div>
                      <div className="frow">
                        <div className="fgrp">
                          <label className="flbl">Years of Experience</label>
                          <input className="finp" type="number" placeholder="e.g. 10" value={f.expYrs || ''} onChange={e => upd('expYrs', e.target.value)} />
                        </div>
                        <div className="fgrp">
                          <label className="flbl">Crop Specializations</label>
                          <input className="finp" placeholder="e.g. Tomato, Wheat, Cotton" value={f.crops || ''} onChange={e => upd('crops', e.target.value)} />
                        </div>
                      </div>
                      <div style={{
                padding: 13,
                background: 'var(--bp)',
                borderRadius: 10,
                fontSize: 13,
                color: 'var(--b1)',
                fontWeight: 600
              }}>
                        ℹ️ Documents verify honge approval ke baad (2–3 din)
                      </div>
                    </>}
                </div>}

              {regStep === 3 && <div className="fade-in">
                  <div style={{
              textAlign: 'center',
              padding: 14,
              background: 'var(--gp)',
              borderRadius: 10,
              marginBottom: 18,
              fontSize: 13.5,
              color: 'var(--g2)',
              fontWeight: 600
            }}>
                    🎉 Almost done! Sirf password set karein.
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Password Banayein *</label>
                    <input className="finp" type="password" placeholder="Min 8 characters" value={f.password} onChange={e => upd('password', e.target.value)} />
                    {errs.password && <div className="ferr">⚠️ {errs.password}</div>}
                    <div style={{
                fontSize: 11,
                color: 'var(--tx3)',
                marginTop: 4
              }}>Letters, numbers & symbols mix karein</div>
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Confirm Password *</label>
                    <input className="finp" type="password" placeholder="Dobara likhein" value={f.confirm} onChange={e => upd('confirm', e.target.value)} />
                    {errs.confirm && <div className="ferr">⚠️ {errs.confirm}</div>}
                  </div>
                  <div style={{
              fontSize: 12,
              color: 'var(--tx3)',
              marginBottom: 14,
              lineHeight: 1.6
            }}>
                    Account banake aap BeejHealth Terms of Service aur Privacy Policy se agree karte hain.
                  </div>
                </div>}

              <div style={{
            display: 'flex',
            gap: 9,
            marginTop: 6
          }}>
                {regStep > 1 && <button className="btn btn-out btn-md" style={{
              flex: 1
            }} onClick={() => setRegStep(p => p - 1)}>← Wapas</button>}
                <button className="btn btn-g btn-md" style={{
              flex: 2,
              background: thm
            }} onClick={doRegNext}>
                  {busy ? <><div className="spin" />Saving...</> : regStep < 3 ? 'Aage Badho →' : '✅ Account Banao'}
                </button>
              </div>
              <div className="auth-or">OR</div>
              <button className="g-btn"><span style={{
              fontWeight: 900,
              fontSize: 16,
              color: '#4285f4'
            }}>G</span> Continue with Google</button>
              <div className="auth-sw">Account hai? <span className={isEx ? 'ex-link' : ''} onClick={() => {
              setMode('login');
              clr();
              setRegStep(1);
            }}>Login Karein</span></div>
            </>}
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   FARMER ONBOARDING
════════════════════════════════════════════════════════════════ */
