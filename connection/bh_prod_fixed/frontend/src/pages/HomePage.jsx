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
import BookingPage from './BookingPage.jsx';
import ExpertsPage from './ExpertsPage.jsx';
import FarmerDash from './FarmerDash.jsx';
import ExpertDash from './ExpertDash.jsx';
import MyConsultPage from './MyConsultPage.jsx';
import AIReportPage from './AIReportPage.jsx';
import ConsultPage from './ConsultPage.jsx';
import ExpertOnboarding from '../components/ExpertOnboarding.jsx';
import FarmerOnboarding from '../components/FarmerOnboarding.jsx';
import AuthModal from '../components/AuthModal.jsx';

export default /* ════════════════════════════════════════════════════════════════
   HOME PAGE
════════════════════════════════════════════════════════════════ */
function HomePage({
  nav,
  setAuth,
  setAuthMode,
  user
}) {
  const [contactForm, setContactForm] = useState({
    name: '',
    mobile: '',
    email: '',
    company: '',
    subject: '',
    message: '',
    type: 'farmer'
  });
  const [contactBusy, setContactBusy] = useState(false);
  const [contactDone, setContactDone] = useState(false);
  const [faqOpen, setFaqOpen] = useState(null);
  const submitContact = async () => {
    if (!contactForm.name.trim() || !contactForm.message.trim()) {
      alert('Naam aur message zaroor bharo');
      return;
    }
    setContactBusy(true);
    try {
      await API.post('/api/consultations', {
        cropId: 'contact',
        cropName: 'Website Enquiry',
        cropEmoji: '📩',
        method: 'manual',
        disease: 'Contact Form: ' + contactForm.subject,
        confidence: 100,
        severity: 1,
        answers: {
          ...contactForm
        }
      });
      setContactDone(true);
    } catch (e) {
      alert('Send fail hua, dobara try karein');
    }
    setContactBusy(false);
  };
  const FAQS = [{
    q: 'BeejHealth kaise kaam karta hai?',
    a: 'Aap apni fasal ki photo upload karte hain, AI (EfficientNetV2 model) se disease detect hoti hai 94%+ accuracy ke saath, phir certified expert se consultation lete hain.'
  }, {
    q: 'Kya ye free hai?',
    a: 'Basic AI scan aur Chat bilkul free hai. Audio Call aur Video Call paid hain. Insurance claims aur B2B pricing alag hai.'
  }, {
    q: 'Konsi crops support karta hai?',
    a: 'Abhi 17 crops: Tomato, Wheat, Potato, Cotton, Corn, Apple, Grape, Orange, Pepper, Soybean, Strawberry, Peach, Cherry, Blueberry, Squash, Raspberry, aur Coconut.'
  }, {
    q: 'Expert real doctors hain kya?',
    a: 'Haan — sab certified agricultural scientists hain. Plant Pathologists, Soil Scientists, Horticulture Experts — IARI, PAU, BHU jaise institutions se trained.'
  }, {
    q: 'Data private rahega?',
    a: 'Haan. Aapki farm data, photos aur consultations fully encrypted hain. Kisi third party ko share nahi karte.'
  }, {
    q: 'Coconut ke liye special kya hai?',
    a: 'Coconut ke liye EfficientNetV2-S Transfer Learning model se 8 disease classes detect hoti hain — Gray Leaf Spot, Leaf Rot, Bud Rot, Stem Bleeding. 30 adaptive questions sirf coconut ke liye.'
  }, {
    q: 'Robot farming kab aayegi?',
    a: 'BeejHealth Robot 2025-26 mein launch hoga — autonomous field scanning, targeted spray, real-time crop monitoring.'
  }, {
    q: 'Kya offline bhi kaam karta hai?',
    a: 'Basic features offline mein bhi kaam karte hain. Expert consultation ke liye internet chahiye. PWA support coming soon.'
  }];
  const TESTIMONIALS = [{
    name: 'Ramesh Patil',
    loc: 'Pune, Maharashtra',
    crop: '🍅 Tomato',
    text: 'BeejHealth ne meri poori tomato fasal bachi. AI ne exactly bata diya — Early Blight hai aur Mancozeb spray karo. 7 din mein result dikha.',
    rating: 5,
    saved: '₹45,000'
  }, {
    name: 'Gurpreet Singh',
    loc: 'Ludhiana, Punjab',
    crop: '🌾 Wheat',
    text: 'Expert Dr. Rajesh Kumar ne video call pe ek ghante mein poori problem solve kar di. Itni sasti expert consultation pehle kabhi nahi mili.',
    rating: 5,
    saved: '₹28,000'
  }, {
    name: 'Kavita Devi',
    loc: 'Solapur, Maharashtra',
    crop: '🌸 Cotton',
    text: 'Cotton mein Bacterial Blight aa gayi thi, photo upload ki, 30 second mein report aa gayi. Kamaal ki service!',
    rating: 5,
    saved: '₹62,000'
  }, {
    name: 'Mohammad Rizwan',
    loc: 'Nanded, Maharashtra',
    crop: '🥥 Coconut',
    text: 'Nariyal ke ped mein Bud Rot ho rahi thi. BeejHealth ke coconut specialist questions se exact diagnosis hua. Shukriya!',
    rating: 5,
    saved: '₹38,000'
  }, {
    name: 'Priya Shinde',
    loc: 'Nashik, Maharashtra',
    crop: '🍇 Grape',
    text: 'Grape export quality maintain karna tha. AI scan + expert se spray schedule banaya. Is season pehle se zyada production mili.',
    rating: 5,
    saved: '₹1,20,000'
  }, {
    name: 'Suresh Kumar',
    loc: 'Nagpur, Maharashtra',
    crop: '🍊 Orange',
    text: 'Satellite map se apna bagaan dekha, soil sensor data samjha. Technology ne farming ko alag hi level pe le gaya.',
    rating: 5,
    saved: '₹55,000'
  }];
  const TEAM = [{
    name: 'Dr. Anjali Sharma',
    role: 'Founder & CEO',
    bg: 'var(--g4)',
    init: 'AS',
    desc: 'Ex-IARI scientist, 15 years agricultural AI research'
  }, {
    name: 'Rahul Mehta',
    role: 'CTO',
    bg: 'var(--b3)',
    init: 'RM',
    desc: 'IIT Bombay, Ex-Google, ML & computer vision expert'
  }, {
    name: 'Dr. Priya Nair',
    role: 'Chief Agro Officer',
    bg: 'var(--a2)',
    init: 'PN',
    desc: 'Kerala Agri Univ, plant pathology specialist'
  }, {
    name: 'Vikram Singh',
    role: 'Head of Operations',
    bg: 'var(--g3)',
    init: 'VS',
    desc: 'Ex-Mahindra Agri, farmer network across 12 states'
  }];
  return <>
      {/* ══ HERO — Light green, original style ══════════════ */}
      <section className="hero">
        <div style={{
        position: 'absolute',
        width: 600,
        height: 600,
        background: 'radial-gradient(circle,rgba(77,189,122,.1),transparent)',
        top: -100,
        right: -80,
        borderRadius: '50%',
        pointerEvents: 'none'
      }} />
        <div style={{
        position: 'absolute',
        width: 350,
        height: 350,
        background: 'radial-gradient(circle,rgba(26,111,212,.07),transparent)',
        bottom: -60,
        left: -40,
        borderRadius: '50%',
        pointerEvents: 'none'
      }} />
        <div className="hero-in">
          <div style={{
          animation: 'slideUp .5s ease'
        }}>
            <div className="hero-pill" style={{
            marginBottom: 18
          }}>
              <div className="hero-dot" />🏆 India's #1 AI Farming Platform — 50,000+ Farmers Trust Us
            </div>
            <h1 className="hero-h1">Apni Fasal Ko<br /><em>Smart Banao</em> 🌱</h1>
            <p className="hero-p">
              BeejHealth — AI se crop disease instant detect karo, certified experts se real-time consult karo,
              weather alerts pao, aur apni farm ko digitally manage karo.{' '}
              <strong style={{
              color: 'var(--g3)'
            }}>Free mein shuru karo.</strong>
            </p>
            <div className="hero-btns" style={{
            marginBottom: 28
          }}>
              <button className="btn btn-g btn-xl" onClick={() => nav('consultation')}>🔬 Free Scan Karo</button>
              <button className="btn btn-out btn-xl" onClick={() => nav('experts')}>👨‍⚕️ Expert Dhundho</button>
            </div>
            {/* Trust badges */}
            <div style={{
            display: 'flex',
            gap: 14,
            flexWrap: 'wrap',
            marginBottom: 28
          }}>
              {['🔒 100% Secure', '✅ ICAR Certified', '🌐 12 Languages', '📱 Works Offline'].map(b => <div key={b} style={{
              fontSize: 12,
              fontWeight: 700,
              color: 'var(--tx3)',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              background: 'white',
              padding: '5px 12px',
              borderRadius: 100,
              border: '1px solid var(--br)',
              boxShadow: '0 1px 4px rgba(0,0,0,.04)'
            }}>{b}</div>)}
            </div>
            <div className="stats-row">
              {[['50K+', 'Farmers'], ['200+', 'Experts'], ['94%', 'Accuracy'], ['₹12Cr+', 'Saved']].map(([n, l]) => <div key={l}><div className="stat-n">{n}</div><div className="stat-l">{l}</div></div>)}
            </div>
          </div>

          {/* Hero Demo Card */}
          <div className="hero-card" style={{
          animation: 'slideUp .55s .1s ease both'
        }}>
            <div className="hc-lbl">🤖 Live AI Analysis Demo</div>
            <div className="dis-card">
              <div className="dc-crop">🥥 Coconut • Palakkad, Kerala</div>
              <div className="dc-name">Bud Rot Detected</div>
              <div className="dc-sci">Phytophthora palmivora • Stage 4/5</div>
              <div className="dc-bar-row"><span>AI Confidence</span><span style={{
                color: 'var(--r2)',
                fontWeight: 800
              }}>90.2%</span></div>
              <div className="dc-bar"><div className="dc-fill" style={{
                width: '90%',
                background: 'var(--r3)'
              }} /></div>
              <div className="dc-bar-row" style={{
              marginTop: 6
            }}><span>Treatment Urgency</span><span style={{
                color: 'var(--a1)',
                fontWeight: 800
              }}>HIGH</span></div>
              <div className="dc-bar"><div className="dc-fill" style={{
                width: '80%',
                background: 'var(--a3)'
              }} /></div>
              <div style={{
              display: 'flex',
              gap: 6,
              marginTop: 10,
              flexWrap: 'wrap'
            }}>
                <span style={{
                padding: '4px 10px',
                background: 'var(--rp)',
                borderRadius: 100,
                fontSize: 11,
                fontWeight: 700,
                color: 'var(--r2)'
              }}>🚨 Critical</span>
                <span style={{
                padding: '4px 10px',
                background: 'var(--bp)',
                borderRadius: 100,
                fontSize: 11,
                fontWeight: 700,
                color: 'var(--b2)'
              }}>EfficientNetV2-S</span>
                <span style={{
                padding: '4px 10px',
                background: 'var(--gp)',
                borderRadius: 100,
                fontSize: 11,
                fontWeight: 700,
                color: 'var(--g3)'
              }}>Expert Assigned</span>
              </div>
              <div className="dc-pill" style={{
              marginTop: 10
            }}>💊 Ridomil MZ 72 WP — Apply immediately</div>
            </div>
            <div style={{
            display: 'flex',
            gap: 8,
            marginTop: 12
          }}>
              <button className="btn btn-g btn-sm" style={{
              flex: 1
            }} onClick={() => nav('consultation')}>🔬 Try Free Scan</button>
              <button className="btn btn-out btn-sm" style={{
              flex: 1
            }} onClick={() => nav('experts')}>👨‍⚕️ Book Expert</button>
            </div>
            <div style={{
            marginTop: 10,
            padding: '8px 12px',
            background: 'var(--gp)',
            borderRadius: 8,
            fontSize: 12,
            color: 'var(--tx3)',
            display: 'flex',
            alignItems: 'center',
            gap: 8
          }}>
              <div style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: 'var(--g4)',
              flexShrink: 0
            }} />
              <span>23 farmers currently online • 4 experts available</span>
            </div>
          </div>
        </div>
      </section>

      {/* ══ ACHIEVEMENT STATS ══════════════════════════════════ */}
      <section style={{
      padding: '32px 28px',
      background: 'white',
      borderBottom: '1px solid var(--br)'
    }}>
        <div style={{
        maxWidth: 1160,
        margin: '0 auto'
      }}>
          <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(8,1fr)',
          gap: 8
        }}>
            {[['👨‍🌾', '50K+', 'Active Farmers'], ['👨‍⚕️', '200+', 'Certified Experts'], ['🎯', '94.3%', 'AI Accuracy'], ['🦠', '58+', 'Disease Classes'], ['🌾', '17', 'Crops Supported'], ['💰', '₹12Cr+', 'Farmer Savings'], ['⭐', '4.9★', 'App Rating'], ['🕐', '24/7', 'Expert Support']].map(([ic, n, l]) => <div key={l} style={{
            textAlign: 'center',
            padding: '12px 4px'
          }}>
                <div style={{
              fontSize: 20,
              marginBottom: 4
            }}>{ic}</div>
                <div style={{
              fontFamily: "'Baloo 2',cursive",
              fontSize: 20,
              fontWeight: 900,
              color: 'var(--g1)'
            }}>{n}</div>
                <div style={{
              fontSize: 10,
              color: 'var(--tx3)',
              fontWeight: 600,
              marginTop: 2,
              lineHeight: 1.3
            }}>{l}</div>
              </div>)}
          </div>
        </div>
      </section>

      {/* ══ HOW IT WORKS ══════════════════════════════════════ */}
      <section style={{
      padding: '72px 28px',
      background: 'var(--gb)'
    }}>
        <div style={{
        maxWidth: 1160,
        margin: '0 auto'
      }}>
          <div style={{
          textAlign: 'center',
          marginBottom: 48
        }}>
            <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 34,
            fontWeight: 900,
            color: 'var(--g1)',
            marginBottom: 8
          }}>⚡ Kaise Kaam Karta Hai?</div>
            <div style={{
            fontSize: 15,
            color: 'var(--tx2)'
          }}>3 easy steps — 30 seconds mein AI diagnosis</div>
          </div>
          <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3,1fr)',
          gap: 28,
          position: 'relative'
        }}>
            <div style={{
            position: 'absolute',
            top: 52,
            left: '16%',
            right: '16%',
            height: 2,
            background: 'linear-gradient(90deg,var(--g4),var(--b3))',
            zIndex: 0,
            borderRadius: 2
          }} />
            {[{
            n: '01',
            i: '📸',
            t: 'Photo Lo ya Upload Karo',
            d: 'Apni fasal ki koi bhi achi photo lo. Gallery se upload karo ya seedha camera se khicho.',
            c: 'var(--g4)',
            bg: 'var(--gp)'
          }, {
            n: '02',
            i: '🤖',
            t: 'AI Instant Diagnosis',
            d: 'EfficientNetV2 model 30 seconds mein disease pehchanta hai. 58+ diseases, 94.3% accuracy.',
            c: 'var(--b3)',
            bg: 'var(--bp)'
          }, {
            n: '03',
            i: '👨‍⚕️',
            t: 'Expert Report & Solution',
            d: 'Certified expert aapko AI report ke saath contact karta hai. Treatment plan, medicine list — sab detail mein.',
            c: 'var(--a2)',
            bg: 'var(--ap)'
          }].map((s, i) => <div key={s.n} className="card" style={{
            padding: 28,
            position: 'relative',
            zIndex: 1
          }}>
                <div style={{
              width: 52,
              height: 52,
              borderRadius: 14,
              background: s.bg,
              border: `2px solid ${s.c}44`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 24,
              marginBottom: 16
            }}>{s.i}</div>
                <div style={{
              position: 'absolute',
              top: 18,
              right: 20,
              fontFamily: "'Baloo 2',cursive",
              fontSize: 44,
              fontWeight: 900,
              color: 'var(--br)',
              lineHeight: 1
            }}>{s.n}</div>
                <div style={{
              fontSize: 16,
              fontWeight: 800,
              color: 'var(--tx)',
              marginBottom: 8
            }}>{s.t}</div>
                <div style={{
              fontSize: 13.5,
              color: 'var(--tx2)',
              lineHeight: 1.7
            }}>{s.d}</div>
              </div>)}
          </div>
          <div style={{
          textAlign: 'center',
          marginTop: 32
        }}>
            <button className="btn btn-g btn-xl" onClick={() => nav('consultation')}>🚀 Abhi Shuru Karo — Free</button>
          </div>
        </div>
      </section>

      {/* ══ FEATURES GRID ══════════════════════════════════════ */}
      <section style={{
      padding: '72px 28px',
      background: 'white'
    }}>
        <div style={{
        maxWidth: 1160,
        margin: '0 auto'
      }}>
          <div style={{
          textAlign: 'center',
          marginBottom: 48
        }}>
            <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 34,
            fontWeight: 900,
            color: 'var(--g1)',
            marginBottom: 8
          }}>🚀 Complete Farming Platform</div>
            <div style={{
            fontSize: 15,
            color: 'var(--tx2)'
          }}>Ek app mein sab kuch — AI, Experts, Weather, Market, Robots</div>
          </div>
          <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3,1fr)',
          gap: 20
        }}>
            {[{
            i: '🤖',
            t: 'AI Disease Detection',
            d: 'CNN + Transfer Learning. 30 seconds mein 58+ diseases detect. 94.3% accuracy on Indian crops.',
            tag: 'Free',
            bg: 'var(--gp)',
            tc: 'var(--g3)'
          }, {
            i: '👨‍⚕️',
            t: 'Certified Expert Network',
            d: '200+ verified Plant Pathologists, Soil Scientists, Horticulture Experts. Video call, voice call, written report.',
            tag: '₹300+',
            bg: 'var(--bp)',
            tc: 'var(--b2)'
          }, {
            i: '🥥',
            t: 'Coconut Specialist AI',
            d: 'Special EfficientNetV2-S model. 8 disease classes: Bud Rot, Stem Bleeding, Gray Leaf Spot, Leaf Rot.',
            tag: 'New',
            bg: 'var(--ap)',
            tc: 'var(--a1)'
          }, {
            i: '🌦️',
            t: 'Weather Intelligence',
            d: 'OpenWeatherMap + IMD data. Disease outbreak prediction. Spray timing alerts. District level forecast.',
            tag: 'Live',
            bg: 'var(--tp)',
            tc: 'var(--t2)'
          }, {
            i: '📊',
            t: 'Farm Analytics',
            d: 'Digital farm twin. Crop health history, disease trends, yield tracking, cost analysis. Satellite NDVI map.',
            tag: 'Pro',
            bg: 'var(--pup)',
            tc: 'var(--pu)'
          }, {
            i: '🛒',
            t: 'AgriMart + Price Tracker',
            d: 'AI-recommended medicines, seeds, fertilizers. Real-time mandi prices for 12 crops. Village delivery.',
            tag: 'Live',
            bg: 'var(--gp)',
            tc: 'var(--g3)'
          }, {
            i: '🏦',
            t: 'Insurance & Govt Schemes',
            d: 'PMFBY, WBCIS, Coconut Palm Insurance — ek click mein apply. Subsidy calculator.',
            tag: 'Free',
            bg: 'var(--bp)',
            tc: 'var(--b2)'
          }, {
            i: '🎤',
            t: 'Voice Input (Hindi)',
            d: 'Hindi mein bolo — AI samjhega. Crop naam, disease symptoms voice se describe karo.',
            tag: 'Beta',
            bg: 'var(--ap)',
            tc: 'var(--a1)'
          }, {
            i: '🤖',
            t: 'Robot Fleet (2025+)',
            d: 'BeejHealth autonomous robot — field scan, targeted spray, real-time monitoring. Early access open.',
            tag: 'Soon',
            bg: 'var(--tp)',
            tc: 'var(--t2)'
          }].map(f => <div key={f.t} className="card card-hov" style={{
            padding: 22,
            background: f.bg,
            border: 'none',
            position: 'relative'
          }}>
                <div style={{
              position: 'absolute',
              top: 14,
              right: 14,
              fontSize: 10,
              fontWeight: 800,
              padding: '3px 9px',
              borderRadius: 100,
              background: 'rgba(255,255,255,.8)',
              color: f.tc,
              border: `1px solid ${f.tc}33`
            }}>{f.tag}</div>
                <div style={{
              fontSize: 30,
              marginBottom: 12
            }}>{f.i}</div>
                <div style={{
              fontSize: 15,
              fontWeight: 800,
              color: 'var(--tx)',
              marginBottom: 7
            }}>{f.t}</div>
                <div style={{
              fontSize: 13,
              color: 'var(--tx2)',
              lineHeight: 1.65
            }}>{f.d}</div>
              </div>)}
          </div>
        </div>
      </section>

      {/* ══ TESTIMONIALS ══════════════════════════════════════ */}
      <section style={{
      padding: '72px 28px',
      background: 'var(--gb)'
    }}>
        <div style={{
        maxWidth: 1160,
        margin: '0 auto'
      }}>
          <div style={{
          textAlign: 'center',
          marginBottom: 44
        }}>
            <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 34,
            fontWeight: 900,
            color: 'var(--g1)',
            marginBottom: 8
          }}>❤️ Farmers Ki Kahaniyan</div>
            <div style={{
            fontSize: 15,
            color: 'var(--tx2)'
          }}>50,000+ farmers ne apni fasal bachaayi — unhi ki zubani</div>
          </div>
          <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3,1fr)',
          gap: 20
        }}>
            {TESTIMONIALS.map((t, i) => <div key={i} className="card" style={{
            padding: 22,
            borderTop: `3px solid var(--g4)`
          }}>
                <div style={{
              display: 'flex',
              gap: 2,
              marginBottom: 10
            }}>
                  {[...Array(t.rating)].map((_, j) => <span key={j} style={{
                color: 'var(--a2)',
                fontSize: 14
              }}>★</span>)}
                </div>
                <div style={{
              fontSize: 13.5,
              color: 'var(--tx)',
              lineHeight: 1.72,
              marginBottom: 16,
              fontStyle: 'italic'
            }}>"{t.text}"</div>
                <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
                  <div>
                    <div style={{
                  fontWeight: 800,
                  fontSize: 14,
                  color: 'var(--tx)'
                }}>{t.name}</div>
                    <div style={{
                  fontSize: 12,
                  color: 'var(--tx3)'
                }}>{t.crop} • {t.loc}</div>
                  </div>
                  <div style={{
                textAlign: 'right'
              }}>
                    <div style={{
                  fontSize: 10,
                  color: 'var(--tx3)',
                  marginBottom: 2
                }}>Saved</div>
                    <div style={{
                  fontFamily: "'Baloo 2',cursive",
                  fontSize: 18,
                  fontWeight: 900,
                  color: 'var(--g4)'
                }}>{t.saved}</div>
                  </div>
                </div>
              </div>)}
          </div>
        </div>
      </section>

      {/* ══ CROPS COVERAGE ══════════════════════════════════════ */}
      <section style={{
      padding: '56px 28px',
      background: 'white'
    }}>
        <div style={{
        maxWidth: 1160,
        margin: '0 auto'
      }}>
          <div style={{
          textAlign: 'center',
          marginBottom: 32
        }}>
            <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 30,
            fontWeight: 900,
            color: 'var(--g1)',
            marginBottom: 6
          }}>🌾 17 Crops — Growing List</div>
            <div style={{
            fontSize: 14,
            color: 'var(--tx2)'
          }}>Har crop ke liye specialized disease database aur treatment protocols</div>
          </div>
          <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 10,
          justifyContent: 'center'
        }}>
            {[['🍅', 'Tomato', '58 diseases'], ['🌾', 'Wheat', '24 diseases'], ['🥔', 'Potato', '31 diseases'], ['🌸', 'Cotton', '28 diseases'], ['🌽', 'Corn', '22 diseases'], ['🍎', 'Apple', '35 diseases'], ['🍇', 'Grape', '29 diseases'], ['🍊', 'Orange', '26 diseases'], ['🫑', 'Pepper', '33 diseases'], ['🫘', 'Soybean', '18 diseases'], ['🍓', 'Strawberry', '21 diseases'], ['🍑', 'Peach', '27 diseases'], ['🍒', 'Cherry', '24 diseases'], ['🫐', 'Blueberry', '19 diseases'], ['🎃', 'Squash', '22 diseases'], ['🍓', 'Raspberry', '18 diseases'], ['🥥', 'Coconut', '8 AI classes', 'NEW']].map(([em, nm, cnt, tag]) => <div key={nm} className="card card-hov" style={{
            padding: '10px 16px',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            cursor: 'pointer',
            minWidth: 140,
            border: tag ? '1.5px solid var(--g4)' : '1.5px solid var(--br)'
          }} onClick={() => nav('consultation')}>
                <span style={{
              fontSize: 24
            }}>{em}</span>
                <div>
                  <div style={{
                fontSize: 13,
                fontWeight: 800,
                color: 'var(--tx)',
                display: 'flex',
                alignItems: 'center',
                gap: 5
              }}>
                    {nm}{tag && <span style={{
                  fontSize: 9,
                  background: 'var(--g4)',
                  color: 'white',
                  padding: '1px 6px',
                  borderRadius: 100,
                  fontWeight: 700
                }}>{tag}</span>}
                  </div>
                  <div style={{
                fontSize: 11,
                color: 'var(--tx3)'
              }}>{cnt}</div>
                </div>
              </div>)}
          </div>
        </div>
      </section>

      {/* ══ TEAM SECTION ══════════════════════════════════════ */}
      <section style={{
      padding: '72px 28px',
      background: 'var(--gb)'
    }}>
        <div style={{
        maxWidth: 1160,
        margin: '0 auto'
      }}>
          <div style={{
          textAlign: 'center',
          marginBottom: 44
        }}>
            <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 34,
            fontWeight: 900,
            color: 'var(--g1)',
            marginBottom: 8
          }}>👥 Hamaari Team</div>
            <div style={{
            fontSize: 15,
            color: 'var(--tx2)'
          }}>Scientists, engineers aur farmers — milke farming ka future bana rahe hain</div>
          </div>
          <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4,1fr)',
          gap: 22
        }}>
            {TEAM.map(m => <div key={m.name} className="card" style={{
            padding: 24,
            textAlign: 'center'
          }}>
                <div style={{
              width: 68,
              height: 68,
              borderRadius: '50%',
              background: m.bg,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontSize: 22,
              fontWeight: 900,
              margin: '0 auto 14px'
            }}>{m.init}</div>
                <div style={{
              fontWeight: 800,
              fontSize: 15,
              color: 'var(--tx)',
              marginBottom: 3
            }}>{m.name}</div>
                <div style={{
              fontSize: 12,
              color: 'var(--g4)',
              fontWeight: 700,
              marginBottom: 8
            }}>{m.role}</div>
                <div style={{
              fontSize: 12,
              color: 'var(--tx2)',
              lineHeight: 1.6
            }}>{m.desc}</div>
              </div>)}
          </div>
        </div>
      </section>

      {/* ══ FAQ SECTION ══════════════════════════════════════ */}
      <section style={{
      padding: '72px 28px',
      background: 'white'
    }}>
        <div style={{
        maxWidth: 760,
        margin: '0 auto'
      }}>
          <div style={{
          textAlign: 'center',
          marginBottom: 44
        }}>
            <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 34,
            fontWeight: 900,
            color: 'var(--g1)',
            marginBottom: 8
          }}>❓ Aksar Pooche Jaane Wale Sawaal</div>
            <div style={{
            fontSize: 15,
            color: 'var(--tx2)'
          }}>Koi sawaal hai? Hum yahan hain</div>
          </div>
          <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 10
        }}>
            {FAQS.map((f, i) => <div key={i} className="card" style={{
            overflow: 'hidden',
            cursor: 'pointer'
          }} onClick={() => setFaqOpen(faqOpen === i ? null : i)}>
                <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '15px 20px'
            }}>
                  <div style={{
                fontWeight: 700,
                fontSize: 14.5,
                color: 'var(--tx)',
                paddingRight: 16
              }}>{f.q}</div>
                  <div style={{
                fontSize: 20,
                color: 'var(--g4)',
                flexShrink: 0,
                transform: faqOpen === i ? 'rotate(45deg)' : 'none',
                transition: 'transform .2s',
                lineHeight: 1
              }}>+</div>
                </div>
                {faqOpen === i && <div style={{
              padding: '0 20px 16px',
              fontSize: 13.5,
              color: 'var(--tx2)',
              lineHeight: 1.72,
              borderTop: '1px solid var(--br)',
              paddingTop: 12
            }}>
                    {f.a}
                  </div>}
              </div>)}
          </div>
        </div>
      </section>

      {/* ══ CONTACT / ENQUIRY FORM ══════════════════════════ */}
      <section style={{
      padding: '72px 28px',
      background: 'var(--gb)'
    }}>
        <div style={{
        maxWidth: 1100,
        margin: '0 auto',
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 48,
        alignItems: 'start'
      }}>
          <div>
            <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 34,
            fontWeight: 900,
            color: 'var(--g1)',
            marginBottom: 10
          }}>📩 Humse Baat Karo</div>
            <div style={{
            fontSize: 15,
            color: 'var(--tx2)',
            lineHeight: 1.75,
            marginBottom: 28
          }}>
              Farmer ho, expert banna chahte ho, B2B partnership chahiye, ya koi bhi sawaal hai — hum yahan hain. 24 ghante mein reply guaranteed.
            </div>
            <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 18
          }}>
              {[{
              i: '📧',
              t: 'Email',
              v: 'support@beejhealth.com'
            }, {
              i: '📱',
              t: 'WhatsApp / Phone',
              v: '+91 98765 43210'
            }, {
              i: '📍',
              t: 'Office',
              v: 'Bandra Kurla Complex, Mumbai, Maharashtra 400051'
            }, {
              i: '🕐',
              t: 'Support Hours',
              v: 'Mon-Sat: 8 AM – 8 PM IST'
            }].map(c => <div key={c.t} style={{
              display: 'flex',
              gap: 14,
              alignItems: 'flex-start'
            }}>
                  <div style={{
                width: 40,
                height: 40,
                borderRadius: 10,
                background: 'var(--gp)',
                border: '1px solid var(--br)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 18,
                flexShrink: 0
              }}>{c.i}</div>
                  <div>
                    <div style={{
                  fontSize: 11,
                  fontWeight: 700,
                  color: 'var(--tx3)',
                  textTransform: 'uppercase',
                  letterSpacing: .4,
                  marginBottom: 2
                }}>{c.t}</div>
                    <div style={{
                  fontSize: 14,
                  fontWeight: 600,
                  color: 'var(--tx)'
                }}>{c.v}</div>
                  </div>
                </div>)}
            </div>
            <div style={{
            marginTop: 24
          }}>
              <div style={{
              fontSize: 12,
              fontWeight: 700,
              color: 'var(--tx3)',
              marginBottom: 10,
              textTransform: 'uppercase',
              letterSpacing: .4
            }}>Social Media</div>
              <div style={{
              display: 'flex',
              gap: 9
            }}>
                {['📷', '🐦', '💼', '📘', '▶️'].map((ic, j) => <div key={j} style={{
                width: 38,
                height: 38,
                borderRadius: 9,
                background: 'white',
                border: '1px solid var(--br)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                fontSize: 16
              }}>{ic}</div>)}
              </div>
            </div>
          </div>

          <div className="card" style={{
          padding: 28
        }}>
            {contactDone ? <div style={{
            textAlign: 'center',
            padding: '36px 16px'
          }}>
                <div style={{
              fontSize: 52,
              marginBottom: 14
            }}>🎉</div>
                <div style={{
              fontFamily: "'Baloo 2',cursive",
              fontSize: 22,
              fontWeight: 900,
              color: 'var(--g1)',
              marginBottom: 8
            }}>Message Bhej Diya!</div>
                <div style={{
              fontSize: 14,
              color: 'var(--tx2)',
              lineHeight: 1.7
            }}>24 ghante mein humari team aapse contact karegi.</div>
                <button className="btn btn-g btn-md" style={{
              marginTop: 18
            }} onClick={() => setContactDone(false)}>Dobara Bhejo</button>
              </div> : <>
                <div style={{
              fontFamily: "'Baloo 2',cursive",
              fontSize: 20,
              fontWeight: 900,
              color: 'var(--g1)',
              marginBottom: 18
            }}>✉️ Enquiry Form</div>
                <div className="fgrp">
                  <label className="flbl">Aap Kaun Hain? *</label>
                  <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3,1fr)',
                gap: 8
              }}>
                    {[['farmer', '👨‍🌾 Farmer'], ['expert', '👨‍⚕️ Expert Banna'], ['business', '💼 Business']].map(([v, l]) => <button key={v} onClick={() => setContactForm(p => ({
                  ...p,
                  type: v
                }))} style={{
                  padding: '9px 6px',
                  borderRadius: 9,
                  fontSize: 12,
                  fontWeight: 700,
                  border: `2px solid ${contactForm.type === v ? 'var(--g4)' : 'var(--br)'}`,
                  background: contactForm.type === v ? 'var(--gp)' : 'white',
                  color: contactForm.type === v ? 'var(--g3)' : 'var(--tx2)',
                  cursor: 'pointer'
                }}>{l}</button>)}
                  </div>
                </div>
                <div className="frow">
                  <div className="fgrp">
                    <label className="flbl">Naam *</label>
                    <input className="finp" placeholder="Aapka poora naam" value={contactForm.name} onChange={e => setContactForm(p => ({
                  ...p,
                  name: e.target.value
                }))} />
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Mobile *</label>
                    <input className="finp" placeholder="10-digit" value={contactForm.mobile} maxLength={10} onChange={e => setContactForm(p => ({
                  ...p,
                  mobile: e.target.value.replace(/[^0-9]/g, '')
                }))} />
                  </div>
                </div>
                <div className="frow">
                  <div className="fgrp">
                    <label className="flbl">Email</label>
                    <input className="finp" type="email" placeholder="email@example.com" value={contactForm.email} onChange={e => setContactForm(p => ({
                  ...p,
                  email: e.target.value
                }))} />
                  </div>
                  <div className="fgrp">
                    <label className="flbl">Company / Farm</label>
                    <input className="finp" placeholder="Optional" value={contactForm.company} onChange={e => setContactForm(p => ({
                  ...p,
                  company: e.target.value
                }))} />
                  </div>
                </div>
                <div className="fgrp">
                  <label className="flbl">Subject</label>
                  <select className="fsel" value={contactForm.subject} onChange={e => setContactForm(p => ({
                ...p,
                subject: e.target.value
              }))}>
                    <option value="">-- Topic Select Karein --</option>
                    <option>Crop Disease Help</option>
                    <option>Expert Registration</option>
                    <option>B2B / Bulk Partnership</option>
                    <option>Investment / Funding</option>
                    <option>Technical Support</option>
                    <option>Media / Press</option>
                    <option>Other</option>
                  </select>
                </div>
                <div className="fgrp">
                  <label className="flbl">Message *</label>
                  <textarea className="ftxt" rows={4} placeholder="Aapka sawaal ya message..." value={contactForm.message} onChange={e => setContactForm(p => ({
                ...p,
                message: e.target.value
              }))} />
                </div>
                <button className="btn btn-g btn-full" onClick={submitContact} disabled={contactBusy || !contactForm.name || !contactForm.message}>
                  {contactBusy ? <><div className="spin" />Sending...</> : '📩 Message Bhejo →'}
                </button>
                <div style={{
              fontSize: 11,
              color: 'var(--tx4)',
              textAlign: 'center',
              marginTop: 8
            }}>🔒 Aapki jankari 100% safe hai. Koi spam nahi.</div>
              </>}
          </div>
        </div>
      </section>

      {/* ══ CTA BANNER — green theme (no dark) ══════════════ */}
      <section style={{
      padding: '60px 28px',
      background: 'var(--g1)',
      textAlign: 'center'
    }}>
        <div style={{
        maxWidth: 680,
        margin: '0 auto'
      }}>
          <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 34,
          fontWeight: 900,
          color: 'white',
          marginBottom: 10
        }}>🌱 Aaj Se Hi Shuru Karo</div>
          <div style={{
          fontSize: 15,
          color: 'rgba(255,255,255,.82)',
          marginBottom: 26,
          lineHeight: 1.75
        }}>
            50,000+ farmers already BeejHealth use kar rahe hain. Free scan karo, expert se milo, apni fasal ko bachao.
          </div>
          <div style={{
          display: 'flex',
          gap: 12,
          justifyContent: 'center',
          flexWrap: 'wrap'
        }}>
            <button className="btn btn-xl" style={{
            padding: '13px 34px',
            background: 'var(--g5)',
            color: 'white',
            border: 'none',
            fontSize: 15,
            borderRadius: 12
          }} onClick={() => nav('consultation')}>
              🔬 Free Scan Karo
            </button>
            <button className="btn btn-xl" style={{
            padding: '13px 34px',
            background: 'transparent',
            color: 'white',
            border: '1.5px solid rgba(255,255,255,.4)',
            fontSize: 15,
            borderRadius: 12
          }} onClick={() => {
            setAuthMode('register');
            setAuth(true);
          }}>
              📝 Register Karo
            </button>
          </div>
          <div style={{
          marginTop: 16,
          fontSize: 12,
          color: 'rgba(255,255,255,.5)'
        }}>No credit card required • Free forever for basic features</div>
        </div>
      </section>

      {/* ══ FOOTER ══════════════════════════════════════════ */}
      <footer className="footer">
        <div style={{
        maxWidth: 1160,
        margin: '0 auto'
      }}>
          <div style={{
          display: 'grid',
          gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr',
          gap: 36,
          marginBottom: 36
        }}>
            <div>
              <div style={{
              fontFamily: "'Baloo 2',cursive",
              fontSize: 24,
              fontWeight: 900,
              marginBottom: 10
            }}>🌱 BeejHealth</div>
              <div style={{
              fontSize: 13.5,
              opacity: .72,
              lineHeight: 1.75,
              marginBottom: 18,
              maxWidth: 260
            }}>India ka #1 AI farming platform. Crop disease detection, expert consultation, weather intelligence — sab ek jagah.</div>
              <div style={{
              marginBottom: 14
            }}>
                <div style={{
                fontSize: 10,
                opacity: .45,
                textTransform: 'uppercase',
                letterSpacing: .6,
                marginBottom: 7
              }}>Certified By</div>
                <div style={{
                display: 'flex',
                gap: 8,
                flexWrap: 'wrap'
              }}>
                  {['ICAR', 'IARI', 'NABARD', 'Startup India'].map(c => <span key={c} style={{
                  fontSize: 10,
                  padding: '2px 9px',
                  border: '1px solid rgba(255,255,255,.2)',
                  borderRadius: 100,
                  opacity: .65
                }}>{c}</span>)}
                </div>
              </div>
              <div style={{
              display: 'flex',
              gap: 9
            }}>
                {['📷', '🐦', '💼', '📘', '▶️'].map((s, i) => <div key={i} style={{
                width: 34,
                height: 34,
                borderRadius: 8,
                background: 'rgba(255,255,255,.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                fontSize: 14
              }}>{s}</div>)}
              </div>
            </div>
            {[['Platform', ['AI Scan (Free)', 'Expert Consult', 'Crop Forecast', 'AgriMart', 'Insurance', 'Robot Fleet']], ['Company', ['About Us', 'Our Team', 'Careers', 'Press & Media', 'Blog', 'Investors']], ['Support', ['Help Center', 'Video Tutorials', 'Terms of Service', 'Privacy Policy', 'Refund Policy', 'API Docs']], ['Contact', ['support@beejhealth.com', '+91 98765 43210', 'WhatsApp Support', 'Mumbai, Maharashtra', 'Mon-Sat 8AM-8PM', 'Emergency: 24/7']]].map(([title, items]) => <div key={title}>
                <div style={{
              fontSize: 10,
              fontWeight: 800,
              textTransform: 'uppercase',
              letterSpacing: .8,
              opacity: .45,
              marginBottom: 12
            }}>{title}</div>
                {items.map(item => <div key={item} style={{
              fontSize: 13,
              opacity: .68,
              cursor: 'pointer',
              marginBottom: 8
            }} onMouseEnter={e => e.currentTarget.style.opacity = '1'} onMouseLeave={e => e.currentTarget.style.opacity = '.68'}>
                    {item}
                  </div>)}
              </div>)}
          </div>
          <div style={{
          borderTop: '1px solid rgba(255,255,255,.1)',
          paddingTop: 18,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 10
        }}>
            <div style={{
            fontSize: 12,
            opacity: .5
          }}>© 2026 BeejHealth Technologies Pvt. Ltd. All rights reserved.</div>
            <div style={{
            fontSize: 12,
            opacity: .5
          }}>Made with 💚 for 140 Million Indian Farmers</div>
          </div>
        </div>
      </footer>
    </>;
}
