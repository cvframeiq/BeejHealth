import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import { API, saveSession, clearSession, loadSession } from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody } from '../utils/helpers.jsx';
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary } from '../utils/constants.jsx';
import { cropName, formatCurrency, formatDateTime, formatNumber, formatPercent, reportValue, tx } from '../utils/localize.jsx';

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
import ExpertDash from './ExpertDash.jsx';
import MyConsultPage from './MyConsultPage.jsx';
import AIReportPage from './AIReportPage.jsx';
import ConsultPage from './ConsultPage.jsx';
import HomePage from './HomePage.jsx';
import ExpertOnboarding from '../components/ExpertOnboarding.jsx';
import FarmerOnboarding from '../components/FarmerOnboarding.jsx';
import AuthModal from '../components/AuthModal.jsx';

export default /* ════════════════════════════════════════════════════════════════
   QUESTION FLOW STYLES
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   CONSULT PAGE  — Crop → Upload → Question Flow → Processing
════════════════════════════════════════════════════════════════ */
/* ════════════════════════════════════════════════════════════════
   FARMER DASHBOARD
════════════════════════════════════════════════════════════════ */
function FarmerDash({
  user,
  nav,
  toast
}) {
  const { i18n } = useTranslation();
  const N = value => formatNumber(value, i18n);
  const P = value => formatPercent(value, i18n);
  const C = value => formatCurrency(value, i18n);
  const D = value => formatDateTime(value, i18n, { day: '2-digit', month: 'short', year: 'numeric' });
  const RV = value => reportValue(value, i18n);
  const dashText = {
    en: {
      hello: tx(i18n, 'hello'),
      activeCrops: (count) => `${count} crops are active on your farm today`,
      farmHealthScore: 'Farm Health Score',
      farmHealth: 'Farm Health',
      cropCheck: 'Crop Check',
      voice: 'Voice Input',
      forecast: 'Forecast',
      satellite: 'Satellite',
      soil: 'Soil Sensors',
      robots: 'Robots',
      insurance: 'Insurance',
      market: 'Market',
      outbreak: 'Disease Outbreak:',
      outbreakText: 'Late Blight - 8 cases within 5km of your farm! Do preventive spray.',
      view: 'View',
      myCrops: 'My Crops',
      viewAll: 'View All',
      newCrop: 'New Crop',
      add: 'Add',
      recent: 'Recent Consultations',
      all: 'All'
    },
    hi: {
      hello: tx(i18n, 'hello'),
      activeCrops: (count) => `आज आपके खेत में ${count} फसलें सक्रिय हैं`,
      farmHealthScore: 'खेत स्वास्थ्य स्कोर',
      farmHealth: 'खेत स्वास्थ्य',
      cropCheck: 'फसल जांच',
      voice: 'वॉइस इनपुट',
      forecast: 'पूर्वानुमान',
      satellite: 'सैटेलाइट',
      soil: 'मिट्टी सेंसर',
      robots: 'रोबोट',
      insurance: 'बीमा',
      market: 'बाज़ार',
      outbreak: 'रोग अलर्ट:',
      outbreakText: 'लेट ब्लाइट - आपके खेत से 5km के अंदर 8 केस! बचाव स्प्रे करें.',
      view: 'देखें',
      myCrops: 'मेरी फसलें',
      viewAll: 'सब देखें',
      newCrop: 'नई फसल',
      add: 'जोड़ें',
      recent: 'हाल की सलाह',
      all: 'सब'
    },
    mr: {
      hello: tx(i18n, 'hello'),
      activeCrops: (count) => `आज तुमच्या शेतात ${count} पिके सक्रिय आहेत`,
      farmHealthScore: 'शेत आरोग्य स्कोर',
      farmHealth: 'शेत आरोग्य',
      cropCheck: 'पीक तपासा',
      voice: 'व्हॉइस इनपुट',
      forecast: 'अंदाज',
      satellite: 'सॅटेलाइट',
      soil: 'माती सेन्सर',
      robots: 'रोबोट',
      insurance: 'विमा',
      market: 'बाजार',
      outbreak: 'रोग अलर्ट:',
      outbreakText: 'लेट ब्लाइट - तुमच्या शेतापासून 5km मध्ये 8 केस! प्रतिबंधक फवारणी करा.',
      view: 'पाहा',
      myCrops: 'माझी पिके',
      viewAll: 'सर्व पाहा',
      newCrop: 'नवे पीक',
      add: 'जोडा',
      recent: 'अलीकडील सल्ले',
      all: 'सर्व'
    }
  };
  const dt = dashText[(i18n.language || 'en').split('-')[0]] || dashText.en;
  const myCrops = CROPS.filter(c => user?.crops?.includes(c.id));
  const displayCrops = myCrops.length ? myCrops : CROPS.slice(0, 5);
  const [farmScore, setFarmScore] = useState(78);
  const [farmConsults, setFarmConsults] = useState(CONSULTATIONS.slice(0, 3));
  useEffect(() => {
    if (!user) return;
    API.get('/api/consultations').then(d => {
      if (d.consultations && d.consultations.length > 0) {
        const list = d.consultations;
        const recent = list.slice(0, 5);
        const avgSev = recent.reduce((s, c) => s + (c.severity || 1), 0) / recent.length;
        const completedRatio = list.filter(c => c.status === 'completed').length / list.length;
        const calculated = Math.max(40, Math.min(100, Math.round(100 - avgSev * 8 + completedRatio * 10)));
        setFarmScore(calculated);
        setFarmConsults(list.slice(0, 3).map(c => ({
          id: c._id,
          emoji: c.cropEmoji || '🌱',
          crop: cropName(c.cropId || c.cropName, i18n) || c.cropName,
          issue: `${RV(c.disease)} — ${P(c.confidence || 0)} ${tx(i18n, 'confidenceWord')}`,
          date: D(c.createdAt) + ' · ' + formatDateTime(c.createdAt, i18n, {
            hour: '2-digit',
            minute: '2-digit'
          }),
          expert: c.expertName || 'Auto-assign',
          status: c.status === 'completed' ? 'completed' : c.status === 'expert_assigned' ? 'expert' : 'pending',
          statusLabel: c.status === 'completed' ? tx(i18n, 'completed') : c.status === 'expert_assigned' ? tx(i18n, 'expertAssigned') : tx(i18n, 'pending'),
          sev: c.severity || 1,
          conf: c.confidence || 0
        })));
      }
    }).catch(() => {});
  }, [user]);
  return <div className="wrap">
      {/* ── GREETING CARD ── */}
      <div className="greet-card">
        <div className="gc-ring1" /><div className="gc-ring2" />
        <div className="gc-top">
          <div>
            <div className="gc-name">{dt.hello}, {user?.name?.split(' ')?.[0] || 'Kisan'}! 🙏</div>
            <div className="gc-sub">{dt.activeCrops(N(displayCrops.length))}</div>
          </div>
          <div style={{
          textAlign: 'right'
        }}>
            <div className="gc-score-n">{N(farmScore)}</div>
            <div className="gc-score-l">{dt.farmHealthScore} 🏆</div>
          </div>
        </div>
        <div style={{
        marginBottom: 16
      }}>
          <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 12,
          fontWeight: 600,
          opacity: .82,
          marginBottom: 5
        }}>
            <span>{dt.farmHealth}</span><span>{N(farmScore)}/{N(100)}</span>
          </div>
          <div className="gc-prog"><div className="gc-prog-f" style={{
            width: `${farmScore}%`
          }} /></div>
        </div>
        <div className="gc-btns">
          <button className="gc-btn prim" onClick={() => nav('consultation')}>🔬 {dt.cropCheck}</button>
          <button className="gc-btn" onClick={() => nav('voice')}>🎤 {dt.voice}</button>
          <button className="gc-btn" onClick={() => nav('forecast')}>📊 {dt.forecast}</button>
          <button className="gc-btn" onClick={() => nav('satellite')}>🛰️ {dt.satellite}</button>
          <button className="gc-btn" onClick={() => nav('soil-sensors')}>🌱 {dt.soil}</button>
          <button className="gc-btn" onClick={() => nav('robot-dashboard')}>🤖 {dt.robots}</button>
          <button className="gc-btn" onClick={() => nav('insurance')}>🏦 {dt.insurance}</button>
          <button className="gc-btn" onClick={() => nav('marketplace')}>📦 {dt.market}</button>
        </div>
      </div>

      {/* ── URGENT ALERT BANNER ── */}
      <div className="alert-bar">
        <div className="alert-txt">🔴 <span>{dt.outbreak}</span> {dt.outbreakText}</div>
        <button className="btn btn-red btn-sm" onClick={() => nav('notifications')}>{dt.view} →</button>
      </div>

      {/* ── MY CROPS (Horizontal scroll) ── */}
      <div style={{
      marginBottom: 24
    }}>
        <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 13
      }}>
          <div style={{
          fontSize: 16,
          fontWeight: 800,
          color: 'var(--g1)'
        }}>🌾 {dt.myCrops}</div>
          <button className="btn btn-ghost btn-sm" onClick={() => nav('my-farm')}>{dt.viewAll} →</button>
        </div>
        <div className="crops-scroll">
          {displayCrops.map(c => <div key={c.id} className="crop-sc-card" onClick={() => {
          nav('consultation');
        }}>
              <div className="crop-sc-em">{c.emoji}</div>
              <div className="crop-sc-nm">{cropName(c, i18n)}</div>
              <div className="crop-sc-hl" style={{
            color: c.health > 80 ? 'var(--g4)' : c.health > 60 ? 'var(--a2)' : 'var(--r2)'
          }}>● {P(c.health)}</div>
              <div className="crop-sc-st">{c.stage}</div>
            </div>)}
          <div className="crop-sc-card" style={{
          background: 'var(--gp)',
          border: '2px dashed var(--br2)'
        }} onClick={() => nav('consultation')}>
            <div style={{
            fontSize: 26,
            marginBottom: 7
          }}>➕</div>
            <div style={{
            fontSize: 12.5,
            fontWeight: 700,
            color: 'var(--g3)'
          }}>{dt.newCrop}</div>
            <div style={{
            fontSize: 11,
            color: 'var(--tx3)',
            marginTop: 3
          }}>{dt.add}</div>
          </div>
        </div>
      </div>

      {/* ── MAIN 2-COL GRID ── */}
      <div className="dash-2">
        {/* LEFT: Recent Consultations */}
        <div>
          <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 13
        }}>
            <div style={{
            fontSize: 16,
            fontWeight: 800,
            color: 'var(--g1)'
          }}>📋 {dt.recent}</div>
            <button className="btn btn-ghost btn-sm" onClick={() => nav('my-consultations')}>{dt.all} →</button>
          </div>
          {farmConsults.map(c => <div key={c.id} className="cons-card card-hov" style={{
          marginBottom: 11
        }} onClick={() => {
          rememberConsultationContext(c.id);
          nav('ai-report');
        }}>
              <div style={{
            display: 'flex',
            gap: 11,
            padding: 14
          }}>
                <div style={{
              width: 56,
              height: 56,
              borderRadius: 10,
              flexShrink: 0,
              fontSize: 28,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'linear-gradient(135deg,var(--gp),var(--gpb))'
            }}>{c.emoji}</div>
                <div style={{
              flex: 1,
              minWidth: 0
            }}>
                  <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginBottom: 3,
                alignItems: 'flex-start',
                gap: 6
              }}>
                    <div className="cons-nm">{cropName(c.crop, i18n) || c.crop}</div>
                    <span className={`badge ${c.status === 'completed' ? 'bg-g' : c.status === 'expert' ? 'bg-b' : c.status === 'pending' ? 'bg-a' : 'bg-p'}`} style={{
                  flexShrink: 0,
                  fontSize: 11
                }}>{c.status === 'completed' ? '✅' : c.status === 'expert' ? '👨‍⚕️' : c.status === 'pending' ? '⏳' : '🔵'} {c.statusLabel}</span>
                  </div>
                  <div className="cons-issue">{c.issue}</div>
                  <div style={{
                fontSize: 11.5,
                color: 'var(--tx3)',
                marginBottom: 8
                }}>👨‍⚕️ {c.expert} • {c.date.split('·')[0].trim()}</div>
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
                }}>💬 {tx(i18n, 'chatAction')}</button>
                  </div>
                </div>
              </div>
            </div>)}
          <button className="btn btn-out btn-sm" style={{
          width: '100%',
          marginTop: 4
        }} onClick={() => nav('my-consultations')}>
            {tx(i18n, 'viewAllConsultations')} →
          </button>
        </div>

        {/* RIGHT: Weather + Mandi + Quick Actions */}
        <div className="dash-r">
          {/* Weather Widget */}
          <div className="weather-card">
            <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 11
          }}>
              <div style={{
              fontSize: 15,
              fontWeight: 800,
              color: 'var(--g1)'
            }}>🌤️ {tx(i18n, 'todaysWeather')}</div>
              <div style={{
              fontSize: 11,
              color: 'var(--tx3)',
              fontWeight: 600
            }}>📍 {user?.district || 'Pune'}, MH</div>
            </div>
            <div className="wt-main">
              <div>
                <div className="wt-temp">{N(28)}°C</div>
                <div style={{
                fontSize: 13,
                color: 'var(--tx2)',
                marginTop: 2
              }}>{i18n.language?.startsWith('hi') ? '🌤️ आंशिक बादल' : i18n.language?.startsWith('mr') ? '🌤️ अंशतः ढगाळ' : '🌤️ Partly Cloudy'}</div>
              </div>
              <div style={{
              textAlign: 'right'
            }}>
                <span className="wt-risk risk-med">{i18n.language?.startsWith('hi') ? '🟡 मध्यम जोखिम' : i18n.language?.startsWith('mr') ? '🟡 मध्यम जोखीम' : '🟡 Medium Risk'}</span>
                <div style={{
                fontSize: 12,
                color: 'var(--tx2)',
                marginTop: 5
              }}>{i18n.language?.startsWith('hi') ? 'नमी' : i18n.language?.startsWith('mr') ? 'आर्द्रता' : 'Humidity'}: {P(68)}</div>
                <div style={{
                fontSize: 12,
                color: 'var(--tx3)',
                marginTop: 2
              }}>{i18n.language?.startsWith('hi') ? 'हवा' : i18n.language?.startsWith('mr') ? 'वारा' : 'Wind'}: {N(12)} km/h</div>
              </div>
            </div>
            <div style={{
            display: 'flex',
            gap: 8,
            marginBottom: 10
          }}>
              {[{
              d: i18n.language?.startsWith('hi') ? 'कल' : i18n.language?.startsWith('mr') ? 'उद्या' : 'Tomorrow',
              t: `${N(26)}°C`,
              i: '🌧️'
            }, {
              d: i18n.language?.startsWith('hi') ? 'परसों' : i18n.language?.startsWith('mr') ? 'परवा' : 'Day 2',
              t: `${N(24)}°C`,
              i: '🌧️'
            }, {
              d: i18n.language?.startsWith('hi') ? '3 दिन' : i18n.language?.startsWith('mr') ? '3 दिवस' : '3 days',
              t: `${N(29)}°C`,
              i: '⛅'
            }].map(w => <div key={w.d} style={{
              flex: 1,
              textAlign: 'center',
              padding: '7px 4px',
              background: 'rgba(255,255,255,.6)',
              borderRadius: 8
            }}>
                  <div style={{
                fontSize: 14
              }}>{w.i}</div>
                  <div style={{
                fontSize: 10.5,
                fontWeight: 700,
                color: 'var(--tx2)',
                marginTop: 2
              }}>{w.d}</div>
                  <div style={{
                fontSize: 11,
                color: 'var(--tx3)'
              }}>{w.t}</div>
                </div>)}
            </div>
            <div style={{
            padding: '9px 13px',
            background: 'rgba(240,165,0,.12)',
            borderRadius: 8,
            fontSize: 12.5,
            color: 'var(--a1)',
            fontWeight: 600
          }}>
              ⚠️ {i18n.language?.startsWith('hi') ? 'अगले 2 दिन बारिश - स्प्रे रोकें' : i18n.language?.startsWith('mr') ? 'पुढील 2 दिवस पाऊस - फवारणी थांबवा' : 'Rain for next 2 days - avoid spraying'}
            </div>
          </div>

          {/* Mandi Prices */}
          <div className="card" style={{
          padding: 18
        }}>
            <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 13
          }}>
              <div style={{
              fontSize: 15,
              fontWeight: 800,
              color: 'var(--g1)'
            }}>📈 {tx(i18n, 'mandiPrices')}</div>
              <span style={{
              fontSize: 11,
              color: 'var(--tx3)',
              fontWeight: 600
            }}>{tx(i18n, 'live')}</span>
            </div>
            {[{
            nm: `🍅 ${cropName('tomato', i18n)}`,
            pr: `${C(1240)}/qtl`,
            ch: `+${P(12)}`,
            d: 'up',
            mspr: `${tx(i18n, 'msp')}: N/A`
          }, {
            nm: `🌾 ${cropName('wheat', i18n)}`,
            pr: `${C(2180)}/qtl`,
            ch: tx(i18n, 'stable'),
            d: 'st',
            mspr: `${tx(i18n, 'msp')}: ${C(2275)}`
          }, {
            nm: `🥔 ${cropName('potato', i18n)}`,
            pr: `${C(780)}/qtl`,
            ch: `-${P(3)}`,
            d: 'dn',
            mspr: `${tx(i18n, 'msp')}: N/A`
          }].map(m => <div key={m.nm} className="mandi-row">
                <div>
                  <div className="mandi-crop">{m.nm}</div>
                  <div style={{
                fontSize: 11,
                color: 'var(--tx3)'
              }}>{m.mspr}</div>
                </div>
                <div style={{
              textAlign: 'right'
            }}>
                  <div className="mandi-price">{m.pr}</div>
                  <div className={`mandi-ch ${m.d === 'up' ? 'ch-up' : m.d === 'dn' ? 'ch-dn' : 'ch-st'}`}>{m.d === 'up' ? '↑' : m.d === 'dn' ? '↓' : '→'} {m.ch}</div>
                </div>
              </div>)}
            <div style={{
            marginTop: 11,
            padding: '9px 13px',
            background: 'var(--gp)',
            borderRadius: 8,
            fontSize: 12.5,
            color: 'var(--g2)',
            fontWeight: 700
          }}>
              🍅 {cropName('tomato', i18n)} {tx(i18n, 'priceUp')} ↑ {P(18)} — {tx(i18n, 'sellTime')}!
            </div>
          </div>

          {/* Quick Stats */}
          <div className="card" style={{
          padding: 18
        }}>
            <div style={{
            fontSize: 15,
            fontWeight: 800,
            color: 'var(--g1)',
            marginBottom: 13
          }}>💰 {tx(i18n, 'seasonCost')}</div>
            {[[`🌱 ${tx(i18n, 'seeds')}`, C(4500)], [`🧪 ${tx(i18n, 'fertilizer')}`, C(6200)], [`💊 ${tx(i18n, 'medicines')}`, C(1840)], [`👨‍⚕️ ${tx(i18n, 'consultations')}`, C(984)]].map(([l, v]) => <div key={l} style={{
            display: 'flex',
            justifyContent: 'space-between',
            padding: '7px 0',
            borderBottom: '1px solid var(--gp)',
            fontSize: 13
          }}>
                <span style={{
              color: 'var(--tx2)',
              fontWeight: 600
            }}>{l}</span>
                <span style={{
              fontWeight: 800,
              color: 'var(--tx)'
            }}>{v}</span>
              </div>)}
            <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            paddingTop: 10,
            fontSize: 14
          }}>
              <span style={{
              fontWeight: 700,
              color: 'var(--tx)'
            }}>{tx(i18n, 'totalCost')}</span>
              <span style={{
              fontWeight: 900,
              color: 'var(--r2)'
            }}>{C(13524)}</span>
            </div>
            <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: 13
          }}>
              <span style={{
              fontWeight: 600,
              color: 'var(--tx3)'
            }}>{tx(i18n, 'expectedYield')}</span>
              <span style={{
              fontWeight: 800,
              color: 'var(--g3)'
            }}>{C(85000)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   EXPERT DASHBOARD
════════════════════════════════════════════════════════════════ */
/* ════════════════════════════════════════════════════════════════
   EXPERTS PAGE
════════════════════════════════════════════════════════════════ */
