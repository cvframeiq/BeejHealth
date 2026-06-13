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
   3. PREDICTIVE DISEASE FORECAST 📊
════════════════════════════════════════════════════════════════ */
function ForecastPage({
  user,
  nav,
  toast
}) {
  const [weather, setWeather] = useState(null);
  const [loadingWx, setLoadingWx] = useState(true);
  const [wxErr, setWxErr] = useState(false);
  useEffect(() => {
    const district = user?.district || 'Pune';
    // OpenWeatherMap free tier API
    const API_KEY = '4d8fb5b93d4af21d66a2948a6e8a74a0'; // Demo key — replace with yours
    fetch(`https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(district)},IN&appid=${API_KEY}&units=metric&lang=hi`).then(r => r.json()).then(d => {
      if (d.cod === 200) {
        setWeather({
          temp: Math.round(d.main.temp),
          feels: Math.round(d.main.feels_like),
          humidity: d.main.humidity,
          desc: d.weather[0].description,
          wind: Math.round(d.wind.speed * 3.6),
          // m/s to km/h
          city: d.name,
          icon: d.weather[0].icon,
          pressure: d.main.pressure
        });
      } else {
        setWxErr(true);
      }
      setLoadingWx(false);
    }).catch(() => {
      setWxErr(true);
      setLoadingWx(false);
    });
  }, [user?.district]);
  const forecasts = [{
    disease: 'Late Blight',
    crop: '🥔 Potato',
    risk: 78,
    level: 'high',
    days: 5,
    reason: 'High humidity (72%) + cool temp expected',
    action: 'Preventive spray karein aaj'
  }, {
    disease: 'Powdery Mildew',
    crop: '🍇 Grape',
    risk: 52,
    level: 'med',
    days: 8,
    reason: 'Dry warm weather forecast',
    action: 'Monitor karein — spray ready rakhein'
  }, {
    disease: 'Leaf Rust',
    crop: '🌾 Wheat',
    risk: 31,
    level: 'low',
    days: 14,
    reason: 'Conditions unfavorable currently',
    action: 'No action needed abhi'
  }, {
    disease: 'Early Blight',
    crop: '🍅 Tomato',
    risk: 67,
    level: 'med',
    days: 6,
    reason: 'Humidity spike + rain forecast',
    action: 'Copper-based fungicide consider karein'
  }];
  const timeline = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const riskData = [22, 35, 48, 61, 78, 72, 55];
  return <div className="wrap-md">
      {/* Live Weather from OpenWeatherMap */}
      {loadingWx && <div style={{
      textAlign: 'center',
      padding: '16px',
      fontSize: 13,
      color: 'var(--tx3)'
    }}>🌐 Live mausam load ho raha hai...</div>}
      {!loadingWx && !wxErr && weather && <div className="card" style={{
      padding: 20,
      marginBottom: 20,
      background: 'linear-gradient(135deg,var(--bp),var(--bpb))',
      border: '1.5px solid var(--bpb)'
    }}>
          <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 10
      }}>
            <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 16,
          fontWeight: 800,
          color: 'var(--b1)'
        }}>🌤️ Live Mausam — {weather.city}</div>
            <div style={{
          fontSize: 10,
          color: 'var(--tx3)'
        }}>OpenWeatherMap Live</div>
          </div>
          <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4,1fr)',
        gap: 8
      }}>
            {[['🌡️', 'Temp', weather.temp + '°C'], ['💧', 'Humidity', weather.humidity + '%'], ['💨', 'Wind', weather.wind + ' km/h'], ['📊', 'Pressure', weather.pressure + ' hPa']].map(([ic, l, v]) => <div key={l} style={{
          background: 'rgba(255,255,255,.6)',
          borderRadius: 8,
          padding: '10px 8px',
          textAlign: 'center'
        }}>
                <div style={{
            fontSize: 18,
            marginBottom: 3
          }}>{ic}</div>
                <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 15,
            fontWeight: 800,
            color: 'var(--b1)'
          }}>{v}</div>
                <div style={{
            fontSize: 10,
            color: 'var(--tx3)'
          }}>{l}</div>
              </div>)}
          </div>
          <div style={{
        marginTop: 8,
        fontSize: 12,
        color: 'var(--b2)',
        fontWeight: 600,
        textTransform: 'capitalize'
      }}>Aaj: {weather.desc} · Feels like {weather.feels}°C</div>
        </div>}
      {!loadingWx && wxErr && <div style={{
      background: 'var(--ap)',
      borderRadius: 8,
      padding: '10px 14px',
      marginBottom: 12,
      fontSize: 12,
      color: 'var(--a1)'
    }}>⚠️ Live weather load nahi hua — internet ya API key check karein</div>}

      <div style={{
      fontFamily: "'Baloo 2',cursive",
      fontSize: 26,
      fontWeight: 900,
      color: 'var(--g1)',
      marginBottom: 5
    }}>📊 Disease Forecast</div>
      <div style={{
      fontSize: 13,
      color: 'var(--tx2)',
      marginBottom: 22
    }}>AI + IMD Weather Data • {user?.district || 'Pune'}, Maharashtra</div>

      {/* 7-Day Risk Timeline */}
      <div className="card" style={{
      padding: 22,
      marginBottom: 22
    }}>
        <div style={{
        fontSize: 15,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 16
      }}>📅 7-Day Disease Risk Timeline</div>
        <div style={{
        display: 'flex',
        gap: 0,
        alignItems: 'flex-end',
        height: 100,
        marginBottom: 8
      }}>
          {riskData.map((r, i) => <div key={i} style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 4
        }}>
              <div style={{
            fontSize: 10,
            fontWeight: 700,
            color: r > 60 ? 'var(--r2)' : r > 40 ? 'var(--a2)' : 'var(--g4)'
          }}>{r}%</div>
              <div style={{
            width: '70%',
            borderRadius: '4px 4px 0 0',
            background: r > 60 ? 'var(--r3)' : r > 40 ? 'var(--a3)' : 'var(--g5)',
            transition: 'height .7s ease',
            height: `${r}%`,
            minHeight: 6
          }} />
            </div>)}
        </div>
        <div style={{
        display: 'flex',
        gap: 0
      }}>
          {timeline.map((d, i) => <div key={d} style={{
          flex: 1,
          textAlign: 'center',
          fontSize: 11,
          fontWeight: 700,
          color: i === 4 ? 'var(--r2)' : 'var(--tx3)',
          paddingTop: 6,
          borderTop: `2px solid ${i === 4 ? 'var(--r2)' : 'var(--br)'}`
        }}>{d}</div>)}
        </div>
        <div style={{
        marginTop: 12,
        padding: '9px 13px',
        background: 'var(--rp)',
        borderRadius: 8,
        fontSize: 13,
        color: 'var(--r2)',
        fontWeight: 700
      }}>
          🔴 Peak Risk: Friday — Late Blight probability 78%
        </div>
      </div>

      {/* Disease Forecasts */}
      <div style={{
      fontSize: 16,
      fontWeight: 800,
      color: 'var(--g1)',
      marginBottom: 14
    }}>🦠 Crop-wise Forecast</div>
      {forecasts.map((f, i) => <div className="forecast-card" key={i}>
          <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: 10
      }}>
            <div>
              <div style={{
            fontSize: 15,
            fontWeight: 800,
            color: 'var(--tx)'
          }}>{f.crop} — {f.disease}</div>
              <div style={{
            fontSize: 12,
            color: 'var(--tx3)',
            marginTop: 2
          }}>⏱️ Peak risk in {f.days} days</div>
            </div>
            <div style={{
          textAlign: 'right'
        }}>
              <div style={{
            fontFamily: "'Baloo 2',cursive",
            fontSize: 22,
            fontWeight: 900,
            color: f.level === 'high' ? 'var(--r2)' : f.level === 'med' ? 'var(--a2)' : 'var(--g4)'
          }}>{f.risk}%</div>
              <span className={`badge ${f.level === 'high' ? 'bg-r' : f.level === 'med' ? 'bg-a' : 'bg-g'}`} style={{
            fontSize: 10
          }}>{f.level === 'high' ? '🔴 High' : f.level === 'med' ? '🟡 Medium' : '🟢 Low'}</span>
            </div>
          </div>
          <div className="risk-meter">
            <div className={`risk-fill ${f.level}`} style={{
          width: `${f.risk}%`
        }} />
          </div>
          <div style={{
        fontSize: 12.5,
        color: 'var(--tx2)',
        margin: '8px 0',
        fontStyle: 'italic'
      }}>📌 {f.reason}</div>
          <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
            <div style={{
          fontSize: 13,
          fontWeight: 700,
          color: f.level === 'high' ? 'var(--r2)' : f.level === 'med' ? 'var(--a1)' : 'var(--g3)'
        }}>💡 {f.action}</div>
            {f.level !== 'low' && <button className="btn btn-g btn-sm" onClick={() => nav('marketplace')}>🛒 Medicine Order</button>}
          </div>
        </div>)}

      {/* AI Insight */}
      <div style={{
      padding: 18,
      background: 'linear-gradient(135deg,var(--gp),var(--gpb))',
      borderRadius: 'var(--rad)',
      border: '1.5px solid var(--br2)',
      marginTop: 8
    }}>
        <div style={{
        fontSize: 14,
        fontWeight: 800,
        color: 'var(--g1)',
        marginBottom: 7
      }}>🤖 AI Insight</div>
        <div style={{
        fontSize: 13.5,
        color: 'var(--tx2)',
        lineHeight: 1.7
      }}>
          Is hafta weather pattern ke hisaab se <strong>fungal diseases ka risk 40% zyada</strong> hai normal se. Humidity 68-75% range mein hai aur temperature drop forecast hai — yeh Late Blight ke liye ideal conditions hain. Preventive action Thursday tak le lein.
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   4. IoT SOIL SENSORS 🌱
════════════════════════════════════════════════════════════════ */
