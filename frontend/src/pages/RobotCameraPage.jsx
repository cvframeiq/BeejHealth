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
import HomePage from './HomePage.jsx';
import ExpertOnboarding from '../components/ExpertOnboarding.jsx';
import FarmerOnboarding from '../components/FarmerOnboarding.jsx';
import AuthModal from '../components/AuthModal.jsx';

export default /* ════════════════════════════════════════════════════════════════
   LIVE CAMERA FEED 📡
════════════════════════════════════════════════════════════════ */
function RobotCameraPage({
  nav,
  toast
}) {
  const [activeCam, setActiveCam] = useState('R01');
  const [zoom, setZoom] = useState(1);
  const [detect, setDetect] = useState(true);
  const [tick, setTick] = useState(0);
  const [cameraFeeds, setCameraFeeds] = useState([]);
  const [cameraInfo, setCameraInfo] = useState(null);
  const [detections, setDetections] = useState([{
    x: 35,
    y: 25,
    w: 15,
    h: 18,
    label: 'Early Blight',
    conf: 94,
    color: '#ff4444'
  }, {
    x: 62,
    y: 48,
    w: 12,
    h: 14,
    label: 'Healthy Leaf',
    conf: 98,
    color: 'var(--g4)'
  }, {
    x: 18,
    y: 60,
    w: 10,
    h: 12,
    label: 'Pest Damage',
    conf: 79,
    color: '#ffd700'
  }]);
  const [gpsInfo, setGpsInfo] = useState({
    lat: '18.591°N',
    lng: '73.741°E',
    alt: '12m',
    speed: '3.2m/s'
  });
  useEffect(() => {
    // Load all camera feeds
    API.get('/api/camera/all').then(d => {
      if (d.feeds && d.feeds.length > 0) setCameraFeeds(d.feeds);
    }).catch(() => {});
    // Load active camera info
    API.get('/api/robots/R01/camera').then(d => {
      if (d) setCameraInfo(d);
      if (d.detections && d.detections.length > 0) setDetections(d.detections);
      if (d.gps) setGpsInfo({
        lat: `${d.gps.lat.toFixed(3)}°N`,
        lng: `${d.gps.lng.toFixed(3)}°E`,
        alt: d.altitude,
        speed: d.speed
      });
    }).catch(() => {});
  }, []);
  useEffect(() => {
    if (!activeCam) return;
    API.get(`/api/robots/${activeCam}/camera`).then(d => {
      if (d.detections) setDetections(d.detections);
      if (d.gps) setGpsInfo({
        lat: `${d.gps.lat.toFixed(3)}°N`,
        lng: `${d.gps.lng.toFixed(3)}°E`,
        alt: d.altitude,
        speed: d.speed
      });
    }).catch(() => {});
  }, [activeCam]);

  // Poll camera info every 5s for live updates
  useEffect(() => {
    const iv = setInterval(() => {
      API.get(`/api/robots/${activeCam}/camera`).then(d => {
        if (d.detections) setDetections(d.detections);
        if (d.gps) setGpsInfo({
          lat: `${d.gps.lat.toFixed(3)}°N`,
          lng: `${d.gps.lng.toFixed(3)}°E`,
          alt: d.altitude,
          speed: d.speed
        });
      }).catch(() => {});
    }, 5000);
    return () => clearInterval(iv);
  }, [activeCam]);
  useEffect(() => {
    const t = setInterval(() => setTick(p => p + 1), 1500);
    return () => clearInterval(t);
  }, []);
  const cameras = cameraFeeds.length > 0 ? cameraFeeds.map(f => ({
    id: f.robotId,
    name: f.robotName,
    type: f.primaryCam,
    status: f.isLive ? 'live' : 'offline',
    field: f.field,
    emoji: f.emoji
  })) : [{
    id: 'R01',
    name: 'DroneBot Alpha',
    type: 'Drone Cam',
    status: 'live',
    field: 'Field 1',
    emoji: '🚁'
  }, {
    id: 'R02',
    name: 'GroundBot Beta',
    type: 'Front Cam',
    status: 'live',
    field: 'Field 2',
    emoji: '🤖'
  }, {
    id: 'R04',
    name: 'SensorBot Delta',
    type: 'Wide Cam',
    status: 'live',
    field: 'All Fields',
    emoji: '📡'
  }];
  const takeSnapshot = async () => {
    try {
      const res = await API.post(`/api/robots/${activeCam}/camera/snapshot`, {
        cameraId: 'front'
      });
      toast(`Screenshot saved! ID: ${res.snapId} 📸`);
    } catch (e) {
      toast('Screenshot save nahi hua', 'err');
    }
  };
  return <div className="rob-shell">
      <div className="rob-wrap">
        <div style={{
        display: 'flex',
        gap: 10,
        alignItems: 'center',
        marginBottom: 22,
        flexWrap: 'wrap'
      }}>
          <button className="rob-btn ghost" onClick={() => nav('robot-dashboard')}>← Back</button>
          <div style={{
          fontFamily: "'Baloo 2',cursive",
          fontSize: 24,
          fontWeight: 900,
          color: 'var(--g3)'
        }}>📡 Live Camera Feed</div>
          <span className="rob-badge online" style={{
          marginLeft: 8
        }}><span className="rob-dot online" />LIVE</span>
        </div>

        <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 280px',
        gap: 18
      }}>
          {/* Main Feed */}
          <div>
            <div className="cam-feed" style={{
            height: 320
          }}>
              {/* Simulated camera view */}
              <div style={{
              position: 'absolute',
              inset: 0,
              background: `linear-gradient(${135 + tick * 2}deg,#0d2b0d,#1a3a1a,#0d2b1a)`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
                <div style={{
                fontSize: 80,
                opacity: .15,
                filter: 'blur(2px)'
              }}>🌿</div>
              </div>
              <div className="cam-scanline" />
              <div className="cam-overlay-tl" /><div className="cam-overlay-tr" />
              <div className="cam-overlay-bl" /><div className="cam-overlay-br" />
              <div className="cam-rec"><span className="cam-rec-dot" />REC • {cameras.find(c => c.id === activeCam)?.name}</div>

              {/* AI Detection Boxes */}
              {detect && detections.map((d, i) => <div key={i} style={{
              position: 'absolute',
              left: `${d.x}%`,
              top: `${d.y}%`,
              width: `${d.w}%`,
              height: `${d.h}%`,
              border: `2px solid ${d.color}`,
              borderRadius: 4,
              zIndex: 6
            }}>
                  <div style={{
                position: 'absolute',
                top: -20,
                left: 0,
                background: d.color + 'cc',
                padding: '2px 6px',
                borderRadius: 4,
                fontSize: 10,
                fontWeight: 700,
                color: d.color === '#ff4444' ? 'white' : '#0a0f1e',
                whiteSpace: 'nowrap'
              }}>
                    {d.label} {d.conf}%
                  </div>
                </div>)}

              {/* HUD Overlay */}
              <div style={{
              position: 'absolute',
              bottom: 12,
              left: 12,
              right: 12,
              display: 'flex',
              justifyContent: 'space-between',
              zIndex: 7
            }}>
                <div style={{
                fontSize: 11,
                color: 'var(--g3)',
                fontFamily: 'monospace',
                background: 'rgba(0,0,0,.55)',
                padding: '4px 8px',
                borderRadius: 6
              }}>
                  ALT: {gpsInfo.alt} | SPD: {gpsInfo.speed} | GPS: {gpsInfo.lat}
                </div>
                <div style={{
                fontSize: 11,
                color: 'var(--g4)',
                fontFamily: 'monospace',
                background: 'rgba(0,0,0,.55)',
                padding: '4px 8px',
                borderRadius: 6
              }}>
                  {new Date().toLocaleTimeString()}
                </div>
              </div>
            </div>

            {/* Controls */}
            <div style={{
            display: 'flex',
            gap: 10,
            marginTop: 12,
            flexWrap: 'wrap'
          }}>
              <button className={`rob-btn ${detect ? 'primary' : 'ghost'}`} onClick={() => setDetect(v => !v)}>
                🎯 AI Detection {detect ? 'ON' : 'OFF'}
              </button>
              <button className="rob-btn ghost" onClick={() => setZoom(v => Math.min(v + 0.5, 4))}>🔍 Zoom In ({zoom}x)</button>
              <button className="rob-btn ghost" onClick={() => setZoom(v => Math.max(v - 0.5, 1))}>🔎 Zoom Out</button>
              <button className="rob-btn ghost" onClick={takeSnapshot}>📸 Screenshot</button>
              <button className="rob-btn ghost" onClick={() => toast('Recording started 🔴')}>⏺️ Record</button>
            </div>

            {/* Detection Log */}
            {detect && <div className="rob-card" style={{
            padding: 16,
            marginTop: 14
          }}>
                <div style={{
              fontSize: 13,
              fontWeight: 700,
              color: 'rgba(0,0,0,.9)',
              marginBottom: 11,
              textTransform: 'uppercase',
              letterSpacing: '.6px'
            }}>AI Detections ({detections.length})</div>
                {detections.map((d, i) => <div key={i} style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '8px 0',
              borderBottom: '1px solid rgba(0,0,0,.8)'
            }}>
                    <div style={{
                display: 'flex',
                gap: 9,
                alignItems: 'center'
              }}>
                      <div style={{
                  width: 10,
                  height: 10,
                  borderRadius: 2,
                  background: d.color,
                  flexShrink: 0
                }} />
                      <span style={{
                  fontSize: 13,
                  color: 'var(--tx)',
                  fontWeight: 600
                }}>{d.label}</span>
                    </div>
                    <div style={{
                display: 'flex',
                gap: 10,
                alignItems: 'center'
              }}>
                      <span style={{
                  fontSize: 13,
                  fontWeight: 800,
                  color: d.color
                }}>{d.conf}%</span>
                      {d.label !== 'Healthy Leaf' && <button className="rob-btn ghost" style={{
                  padding: '4px 10px',
                  fontSize: 11
                }} onClick={() => nav('robot-spray')}>Spray →</button>}
                    </div>
                  </div>)}
              </div>}
          </div>

          {/* Camera List */}
          <div>
            <div style={{
            fontSize: 13,
            fontWeight: 700,
            color: 'rgba(0,0,0,.7)',
            marginBottom: 12,
            textTransform: 'uppercase',
            letterSpacing: '.7px'
          }}>All Cameras</div>
            {cameras.map(c => <div key={c.id} className={`robot-row${activeCam === c.id ? ' sel' : ''}`} onClick={() => setActiveCam(c.id)} style={{
            marginBottom: 8,
            cursor: 'pointer'
          }}>
                <div style={{
              fontSize: 22
            }}>{c.emoji}</div>
                <div style={{
              flex: 1
            }}>
                  <div style={{
                fontSize: 13,
                fontWeight: 700,
                color: 'var(--tx)'
              }}>{c.name}</div>
                  <div style={{
                fontSize: 11,
                color: 'rgba(0,0,0,.6)'
              }}>{c.type} • {c.field}</div>
                  <span className="rob-badge online" style={{
                marginTop: 5,
                fontSize: 10
              }}><span className="rob-dot online" />LIVE</span>
                </div>
              </div>)}

            {/* Quick Stats from active camera */}
            <div className="rob-card" style={{
            padding: 16,
            marginTop: 14
          }}>
              <div style={{
              fontSize: 12,
              fontWeight: 700,
              color: 'rgba(0,0,0,.9)',
              marginBottom: 11,
              textTransform: 'uppercase',
              letterSpacing: '.6px'
            }}>Stream Info</div>
              {[['Resolution', '4K / 30fps'], ['Bitrate', '12 Mbps'], ['Latency', '~220ms'], ['Storage', '128GB (68% free)']].map(([l, v]) => <div key={l} style={{
              display: 'flex',
              justifyContent: 'space-between',
              padding: '6px 0',
              borderBottom: '1px solid rgba(0,0,0,.7)',
              fontSize: 12
            }}>
                  <span style={{
                color: 'rgba(0,0,0,.6)'
              }}>{l}</span>
                  <span style={{
                color: 'rgba(0,0,0,.9)',
                fontWeight: 700
              }}>{v}</span>
                </div>)}
            </div>
          </div>
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   AUTO FIELD NAVIGATION & MAPPING 🗺️
════════════════════════════════════════════════════════════════ */
