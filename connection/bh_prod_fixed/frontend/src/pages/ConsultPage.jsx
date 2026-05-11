import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import {
  API,
  MAX_REPORT_PHOTOS,
  MAX_UPLOAD_DIMENSION,
  MAX_UPLOAD_SOURCE_BYTES,
  MIN_REPORT_PHOTOS,
  UPLOAD_JPEG_QUALITY,
} from '../services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody, fileToUploadDataUrl, optimizeImageSrcToJpeg, uploadPhotoAsset } from '../utils/helpers.jsx';
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary, COCONUT_DISEASE_QS, COCONUT_Q1_DATA, Q1_DATA, BRANCH_QS } from '../utils/constants.jsx';

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
import HomePage from './HomePage.jsx';
import ExpertOnboarding from '../components/ExpertOnboarding.jsx';
import FarmerOnboarding from '../components/FarmerOnboarding.jsx';
import AuthModal from '../components/AuthModal.jsx';

export default function ConsultPage({
  user,
  nav,
  toast,
  selCrop,
  setSelCrop,
  qAnswers,
  setQAnswers
}) {
  const [step, setStep] = useState(1); // 1=crop 2=upload 3=questions 4=processing
  const [photos, setPhotos] = useState([]); // [{preview, photoId, uploading, error, sizeKB}]
  const [uploadErrors, setUploadErrors] = useState([]);
  const [uploadDebug, setUploadDebug] = useState('Idle');
  const readyPhotos = photos.filter(p => p.photoId && !p.error);
  const readyPhotoCount = readyPhotos.length;
  const uploadDone = readyPhotoCount > 0 && !photos.some(p => p.uploading);
  const [preAiResult, setPreAiResult] = useState(null); // Quick AI scan after photo upload
  const [aiScanning, setAiScanning] = useState(false); // AI scan in progress
  // Camera state
  const [showCamera, setShowCamera] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [replacePhotoIndex, setReplacePhotoIndex] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const uploadInputRef = useRef(null);
  const replacePhotoIndexRef = useRef(null);
  const [q1Answer, setQ1Answer] = useState(null);
  const [activeQs, setActiveQs] = useState([]);
  const isCoconut = selCrop?.id === 'coconut';
  const currentQ1 = isCoconut ? COCONUT_Q1_DATA : Q1_DATA;
  const [qIndex, setQIndex] = useState(0);
  const [currAns, setCurrAns] = useState(null);
  const [answeredList, setAnsweredList] = useState([]);
  const [procStep, setProcStep] = useState(0);
  const PROC_STEPS = ['Image quality check...', 'Disease patterns scan...', 'Severity calculate ho rahi hai...', 'Treatment plan generate ho raha hai...'];
  const startProcessing = async () => {
    if (readyPhotoCount < MIN_REPORT_PHOTOS) {
      toast(`Kam se kam ${MIN_REPORT_PHOTOS} photos upload karein`, 'warn');
      setStep(2);
      return;
    }
    setStep(4);
    setProcStep(0);
    for (let i = 0; i < 4; i++) {
      await new Promise(r => setTimeout(r, 680));
      setProcStep(i + 1);
    }
    if (user) {
      try {
        const isCoconutCrop = selCrop?.id === 'coconut';
        const primaryPhoto = readyPhotos[0] || photos[0] || null;
        const photoBase64 = primaryPhoto?.preview || null;
        const photoId0 = primaryPhoto?.photoId || null;
        const photoIds = readyPhotos.map(p => p.photoId);
        const photoUrls = readyPhotos.map(p => p.url).filter(Boolean);
        const photoCount = readyPhotoCount;
        if (isCoconutCrop && (photoBase64 || photoId0)) {
          // ── COCONUT: Use real AI model ────────────────────────────
          const res = await API.post('/api/coconut/analyze', {
            photoBase64: photoBase64,
            photoId: photoId0,
            photoIds,
            photoUrls,
            photoCount,
            questionAnswers: qAnswers,
            preAiDisease: preAiResult?.disease || null // pass pre-scan result
          }, 60000);
          if (res?.consultation_id) {
            localStorage.setItem('bh_latest_consult', res.consultation_id);
            localStorage.setItem('bh_latest_crop', JSON.stringify({
              id: 'coconut',
              name: 'Naariyal',
              emoji: '🥥'
            }));
            rememberConsultationContext(res.consultation_id);
            // Store AI result for immediate display
            localStorage.setItem('bh_ai_result', JSON.stringify(res.ai_result));
          }
        } else {
          // ── OTHER CROPS: Existing flow ────────────────────────────
          const d = DISEASE_DB[selCrop?.id] || DISEASE_DB.default;
          const res = await API.post('/api/consultations', {
            cropId: selCrop?.id || 'tomato',
            cropName: selCrop?.name || 'Crop',
            cropEmoji: selCrop?.emoji || '🌱',
            method: readyPhotoCount >= MIN_REPORT_PHOTOS ? 'photo' : 'manual',
            photoUploaded: readyPhotoCount >= MIN_REPORT_PHOTOS,
            photoId: photoId0,
            photoUrl: primaryPhoto?.url || null,
            photoIds,
            photoUrls,
            photoCount,
            answers: qAnswers,
            disease: d.disease,
            confidence: d.conf,
            severity: d.sev
          });
          if (res?.consultation?._id) {
            localStorage.setItem('bh_latest_consult', res.consultation._id);
            localStorage.setItem('bh_latest_crop', JSON.stringify({
              id: selCrop?.id,
              name: selCrop?.name,
              emoji: selCrop?.emoji
            }));
            rememberConsultationContext(res.consultation._id);
          }
        }
      } catch (e) {
        if (e.message?.includes('AI_SERVER_DOWN') || e.message?.includes('AI server band') || e.message?.includes('ai_down')) {
          toast('⚠️ AI server band hai — Photo save ho gayi, baad mein analyze hogi', 'warn');
        } else {
          console.warn('Consultation save:', e.message);
        }
      }
    }
    nav('ai-report');
  };
  const handleSelectCrop = c => {
    setSelCrop(c);
  };
  const handleNextCrop = () => {
    if (!selCrop) {
      toast('Pehle crop select karein', 'warn');
      return;
    }
    setStep(2);
  };
  const openGallery = (replaceIndex = null) => {
    setUploadDebug(`Opening picker${replaceIndex !== null ? ` for replace #${replaceIndex + 1}` : ''}...`);
    replacePhotoIndexRef.current = replaceIndex;
    setReplacePhotoIndex(replaceIndex);
    const input = uploadInputRef.current;
    if (!input) {
      toast('File upload input missing hai. Page refresh karke dobara try karein.', 'err');
      setUploadDebug('File input missing');
      return;
    }
    input.multiple = replaceIndex === null;
    input.value = '';
    try {
      input.click();
    } catch (err) {
      console.error('File picker open failed:', err);
      toast('File picker open nahi ho raha. Browser se dobara try karein.', 'err');
      setUploadDebug(`File picker open failed: ${err.message}`);
      replacePhotoIndexRef.current = null;
      setReplacePhotoIndex(null);
    }
  };
  const handleFileChange = async (fileList, options = {}) => {
    if (!fileList || fileList.length === 0) {
      setUploadDebug('No file selected');
      return;
    }
    const replaceIndex = Number.isInteger(options.replaceIndex) ? options.replaceIndex : null;
    const files = Array.from(fileList);
    setUploadDebug(`Selected ${files.length} file(s): ${files.map(f => f.name || 'unnamed').join(', ')}`);
    const existingCount = photos.length;
    const availableSlots = replaceIndex !== null ? 1 : MAX_REPORT_PHOTOS - existingCount;
    if (availableSlots <= 0) {
      toast(`Maximum ${MAX_REPORT_PHOTOS} photos allowed`, 'warn');
      return;
    }
    const toProcess = replaceIndex !== null ? files.slice(0, 1) : files.slice(0, availableSlots);
    if (replaceIndex === null && files.length > availableSlots) toast(`Sirf ${availableSlots} aur photo add ho sakti hain (max ${MAX_REPORT_PHOTOS})`, 'warn');
    if (replaceIndex !== null && files.length > 1) toast('Replace ke liye sirf 1 image select karein', 'warn');
    const errors = [];
    setUploadErrors([]);
    let inserted = 0;
    for (const file of toProcess) {
      setUploadDebug(`Processing ${file.name || 'image'} (${Math.round((file.size || 0) / 1024)}KB)...`);
      if (file.size > MAX_UPLOAD_SOURCE_BYTES) {
        errors.push(`${file.name}: 25MB se zyada`);
        continue;
      }
      if (!isLikelyImageFile(file)) {
        errors.push(`${file.name}: Image nahi hai`);
        continue;
      }
      let uploadAsset;
      try {
        uploadAsset = await fileToUploadDataUrl(file);
      } catch (err) {
        const msg = `${file.name}: ${err.message}`;
        errors.push(msg);
        toast(msg, 'err');
        setUploadDebug(msg);
        continue;
      }
      const base64 = uploadAsset.dataUrl;
      const contentType = uploadAsset.contentType || file.type || 'image/jpeg';
      const targetIndex = replaceIndex !== null ? replaceIndex : existingCount + inserted;
      const tempId = `${replaceIndex !== null ? 'repl' : 'temp'}_${Date.now()}_${targetIndex}`;
      const sizeKB = Math.round(base64.length * 3 / 4 / 1024);
      const consultationId = null;
      if (replaceIndex !== null) {
        const oldPhoto = photos[targetIndex];
        if (oldPhoto?.photoId) {
          await API.delete('/api/photos/' + oldPhoto.photoId).catch(() => {});
        }
        setPhotos(prev => prev.map((p, i) => i === targetIndex ? {
          tempId,
          preview: base64,
          uploading: true,
          error: null,
          sizeKB,
          contentType
        } : p));
      } else {
        setPhotos(prev => [...prev, {
          tempId,
          preview: base64,
          uploading: true,
          error: null,
          sizeKB,
          contentType
        }]);
      }
      try {
        setUploadDebug(`Uploading ${file.name || 'image'} to server...`);
        const res = await uploadPhotoAsset({
          base64,
          type: contentType,
          index: targetIndex,
          totalExpected: replaceIndex !== null ? 1 : toProcess.length,
          consultationId
        });
        setPhotos(prev => prev.map((p, i) => i === targetIndex ? {
          ...p,
          uploading: false,
          photoId: res.photoId,
          url: res.url,
          sizeKB: res.sizeKB || p.sizeKB
        } : p));
        setUploadDebug(`Uploaded ${file.name || 'image'} successfully`);
      } catch (err) {
        setPhotos(prev => prev.map((p, i) => i === targetIndex ? {
          ...p,
          uploading: false,
          error: err.message
        } : p));
        toast(`Photo upload fail: ${err.message}`, 'err');
        setUploadDebug(`Upload failed: ${err.message}`);
      }
      inserted += 1;
    }
    if (errors.length > 0) setUploadErrors(prev => [...prev, ...errors]);
    if (errors.length === 0 && inserted === 0) setUploadDebug('No file could be processed');
    replacePhotoIndexRef.current = null;
    setReplacePhotoIndex(null);
  };
  const handleGalleryInputChange = async e => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    setUploadDebug(`Input changed${files.length ? ` with ${files.length} file(s)` : ''}`);
    e.target.value = '';
    const replaceIndex = replacePhotoIndexRef.current;
    replacePhotoIndexRef.current = null;
    setReplacePhotoIndex(null);
    await handleFileChange(files, {
      replaceIndex
    });
  };
  const removePhoto = index => {
    const ph = photos[index];
    if (ph?.photoId) {
      API.delete('/api/photos/' + ph.photoId).catch(() => {});
    }
    setPhotos(prev => prev.filter((_, i) => i !== index));
  };
  const retryPhotoUpload = async index => {
    const ph = photos[index];
    if (!ph?.preview) return;
    setPhotos(prev => prev.map((p, i) => i === index ? { ...p, uploading: true, error: null } : p));
    try {
      const res = await uploadPhotoAsset({
        base64: ph.preview,
        type: ph.contentType || 'image/jpeg',
        index,
        totalExpected: Math.max(1, photos.length),
        consultationId: null
      });
      setPhotos(prev => prev.map((p, i) => i === index ? {
        ...p,
        uploading: false,
        error: null,
        photoId: res.photoId,
        url: res.url,
        sizeKB: res.sizeKB || p.sizeKB
      } : p));
      setUploadDebug('Photo retry upload successful');
    } catch (err) {
      setPhotos(prev => prev.map((p, i) => i === index ? { ...p, uploading: false, error: err.message } : p));
      toast(`Retry fail: ${err.message}`, 'err');
      setUploadDebug(`Retry failed: ${err.message}`);
    }
  };
  const handleNextUpload = async () => {
    if (readyPhotoCount < MIN_REPORT_PHOTOS) {
      toast(`Kam se kam ${MIN_REPORT_PHOTOS} photos upload karein`, 'warn');
      return;
    }
    if (photos.some(p => p.uploading)) {
      toast('Photos upload ho rahi hain, thoda wait karein...', 'warn');
      return;
    }

    // For coconut: Quick AI scan to set disease-specific questions
    const scanPhoto = readyPhotos[0] || photos[0] || null;
    if (selCrop?.id === 'coconut' && scanPhoto?.preview) {
      setAiScanning(true);
      try {
        const quickRes = await fetch('/api/coconut/ai-status');
        const status = await quickRes.json();
        if (status.ai_online) {
          const aiRes = await API.post('/api/coconut/quick-scan', {
            photoBase64: scanPhoto.preview
          });
          if (aiRes?.disease) {
            setPreAiResult(aiRes);
            // Set disease-specific questions
            const dqs = COCONUT_DISEASE_QS[aiRes.disease] || COCONUT_DISEASE_QS['Healthy'];
            setActiveQs(dqs.slice(0, 4));
            toast(`🥥 AI ne detect kiya: ${aiRes.disease} (${aiRes.confidence}%)`, 'inf');
          }
        }
      } catch (e) {
        console.warn('Quick scan:', e.message);
      } finally {
        setAiScanning(false);
      }
    }
    setStep(3);
    setQIndex(0);
    setCurrAns(null);
    setAnsweredList([]);
    setQAnswers({});
    setQ1Answer(null);
    if (!preAiResult || selCrop?.id !== 'coconut') {
      setActiveQs(isCoconut ? BRANCH_QS['coconut_spots'] || [] : BRANCH_QS['spots'] || []);
    }
  };

  // Camera handlers using getUserMedia
  const openCamera = async () => {
    if (photos.length >= MAX_REPORT_PHOTOS) {
      toast(`Maximum ${MAX_REPORT_PHOTOS} photos allowed`, 'warn');
      return;
    }
    setCameraError('');
    setShowCamera(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: {
            ideal: 1920
          },
          height: {
            ideal: 1080
          }
        },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err) {
      console.warn('Camera error:', err);
      if (err.name === 'NotAllowedError') setCameraError('Camera permission denied. Browser settings mein allow karo.');else if (err.name === 'NotFoundError') setCameraError('Camera nahi mila. Gallery use karein.');else setCameraError('Camera nahi khula: ' + err.message);
    }
  };
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setShowCamera(false);
    setCameraError('');
  };
  const capturePhoto = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    if (photos.length >= MAX_REPORT_PHOTOS) {
      toast(`Maximum ${MAX_REPORT_PHOTOS} photos allowed`, 'warn');
      return;
    }
    const srcWidth = video.videoWidth || 1280;
    const srcHeight = video.videoHeight || 720;
    const scale = Math.min(1, MAX_UPLOAD_DIMENSION / srcWidth, MAX_UPLOAD_DIMENSION / srcHeight);
    canvas.width = Math.max(1, Math.round(srcWidth * scale));
    canvas.height = Math.max(1, Math.round(srcHeight * scale));
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    let base64 = canvas.toDataURL('image/jpeg', UPLOAD_JPEG_QUALITY);
    try {
      base64 = await optimizeImageSrcToJpeg(base64);
    } catch (err) {
      console.warn('Camera optimize fail:', err.message);
    }
    // Process as uploaded file
    const sizeKB = Math.round(base64.length * 3 / 4 / 1024);
    const tempId = 'cam_' + Date.now();
    setPhotos(prev => [...prev, {
      tempId,
      preview: base64,
      uploading: true,
      error: null,
      sizeKB,
      contentType: 'image/jpeg'
    }]);
    stopCamera();
    // Upload to backend
    uploadPhotoAsset({
      base64,
      type: 'image/jpeg',
      index: photos.length,
      totalExpected: 1,
      consultationId: null
    }).then(res => {
      setPhotos(prev => prev.map(p => p.tempId === tempId ? {
        ...p,
        uploading: false,
        photoId: res.photoId,
        url: res.url,
        sizeKB: res.sizeKB || sizeKB
      } : p));
      toast('📸 Photo capture ho gayi! ✅');
    }).catch(err => {
      setPhotos(prev => prev.map(p => p.tempId === tempId ? {
        ...p,
        uploading: false,
        error: err.message
      } : p));
      toast('Photo upload fail: ' + err.message, 'err');
    });
  };

  // For coconut with pre-AI result: skip Q1, show 5 disease-specific questions directly
  const coconutSkipQ1 = isCoconut && preAiResult && activeQs.length > 0;
  const currentQ = coconutSkipQ1 ? activeQs[qIndex] || activeQs[0] || activeQs[activeQs.length - 1] : qIndex === 0 ? currentQ1 : activeQs[qIndex - 1] || currentQ1;
  const handleSelectOpt = opt => {
    setCurrAns(opt);
    if (qIndex === 0) {
      setQ1Answer(opt.id);
      setActiveQs(isCoconut ? BRANCH_QS[opt.id] || BRANCH_QS.coconut_none : BRANCH_QS[opt.id] || BRANCH_QS.none);
    }
  };
  const handleNextQ = () => {
    if (!currAns || !currentQ) return;
    const newAnswers = {
      ...qAnswers,
      [currentQ.id]: {
        id: currAns.id,
        label: currAns.label
      }
    };
    setQAnswers(newAnswers);
    setAnsweredList(p => [...p, {
      q: currentQ.text || currentQ.id,
      a: currAns.label,
      icon: currAns.icon
    }]);
    const nextIdx = qIndex + 1;
    // Coconut with AI: 5 disease-specific questions total
    const totalQs = coconutSkipQ1 ? Math.min(5, activeQs.length) : 5;
    if (nextIdx >= totalQs) {
      startProcessing();
      return;
    }
    setQIndex(nextIdx);
    setCurrAns(null);
  };
  const totalQ = 5;
  const progress = step === 1 ? 10 : step === 2 ? 35 : step === 3 ? 55 + qIndex / 5 * 30 : 90;
  return <div className="wrap-md">

      {/* Page Title */}
      <div style={{
      marginBottom: 24
    }}>
        <div style={{
        fontFamily: "'Baloo 2',cursive",
        fontSize: 26,
        fontWeight: 900,
        color: 'var(--g1)'
      }}>
          🔬 Crop Consultation
        </div>
        <div style={{
        fontSize: 14,
        color: 'var(--tx3)',
        marginTop: 4
      }}>
          {step === 1 && 'Step 1/3 — Apni fasal choose karein'}
          {step === 2 && 'Step 2/3 — Affected part ki photo lo'}
          {step === 3 && (isCoconut && preAiResult ? `🥥 Q${qIndex + 1}/5 — ${preAiResult.disease} ke baare mein` : `Step 3/3 — Sawaal ${Math.min(qIndex + 1, 5)} of ${totalQ}`)}
          {step === 4 && 'AI analysis ho rahi hai...'}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="prog-bar" style={{
      marginBottom: 24
    }}>
        <div className="prog-fill" style={{
        width: `${progress}%`,
        transition: 'width .5s ease'
      }} />
      </div>

      {/* ── STEP 1: CROP SELECT ── */}
      {step === 1 && <div className="slide-up">
          <div className="card" style={{
        padding: 26,
        marginBottom: 18
      }}>
            <div style={{
          fontSize: 15,
          fontWeight: 800,
          color: 'var(--g1)',
          marginBottom: 4
        }}>🌱 Apni Fasal Select Karein</div>
            <div style={{
          fontSize: 13,
          color: 'var(--tx3)',
          marginBottom: 18
        }}>Kaunsi crop mein problem hai?</div>
            <div className="crops-grid">
              {CROPS.map(c => <div key={c.id} className={`crop-tile${selCrop?.id === c.id ? ' sel' : ''}`} onClick={() => handleSelectCrop(c)}>
                  <div className="ct-em">{c.emoji}</div>
                  <div className="ct-nm">{c.name}</div>
                </div>)}
              <div key={OTHER_CROP.id} className={`crop-tile${selCrop?.id === OTHER_CROP.id ? ' sel' : ''}`} onClick={() => handleSelectCrop(OTHER_CROP)}>
                <div className="ct-em">{OTHER_CROP.emoji}</div>
                <div className="ct-nm">{OTHER_CROP.name}</div>
              </div>
            </div>
          </div>
          {selCrop && <div style={{
        padding: '12px 16px',
        background: 'var(--gp)',
        border: '1.5px solid var(--gpb)',
        borderRadius: 'var(--rad)',
        marginBottom: 16,
        display: 'flex',
        alignItems: 'center',
        gap: 12
      }} className="fade-in">
              <span style={{
          fontSize: 28
        }}>{selCrop.emoji}</span>
              <div><div style={{
            fontSize: 14,
            fontWeight: 800,
            color: 'var(--g1)'
          }}>{selCrop.name} ✅</div>
              <div style={{
            fontSize: 12,
            color: 'var(--tx3)'
          }}>Health: {selCrop.health}% · {selCrop.stage}</div></div>
              <button style={{
          marginLeft: 'auto'
        }} className="btn btn-ghost btn-sm" onClick={() => setSelCrop(null)}>✕</button>
            </div>}
          <button className="btn btn-g" style={{
        width: '100%',
        padding: '13px',
        fontSize: 15,
        borderRadius: 12
      }} disabled={!selCrop} onClick={handleNextCrop}>
            Aage Badho — Photo Upload →
          </button>
        </div>}

      {/* ── STEP 2: UPLOAD ── */}
      {step === 2 && <div className="slide-up">
          <button className="btn btn-ghost btn-sm" style={{
        marginBottom: 18
      }} onClick={() => setStep(1)}>← Crop Change Karo</button>

          {/* Crop badge */}
          <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        marginBottom: 20,
        padding: '11px 15px',
        background: 'var(--gp)',
        border: '1.5px solid var(--gpb)',
        borderRadius: 'var(--rad2)'
      }}>
            <span style={{
          fontSize: 24
        }}>{selCrop?.emoji}</span>
            <span style={{
          fontSize: 14,
          fontWeight: 700,
          color: 'var(--g1)'
        }}>{selCrop?.name} — Photo Upload</span>
            <span style={{
          marginLeft: 'auto',
          fontSize: 12,
          color: 'var(--tx3)',
          fontWeight: 600
        }}>{readyPhotoCount}/{MAX_REPORT_PHOTOS} photos ready</span>
          </div>

          {/* Gallery file input */}
          <input ref={uploadInputRef} id="galleryInput" type="file" accept="image/*" multiple style={{
        position: 'absolute',
        left: -9999,
        top: 'auto',
        width: 1,
        height: 1,
        opacity: 0
      }} onChange={handleGalleryInputChange} />

          <div style={{
        marginBottom: 16,
        padding: '12px 14px',
        background: 'white',
        border: '1.5px solid var(--br)',
        borderRadius: 12
      }}>
            <div style={{
          fontSize: 12,
          fontWeight: 800,
          color: 'var(--g1)',
          marginBottom: 8
        }}>Direct File Picker</div>
            <input type="file" accept="image/*" multiple style={{
          display: 'block',
          width: '100%',
          fontSize: 13
        }} onChange={handleGalleryInputChange} />
            <div style={{
          marginTop: 8,
          fontSize: 11.5,
          color: 'var(--tx3)'
        }}>Agar button se upload nahi ho raha, yahin se seedha file choose karein.</div>
          </div>

          <div style={{
        marginBottom: 16,
        padding: '10px 12px',
        background: 'var(--gp)',
        border: '1px solid var(--gpb)',
        borderRadius: 10,
        fontSize: 12.5,
        color: 'var(--g2)'
      }}>
            Upload status: {uploadDebug}
          </div>

          {/* Drag & drop review zone */}
          {false && <div className="upload-zone" style={{
        marginBottom: 14,
        background: dragActive ? 'var(--gp)' : 'var(--gb)',
        borderColor: dragActive ? 'var(--g4)' : 'var(--br2)'
      }} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop} onClick={() => openGallery()}>
            <div style={{
          fontSize: 40,
          marginBottom: 8
        }}>📥</div>
            <div style={{
          fontSize: 15,
          fontWeight: 800,
          color: 'var(--g1)'
        }}>Drag & Drop Images Here</div>
            <div style={{
          fontSize: 12.5,
          color: 'var(--tx3)',
          marginTop: 4
        }}>
              Ya click karke file select karein. Preview, replace aur delete neeche mil jayega.
            </div>
          </div>}

          {/* Camera Modal */}
          {showCamera && <div style={{
        position: 'fixed',
        inset: 0,
        background: '#000',
        zIndex: 9999,
        display: 'flex',
        flexDirection: 'column'
      }}>
              <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '12px 16px',
          background: 'rgba(0,0,0,.8)'
        }}>
                <span style={{
            color: 'white',
            fontWeight: 700,
            fontSize: 16
          }}>📷 Camera</span>
                <button style={{
            background: 'rgba(255,255,255,.2)',
            border: 'none',
            color: 'white',
            padding: '6px 14px',
            borderRadius: 8,
            cursor: 'pointer',
            fontWeight: 700
          }} onClick={stopCamera}>✕ Band Karo</button>
              </div>
              <video ref={videoRef} autoPlay playsInline muted style={{
          flex: 1,
          objectFit: 'cover',
          width: '100%'
        }} />
              <canvas ref={canvasRef} style={{
          display: 'none'
        }} />
              <div style={{
          padding: 20,
          background: 'rgba(0,0,0,.8)',
          display: 'flex',
          gap: 12,
          justifyContent: 'center',
          alignItems: 'center'
        }}>
                {cameraError ? <div style={{
            color: '#ff6b6b',
            fontSize: 13,
            textAlign: 'center'
          }}>⚠️ {cameraError}<br /><small>Gallery use karein</small></div> : <button style={{
            width: 68,
            height: 68,
            borderRadius: '50%',
            background: 'white',
            border: '4px solid rgba(255,255,255,.4)',
            cursor: 'pointer',
            fontSize: 28,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 0 3px rgba(255,255,255,.3)'
          }} onClick={capturePhoto}>📸</button>}
              </div>
            </div>}

          {/* Upload buttons */}
          <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 12,
        marginBottom: 20
      }}>
            <button type="button" className="btn btn-g" style={{
          padding: '16px 10px',
          fontSize: 14,
          borderRadius: 12,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 6
        }} onClick={openCamera}>
              <span style={{
            fontSize: 28
          }}>📷</span>
              <span>Camera Se Lo</span>
              <span style={{
            fontSize: 11,
            opacity: .7,
            fontWeight: 400
          }}>Seedha photo khicho</span>
            </button>
            <button type="button" className="btn btn-out" style={{
          padding: '16px 10px',
          fontSize: 14,
          borderRadius: 12,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 6,
          cursor: 'pointer'
        }} onClick={openGallery}>
              <span style={{
            fontSize: 28
          }}>🖼️</span>
              <span>File Upload</span>
              <span style={{
            fontSize: 11,
            opacity: .7,
            fontWeight: 400
          }}>3–5 photos select karo</span>
            </button>
          </div>

          {/* Tips */}
          <div style={{
        display: 'flex',
        gap: 8,
        flexWrap: 'wrap',
        marginBottom: 16
      }}>
            {['💡 Natural light mein lo', '🔍 Close-up zaroor lo', '📐 Alag angles se lo', '🌿 Healthy + sick dono', '📍 3 se 5 photos allowed'].map(t => <div key={t} style={{
          background: 'white',
          border: '1px solid var(--br)',
          borderRadius: 100,
          padding: '4px 11px',
          fontSize: 11.5,
          color: 'var(--tx3)'
        }}>{t}</div>)}
          </div>
          <div style={{
        marginBottom: 14,
        fontSize: 12.5,
        color: 'var(--tx3)',
        lineHeight: 1.6
      }}>
            Har image ko niche review karein. Aap kisi photo ko replace ya delete karke final submission se pehle dubara check kar sakte hain.
          </div>

          {/* Upload progress / errors */}
          {uploadErrors.length > 0 && <div style={{
        background: '#fff0f0',
        border: '1.5px solid var(--r2)',
        borderRadius: 10,
        padding: '10px 14px',
        marginBottom: 14
      }}>
              {uploadErrors.map((err, i) => <div key={i} style={{
          fontSize: 12.5,
          color: 'var(--r2)',
          fontWeight: 600
        }}>⚠️ {err}</div>)}
            </div>}

          {/* Photo grid — show uploaded photos */}
          {photos.length > 0 && <div style={{
        marginBottom: 18
      }}>
              <div style={{
          fontSize: 13,
          fontWeight: 700,
          color: 'var(--g1)',
          marginBottom: 10
        }}>
                📸 Uploaded Photos ({readyPhotoCount}/{MAX_REPORT_PHOTOS})
                {photos.some(p => p.uploading) && <span style={{
            marginLeft: 8,
            fontSize: 11,
            color: 'var(--a2)',
            fontWeight: 600
          }}>⏳ uploading...</span>}
                {photos.every(p => !p.uploading && p.photoId) && <span style={{
            marginLeft: 8,
            fontSize: 11,
            color: 'var(--g4)',
            fontWeight: 600
          }}>✅ sab ready</span>}
              </div>
              <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3,1fr)',
          gap: 10
        }}>
                {photos.map((ph, i) => <div key={ph.photoId || ph.tempId || i} style={{
            position: 'relative',
            borderRadius: 10,
            overflow: 'hidden',
            border: `2px solid ${ph.uploading ? 'var(--a2)' : ph.error ? 'var(--r2)' : 'var(--g4)'}`,
            aspectRatio: '1',
            background: 'var(--gp)'
          }}>
                    <img src={ph.preview} alt={`photo ${i + 1}`} style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              display: 'block'
            }} />
                    {/* Status overlay */}
                    {ph.uploading && <div style={{
              position: 'absolute',
              inset: 0,
              background: 'rgba(0,0,0,.45)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
                        <div className="spin" style={{
                width: 20,
                height: 20,
                borderColor: 'white',
                borderTopColor: 'transparent'
              }} />
                      </div>}
                    {ph.error && <div style={{
              position: 'absolute',
              inset: 0,
              background: 'rgba(255,0,0,.25)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 20
            }}>⚠️</div>}
                    {ph.error && !ph.uploading && <button style={{
              position: 'absolute',
              bottom: 30,
              left: 4,
              minWidth: 54,
              padding: '2px 7px',
              borderRadius: 100,
              background: 'rgba(255,255,255,.92)',
              border: 'none',
              color: 'var(--r2)',
              fontSize: 9.5,
              fontWeight: 800,
              cursor: 'pointer',
              zIndex: 2
            }} onClick={e => {
              e.stopPropagation();
              retryPhotoUpload(i);
            }}>Retry</button>}
                    {!ph.uploading && !ph.error && <div style={{
              position: 'absolute',
              top: 4,
              left: 4,
              background: 'rgba(30,126,66,.85)',
              borderRadius: 100,
              padding: '2px 7px',
              fontSize: 10,
              color: 'white',
              fontWeight: 700
            }}>✓ {i + 1}</div>}
                    {/* Delete button */}
                    <button style={{
              position: 'absolute',
              top: 4,
              right: 4,
              width: 22,
              height: 22,
              borderRadius: '50%',
              background: 'rgba(0,0,0,.6)',
              border: 'none',
              color: 'white',
              fontSize: 12,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }} onClick={() => removePhoto(i)}>✕</button>
                    {/* Replace button */}
                    <button style={{
              position: 'absolute',
              bottom: 4,
              left: 4,
              minWidth: 54,
              padding: '2px 7px',
              borderRadius: 100,
              background: 'rgba(0,0,0,.6)',
              border: 'none',
              color: 'white',
              fontSize: 9.5,
              fontWeight: 700,
              cursor: 'pointer'
            }} onClick={() => openGallery(i)}>↺ Replace</button>
                    {/* Size badge */}
                    {ph.sizeKB && <div style={{
              position: 'absolute',
              bottom: 4,
              right: 4,
              background: 'rgba(0,0,0,.55)',
              borderRadius: 6,
              padding: '1px 5px',
              fontSize: 9,
              color: 'white'
            }}>{ph.sizeKB}KB</div>}
                  </div>)}
                {/* Add more button */}
                {photos.length < MAX_REPORT_PHOTOS && <div style={{
            borderRadius: 10,
            border: '2px dashed var(--br2)',
            aspectRatio: '1',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 6,
            cursor: 'pointer',
            background: 'var(--gp)'
          }} onClick={() => openGallery()}>
                    <span style={{
              fontSize: 24
            }}>➕</span>
                    <span style={{
              fontSize: 11,
              fontWeight: 700,
              color: 'var(--g3)'
            }}>Add More</span>
                  </div>}
              </div>
            </div>}

          {/* Empty state — no photos yet */}
          {photos.length === 0 && <div style={{
        textAlign: 'center',
        padding: '28px 20px',
        background: 'var(--gp)',
        borderRadius: 'var(--rad)',
        border: '2px dashed var(--br2)',
        marginBottom: 16
      }}>
              <div style={{
          fontSize: 48,
          marginBottom: 8
        }}>🌿</div>
              <div style={{
          fontSize: 14,
          fontWeight: 700,
          color: 'var(--tx)',
          marginBottom: 4
        }}>Koi photo nahi upload hui</div>
              <div style={{
          fontSize: 12.5,
          color: 'var(--tx3)'
        }}>Upar Camera ya Gallery button use karein</div>
            </div>}

          <button className="btn btn-g" style={{
        width: '100%',
        padding: '13px',
        fontSize: 15,
        borderRadius: 12
      }} disabled={readyPhotoCount < MIN_REPORT_PHOTOS || photos.some(p => p.uploading) || aiScanning} onClick={handleNextUpload}>
            {photos.some(p => p.uploading) ? <><div className="spin" style={{
            width: 16,
            height: 16,
            display: 'inline-block',
            marginRight: 8
          }} /> Upload ho raha hai...</> : aiScanning ? <><div className="spin" style={{
            width: 16,
            height: 16,
            display: 'inline-block',
            marginRight: 8
          }} /> 🥥 AI Analysis ho rahi hai...</> : selCrop?.id === 'coconut' ? `🥥 ${readyPhotoCount} Photo${readyPhotoCount > 1 ? 's' : ''} Ready — Coconut Diagnosis Shuru Karo →` : `🤖 ${readyPhotoCount} Photo${readyPhotoCount > 1 ? 's' : ''} Ready — Diagnosis Shuru Karo →`}
          </button>
        </div>}

            {/* ── STEP 3: QUESTION FLOW ── */}
      {step === 3 && currentQ && <div>
          <button className="btn btn-ghost btn-sm" style={{
        marginBottom: 18
      }} onClick={() => {
        setStep(2);
      }}>← Photo Change Karo</button>

          {/* Q Progress Dots */}
          <div className="qf-prog">
            {Array.from({
          length: totalQ
        }, (_, i) => [i > 0 && <div key={'l' + i} className={`qf-line${i <= qIndex ? ' done' : ''}`} />, <div key={'d' + i} className={`qf-dot${i < qIndex ? ' done' : i === qIndex ? ' active' : ''}`}>
                {i < qIndex ? '✓' : i + 1}
              </div>]).flat().filter(Boolean)}
          </div>

          {/* Answered Questions (collapsed) */}
          {answeredList.map((a, i) => <div key={i} className="qf-answered">
              <div className="qf-ans-check">✓</div>
              <div>
                <div className="qf-ans-q">{a.q}</div>
                <div className="qf-ans-a">{a.icon} {a.a}</div>
              </div>
            </div>)}

          {/* Current Question Card */}
          <div className="qf-card" key={currentQ.id}>
            <div className="qf-num">Sawaal {qIndex + 1} of {totalQ}</div>
            <div className="qf-q">{currentQ.text}</div>
            {currentQ.hint && <div className="qf-hint">{currentQ.hint}</div>}
            <div className="qf-opts">
              {currentQ.options.map(o => <button key={o.id} className={`qf-opt${currAns?.id === o.id ? ' sel' : ''}`} onClick={() => handleSelectOpt(o)}>
                  <span className="qf-opt-icon">{o.icon}</span>
                  <span>{o.label}</span>
                </button>)}
            </div>
          </div>

          <button className="btn btn-g" style={{
        width: '100%',
        padding: '13px',
        fontSize: 15,
        borderRadius: 12
      }} disabled={!currAns} onClick={handleNextQ}>
            {qIndex < 4 ? 'Agle Sawaal →' : '✅ Diagnosis Shuru Karo →'}
          </button>
        </div>}

      {/* ── STEP 4: PROCESSING ── */}
      {step === 4 && <div className="card" style={{
      padding: 40,
      textAlign: 'center'
    }}>
          <div className="proc-icon">🤖</div>
          <div style={{
        fontFamily: "'Baloo 2',cursive",
        fontSize: 22,
        fontWeight: 900,
        color: 'var(--g1)',
        marginBottom: 7
      }}>AI Analyze Kar Raha Hai...</div>
          <div style={{
        fontSize: 14,
        color: 'var(--tx3)',
        marginBottom: 22
      }}>Aapki photo aur {answeredList.length} answers process ho rahe hain</div>
          <div style={{
        maxWidth: 300,
        margin: '0 auto'
      }}>
            <div className="prog-bar" style={{
          marginBottom: 18
        }}><div className="prog-fill" style={{
            width: `${procStep / 4 * 100}%`
          }} /></div>
            <div className="proc-steps-list">
              {PROC_STEPS.map((s, i) => <div key={i} className={`proc-step${i < procStep ? ' done' : i === procStep ? ' act' : ''}`}>
                  <span style={{
              fontSize: 15
            }}>{i < procStep ? '✅' : i === procStep ? '🔄' : '⏳'}</span>
                  <span>{s}</span>
                </div>)}
            </div>
          </div>
        </div>}
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   AI REPORT PAGE  — Full Branded Report with Logo
════════════════════════════════════════════════════════════════ */
