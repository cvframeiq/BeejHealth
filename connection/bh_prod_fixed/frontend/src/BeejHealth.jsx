import APP_CSS from './styles/App.css?raw';
import BeejHealthApp from './components/BeejHealthApp.jsx';
import RobotAnalyticsPage from './pages/RobotAnalyticsPage.jsx';
import RobotMaintenancePage from './pages/RobotMaintenancePage.jsx';
import RobotControlPage from './pages/RobotControlPage.jsx';
import RobotMapPage from './pages/RobotMapPage.jsx';
import RobotCameraPage from './pages/RobotCameraPage.jsx';
import RobotSprayPage from './pages/RobotSprayPage.jsx';
import RobotDashboard from './pages/RobotDashboard.jsx';
import GovtMapPage from './pages/GovtMapPage.jsx';
import InsurancePage from './pages/InsurancePage.jsx';
import MarketplacePage from './pages/MarketplacePage.jsx';
import B2BPage from './pages/B2BPage.jsx';
import SoilSensorPage from './pages/SoilSensorPage.jsx';
import ForecastPage from './pages/ForecastPage.jsx';
import SatellitePage from './pages/SatellitePage.jsx';
import VoiceInputPage from './pages/VoiceInputPage.jsx';
import SettingsPage from './pages/SettingsPage.jsx';
import EarningsPage from './pages/EarningsPage.jsx';
import ProfilePage from './pages/ProfilePage.jsx';
import SupportPage from './pages/SupportPage.jsx';
import NotifPage from './pages/NotifPage.jsx';
import MyFarmPage from './pages/MyFarmPage.jsx';
import CaseDetailPage from './pages/CaseDetailPage.jsx';
import ChatPage from './pages/ChatPage.jsx';
import BookingPage from './pages/BookingPage.jsx';
import ExpertsPage from './pages/ExpertsPage.jsx';
import FarmerDash from './pages/FarmerDash.jsx';
import ExpertDash from './pages/ExpertDash.jsx';
import MyConsultPage from './pages/MyConsultPage.jsx';
import AIReportPage from './pages/AIReportPage.jsx';
import ConsultPage from './pages/ConsultPage.jsx';
import HomePage from './pages/HomePage.jsx';
import ExpertOnboarding from './components/ExpertOnboarding.jsx';
import FarmerOnboarding from './components/FarmerOnboarding.jsx';
import AuthModal from './components/AuthModal.jsx';

import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import HomePageI18n from './HomePageI18n.jsx';

/* ── API Helper ── */
/* ── App Version — clears stale localStorage on update ── */
const APP_VERSION = '3.5';
(() => {
  try {
    const stored = localStorage.getItem('bh_app_version');
    if (stored !== APP_VERSION) {
      // App version changed.
      // Keep login/session data during app updates.
      localStorage.setItem('bh_app_version', APP_VERSION);
      console.log('BeejHealth v' + APP_VERSION + ' ready');
    }
  } catch (e) {}
})();
import { API, saveSession, clearSession, loadSession } from './services/api.js';
import { getConsultationContextId, rememberConsultationContext, formatChatMessage, resolveConsultPhotoUrls, fileToDataUrl, estimateDataUrlBytes, inferDataUrlMimeType, isLikelyImageFile, loadImageFromSrc, ChatMessageBody } from './utils/helpers.jsx';
function openPrintableReport(reportRoot, fileName = 'BeejHealth-Report') {
  if (!reportRoot) return false;
  const clone = reportRoot.cloneNode(true);
  clone.querySelectorAll('[data-no-export="true"]').forEach(el => el.remove());
  const printCss = `
    @page { size: A4 portrait; margin: 12mm; }
    html, body { margin: 0; padding: 0; background: #ffffff; }
    body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .wrap-sm { max-width: 760px; margin: 0 auto; padding: 0; }
    .rep-sheet { box-shadow: none; border-radius: 0; }
    .rep-body { padding-bottom: 18px; }
    .rep-actions { display: none !important; }
    .report-print-title { display: none; }
    @media print {
      .rep-sheet { break-inside: auto; }
    }
  `;
  const win = window.open('', '_blank', 'noopener,noreferrer,width=980,height=1200');
  if (!win) return false;
  win.document.open();
  win.document.write(`<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>${fileName}</title>
    <style>${APP_CSS}</style>
    <style>${printCss}</style>
  </head>
  <body>
    ${clone.outerHTML}
  </body>
</html>`);
  win.document.close();
  const triggerPrint = () => {
    try {
      win.focus();
    } catch {}
    try {
      win.print();
    } catch {}
  };
  const waitForAssetsAndPrint = async () => {
    try {
      if (win.document.fonts?.ready) await win.document.fonts.ready;
    } catch {}
    const imgs = Array.from(win.document.images || []);
    if (!imgs.length) return triggerPrint();
    let loaded = 0;
    const onDone = () => {
      loaded += 1;
      if (loaded >= imgs.length) triggerPrint();
    };
    imgs.forEach(img => {
      if (img.complete) onDone();else {
        img.addEventListener('load', onDone, {
          once: true
        });
        img.addEventListener('error', onDone, {
          once: true
        });
      }
    });
  };
  if (win.document.readyState === 'complete' || win.document.readyState === 'interactive') {
    setTimeout(waitForAssetsAndPrint, 250);
  } else {
    win.addEventListener('load', () => setTimeout(waitForAssetsAndPrint, 250), {
      once: true
    });
  }
  win.onafterprint = () => {
    try {
      win.close();
    } catch {}
  };
  return true;
}
async function waitForReportAssets(root) {
  try {
    if (document.fonts?.ready) await document.fonts.ready;
  } catch {}
  const imgs = Array.from(root?.querySelectorAll('img') || []);
  await Promise.all(imgs.map(img => new Promise(resolve => {
    if (img.complete && img.naturalWidth > 0) return resolve();
    const done = () => resolve();
    img.addEventListener('load', done, {
      once: true
    });
    img.addEventListener('error', done, {
      once: true
    });
  })));
}
function normalizePdfText(value) {
  return String(value ?? '').replace(/\r/g, '').replace(/\u00a0/g, ' ').replace(/[\u2013\u2014]/g, '-').replace(/[\u2022]/g, '-').replace(/₹/g, 'Rs.').replace(/°/g, ' deg').replace(/×/g, 'x').replace(/[“”]/g, '"').replace(/[‘’]/g, "'").replace(/[^\x09\x0A\x0D\x20-\x7E]/g, '');
}
function buildPdfDocument({
  reportId,
  crop,
  today,
  reportD,
  dbConsult,
  effectiveTop3,
  effectiveTreatments,
  effectiveUrgency,
  triggerText,
  sprayText,
  qAnswers
}) {
  const PAGE_W = 595.28;
  const PAGE_H = 841.89;
  const MARGIN_X = 40;
  const TOP_GAP = 34;
  const BOTTOM_GAP = 40;
  const FIRST_HEADER_H = 88;
  const HEADER_H = 38;
  const CONTENT_W = PAGE_W - MARGIN_X * 2;
  const textCanvas = document.createElement('canvas');
  const textCtx = textCanvas.getContext('2d');
  const pages = [];
  let current = null;
  let cursorY = 0;
  const color = (r, g, b) => `${r.toFixed(3)} ${g.toFixed(3)} ${b.toFixed(3)} rg`;
  const strokeColor = (r, g, b) => `${r.toFixed(3)} ${g.toFixed(3)} ${b.toFixed(3)} RG`;
  const measure = (text, fontSize, bold = false) => {
    if (!textCtx) return String(text || '').length * fontSize * 0.52;
    textCtx.font = `${bold ? '700' : '400'} ${fontSize}px Helvetica, Arial, sans-serif`;
    return textCtx.measureText(String(text || '')).width;
  };
  const wrapText = (text, maxWidth, fontSize, bold = false) => {
    const out = [];
    const paragraphs = normalizePdfText(text).split(/\n+/);
    for (const paragraph of paragraphs) {
      const cleaned = paragraph.trim();
      if (!cleaned) {
        out.push('');
        continue;
      }
      const words = cleaned.split(/\s+/);
      let line = '';
      const pushLine = () => {
        if (line) out.push(line);
        line = '';
      };
      for (const word of words) {
        const candidate = line ? `${line} ${word}` : word;
        if (measure(candidate, fontSize, bold) <= maxWidth) {
          line = candidate;
          continue;
        }
        if (line) pushLine();
        if (measure(word, fontSize, bold) <= maxWidth) {
          line = word;
          continue;
        }
        let chunk = '';
        for (const ch of word) {
          const test = `${chunk}${ch}`;
          if (measure(test, fontSize, bold) <= maxWidth) {
            chunk = test;
          } else {
            if (chunk) out.push(chunk);
            chunk = ch;
          }
        }
        line = chunk;
      }
      pushLine();
    }
    return out.length ? out : [''];
  };
  const addOp = op => {
    current.ops.push(op);
  };
  const textAt = (text, x, yFromTop, fontSize = 11, bold = false, rgb = [0.08, 0.16, 0.09]) => {
    const y = PAGE_H - yFromTop - fontSize;
    addOp(`BT /${bold ? 'F2' : 'F1'} ${fontSize} Tf ${color(rgb[0], rgb[1], rgb[2])} 1 0 0 1 ${x.toFixed(2)} ${y.toFixed(2)} Tm (${normalizePdfText(text).replace(/\\/g, '\\\\').replace(/\(/g, '\\(').replace(/\)/g, '\\)')}) Tj ET`);
  };
  const startPage = (isFirst = false) => {
    current = {
      ops: [],
      headerH: isFirst ? FIRST_HEADER_H : HEADER_H
    };
    pages.push(current);
    if (isFirst) {
      addOp(`${color(0.07, 0.36, 0.18)}`);
      addOp(`0 ${PAGE_H - FIRST_HEADER_H} ${PAGE_W} ${FIRST_HEADER_H} re f`);
      addOp(`${color(0.11, 0.46, 0.24)}`);
      addOp(`0 ${PAGE_H - FIRST_HEADER_H - 12} ${PAGE_W} 12 re f`);
      textAt('BeejHealth AI Report', MARGIN_X, 18, 20, true, [1, 1, 1]);
      textAt(`Report #${reportId}`, MARGIN_X, 46, 9, false, [0.89, 0.97, 0.91]);
      textAt(`${normalizePdfText(crop.name)} | ${today}`, MARGIN_X, 61, 11, false, [0.91, 0.98, 0.94]);
    } else {
      textAt('BeejHealth AI Report', MARGIN_X, 16, 13, true, [0.07, 0.36, 0.18]);
      textAt(`Report #${reportId}`, MARGIN_X, 31, 8.5, false, [0.32, 0.48, 0.35]);
      addOp(`${strokeColor(0.83, 0.9, 0.85)}`);
      addOp(`${MARGIN_X} ${PAGE_H - HEADER_H + 2} m ${PAGE_W - MARGIN_X} ${PAGE_H - HEADER_H + 2} l S`);
    }
    cursorY = 8;
  };
  const ensureSpace = needed => {
    const usable = PAGE_H - TOP_GAP - BOTTOM_GAP - current.headerH;
    if (cursorY + needed > usable) startPage(false);
  };
  const lineHeight = fontSize => Math.max(14, Math.round(fontSize * 1.35));
  const addParagraph = (text, opts = {}) => {
    const fontSize = opts.fontSize ?? 11;
    const indent = opts.indent ?? 0;
    const bold = opts.bold ?? false;
    const rgb = opts.rgb ?? [0.08, 0.16, 0.09];
    const leading = opts.leading ?? lineHeight(fontSize);
    const wrapped = wrapText(text, CONTENT_W - indent, fontSize, bold);
    wrapped.forEach(line => {
      ensureSpace(leading);
      if (line) {
        const yTop = TOP_GAP + current.headerH + cursorY;
        textAt(line, MARGIN_X + indent, yTop, fontSize, bold, rgb);
      }
      cursorY += leading;
    });
    if (opts.after) cursorY += opts.after;
  };
  const addSectionTitle = (title, tag) => {
    ensureSpace(28);
    const yTop = TOP_GAP + current.headerH + cursorY;
    textAt(title, MARGIN_X, yTop, 13.5, true, [0.07, 0.36, 0.18]);
    if (tag) {
      textAt(tag, PAGE_W - MARGIN_X - Math.min(measure(tag, 8.5, true), 150), yTop, 8.5, true, [0.26, 0.46, 0.29]);
    }
    cursorY += 18;
  };
  const addDivider = () => {
    ensureSpace(10);
    const y = PAGE_H - (TOP_GAP + current.headerH + cursorY);
    addOp(`${strokeColor(0.85, 0.9, 0.86)}`);
    addOp(`${MARGIN_X} ${y.toFixed(2)} m ${PAGE_W - MARGIN_X} ${y.toFixed(2)} l S`);
    cursorY += 10;
  };
  const addKeyValue = (label, value, opts = {}) => {
    const labelWidth = opts.labelWidth ?? 138;
    const gap = opts.gap ?? 10;
    const fontSize = opts.fontSize ?? 10.5;
    const bold = opts.bold ?? false;
    const valueFontSize = opts.valueFontSize ?? 11;
    const valueBold = opts.valueBold ?? false;
    const labelColor = opts.labelColor ?? [0.4, 0.56, 0.45];
    const valueColor = opts.valueColor ?? [0.08, 0.16, 0.09];
    const rowLeading = opts.leading ?? 14;
    const valueWidth = CONTENT_W - labelWidth - gap;
    const labelLines = wrapText(label, labelWidth, fontSize, bold);
    const valueLines = wrapText(value, valueWidth, valueFontSize, valueBold);
    const rowHeight = Math.max(labelLines.length, valueLines.length) * rowLeading;
    ensureSpace(rowHeight + 2);
    const baseY = TOP_GAP + current.headerH + cursorY;
    const maxLines = Math.max(labelLines.length, valueLines.length);
    for (let i = 0; i < maxLines; i += 1) {
      const lineY = baseY + i * rowLeading;
      if (labelLines[i]) textAt(i === 0 ? labelLines[i] : '', MARGIN_X, lineY, fontSize, bold, labelColor);
      if (valueLines[i]) textAt(valueLines[i], MARGIN_X + labelWidth + gap, lineY, valueFontSize, valueBold, valueColor);
    }
    cursorY += rowHeight + 3;
  };
  const addBullets = (items, opts = {}) => {
    const fontSize = opts.fontSize ?? 10.8;
    const bulletIndent = opts.bulletIndent ?? 12;
    const gap = opts.gap ?? 4;
    const rowLeading = opts.leading ?? lineHeight(fontSize);
    const bulletPrefix = opts.prefix ?? '- ';
    items.forEach(item => {
      const text = typeof item === 'string' ? item : `${item.title}${item.desc ? ` - ${item.desc}` : ''}`;
      const wrapped = wrapText(text, CONTENT_W - bulletIndent - gap, fontSize, false);
      wrapped.forEach((line, idx) => {
        ensureSpace(rowLeading);
        const yTop = TOP_GAP + current.headerH + cursorY;
        if (idx === 0) {
          textAt(`${bulletPrefix}${line}`, MARGIN_X + bulletIndent, yTop, fontSize, false, [0.08, 0.16, 0.09]);
        } else {
          textAt(line, MARGIN_X + bulletIndent + 10, yTop, fontSize, false, [0.08, 0.16, 0.09]);
        }
        cursorY += rowLeading;
      });
      cursorY += 2;
    });
  };
  startPage(true);
  addSectionTitle('Report Overview');
  addKeyValue('Crop', normalizePdfText(crop?.name || 'Crop'));
  addKeyValue('Detected disease', reportD.disease);
  addKeyValue('Scientific name', reportD.sci);
  addKeyValue('Hindi name', reportD.hindi);
  addKeyValue('Confidence', `${reportD.conf}%`);
  addKeyValue('Severity', reportD.sevLabel);
  addKeyValue('Area affected', `${reportD.aff}%`);
  addKeyValue('Risk level', reportD.sev >= 3 ? 'High' : reportD.sev === 2 ? 'Medium' : 'Low');
  addDivider();
  addSectionTitle('Disease Identification');
  addParagraph(`Report ID ${normalizePdfText(reportId)} was generated on ${today}. The analysis below summarizes the uploaded crop photo and questionnaire signals.`);
  const reportPhotoUrls = resolveConsultPhotoUrls(dbConsult);
  if (dbConsult?.photoCount || reportPhotoUrls.length) {
    const count = dbConsult.photoCount || reportPhotoUrls.length || 1;
    addKeyValue('Uploaded photos', `${count} photo${count > 1 ? 's' : ''}`);
  }
  addKeyValue('Detection source', reportPhotoUrls.length ? 'Photo upload' : 'AI diagnosis preview');
  if (dbConsult?.reportSnapshot?.summary || dbConsult?.aiReportSummary || dbConsult?.report || reportD.cause) {
    addKeyValue('Primary note', dbConsult?.reportSnapshot?.summary || dbConsult?.aiReportSummary || dbConsult?.report || reportD.cause);
  }
  addDivider();
  if (effectiveTop3?.length) {
    addSectionTitle('Model Predictions', 'Coconut / AI result');
    addBullets(effectiveTop3.map(p => `${p.rank}. ${p.disease} - ${p.confidence}% confidence`), {
      fontSize: 10.5
    });
    if (effectiveUrgency) addKeyValue('Urgency', normalizePdfText(effectiveUrgency));
    if (normalizePdfText(effectiveTop3[0]?.disease || '')) addKeyValue('Top prediction', `${normalizePdfText(effectiveTop3[0].disease)} (${effectiveTop3[0].confidence}%)`);
    addDivider();
  }
  if (effectiveTreatments?.length) {
    addSectionTitle('AI Treatment Plan', 'Model based');
    addBullets(effectiveTreatments, {
      fontSize: 10.5
    });
    addDivider();
  }
  addSectionTitle('Questionnaire Signals');
  const answers = qAnswers || {};
  const answerRows = [['Q2', answers.q2?.label], ['Q3', answers.q3?.label], ['Q9', answers.q9?.label], ['Q20', answers.q20?.label]].filter(([, val]) => val);
  if (answerRows.length) {
    answerRows.forEach(([label, val]) => addKeyValue(label, val));
  } else {
    addParagraph('No questionnaire answers were available for this report.');
  }
  addKeyValue('Trigger', triggerText);
  addKeyValue('Spray history', sprayText);
  addDivider();
  addSectionTitle('Treatment Plan');
  addKeyValue('Immediate action', reportD.phases[0]);
  addKeyValue('This week', reportD.phases[1]);
  addKeyValue('This season', reportD.phases[2]);
  addDivider();
  addSectionTitle('Medicines');
  addBullets(reportD.meds.map(m => `${m.nm} - ${m.ty} - ${m.pr}`), {
    fontSize: 10.4
  });
  addDivider();
  addSectionTitle('Weather Risk');
  addKeyValue('Humidity note', reportD.humidity);
  addKeyValue('Risk', reportD.risk);
  addKeyValue('Affected now', `${reportD.aff}% leaves`);
  addKeyValue('Untreated spread', `${reportD.utr}% in 7 days`);
  addKeyValue('Treated spread', `${reportD.tr}% in 5 days`);
  addDivider();
  addSectionTitle('Expert Advice');
  addKeyValue('Recommended expert', 'Dr. Rajesh Kumar');
  addKeyValue('Follow-up', 'Upload a fresh photo after 7 days');
  addParagraph('This PDF is meant for quick sharing and record keeping. Please consult a certified agricultural expert before making major treatment decisions.');
  const objectStrings = [];
  const addObject = content => {
    objectStrings.push(content);
    return objectStrings.length;
  };
  const font1 = addObject('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');
  const font2 = addObject('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>');
  const contentNumbers = [];
  pages.forEach(page => {
    const content = page.ops.join('\n');
    const contentNum = addObject(`<< /Length ${new TextEncoder().encode(content).length} >>\nstream\n${content}\nendstream`);
    contentNumbers.push(contentNum);
  });
  pages.forEach((page, index) => {
    const kids = `<< /Type /Page /Parent {PAGES} 0 R /MediaBox [0 0 ${PAGE_W} ${PAGE_H}] /Resources << /Font << /F1 ${font1} 0 R /F2 ${font2} 0 R >> >> /Contents ${contentNumbers[index]} 0 R >>`;
    addObject(kids);
  });
  const pageObjectNumbers = [];
  for (let i = 3 + contentNumbers.length; i <= 2 + contentNumbers.length + pages.length; i += 1) {
    pageObjectNumbers.push(i);
  }
  const pagesIndex = addObject(`<< /Type /Pages /Kids [${pageObjectNumbers.map(n => `${n} 0 R`).join(' ')}] /Count ${pageObjectNumbers.length} >>`);
  const catalogIndex = addObject(`<< /Type /Catalog /Pages ${pagesIndex} 0 R >>`);
  const finalObjects = objectStrings.map(content => content.split('{PAGES}').join(String(pagesIndex)));
  const header = '%PDF-1.4\n';
  const chunks = [header];
  const offsets = [0];
  let position = new TextEncoder().encode(header).length;
  finalObjects.forEach((content, idx) => {
    const obj = `${idx + 1} 0 obj\n${content}\nendobj\n`;
    offsets.push(position);
    position += new TextEncoder().encode(obj).length;
    chunks.push(obj);
  });
  const xrefStart = position;
  const xref = [`xref\n0 ${finalObjects.length + 1}\n`, '0000000000 65535 f \n'];
  for (let i = 1; i <= finalObjects.length; i += 1) {
    xref.push(`${String(offsets[i]).padStart(10, '0')} 00000 n \n`);
  }
  xref.push(`trailer\n<< /Size ${finalObjects.length + 1} /Root ${catalogIndex} 0 R >>\nstartxref\n${xrefStart}\n%%EOF`);
  const pdfString = chunks.join('') + xref.join('');
  return new Blob([pdfString], {
    type: 'application/pdf'
  });
}
import { SOILS, INDIA_STATES, INDIA_DISTRICTS, MAHARASHTRA_TALUKAS, DISTRICTS, TALUKAS, CROPS, OTHER_CROP, EXPERTS, CONSULTATIONS, NOTIFICATIONS, MESSAGES_DATA, DISEASE_DB, ErrorBoundary } from './utils/constants.jsx';
/* ════════════════════════════════════════════════════════════════
   AUTH MODAL
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   FARMER ONBOARDING
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   EXPERT ONBOARDING
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   HOME PAGE
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   AI REPORT PAGE  — Full Branded Report with Logo
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   MY CONSULTATIONS
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   CONSULTATION PAGE
════════════════════════════════════════════════════════════════ */
const LOGO_FULL_B64 = "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";

/* ════════════════════════════════════════════════════════════════
   QUESTION FLOW DATA
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   COCONUT AI — Disease-specific questions (30 total, 5 shown)
   After AI detects disease, show relevant questions for that disease
════════════════════════════════════════════════════════════════ */

// Maps AI disease name → question branch
const COCONUT_DISEASE_BRANCH = {
  'Gray Leaf Spot': 'coconut_spots',
  'Gray Leaf Spot_multiple': 'coconut_spots',
  'Leaf rot': 'coconut_spots',
  'Leaf rot_multiple': 'coconut_spots',
  'bud rot': 'coconut_wilt',
  'stem bleeding': 'coconut_wilt',
  'stem bleeding_multiple': 'coconut_wilt',
  'Healthy': 'coconut_none'
};

// Per-disease targeted questions — confirm & detail the AI detection
const COCONUT_DISEASE_QS = {
  'Gray Leaf Spot': [{
    id: 'cgls1',
    text: 'Daag ka rang kaisa hai?',
    hint: 'Gray Leaf Spot confirm karne ke liye',
    options: [{
      id: 'gray',
      label: 'Bhoora/Gray rang',
      icon: '⬜'
    }, {
      id: 'brown',
      label: 'Dark Brown',
      icon: '🟤'
    }, {
      id: 'yellow',
      label: 'Peele edges ke saath',
      icon: '🟡'
    }, {
      id: 'black',
      label: 'Kala daag',
      icon: '⚫'
    }]
  }, {
    id: 'cgls2',
    text: 'Daag patte ke kahan hain?',
    hint: 'Location batao',
    options: [{
      id: 'tip',
      label: 'Patte ki nauk pe',
      icon: '📍'
    }, {
      id: 'middle',
      label: 'Beech mein',
      icon: '🎯'
    }, {
      id: 'all',
      label: 'Poore patte pe',
      icon: '🍃'
    }, {
      id: 'base',
      label: 'Neeche base pe',
      icon: '⬇️'
    }]
  }, {
    id: 'cgls3',
    text: 'Kitne patte prabhavit hain?',
    hint: 'Infection ka phailaav',
    options: [{
      id: 'one',
      label: '1-2 patte',
      icon: '1️⃣'
    }, {
      id: 'few',
      label: '3-5 patte',
      icon: '🔢'
    }, {
      id: 'many',
      label: 'Adhe se zyada',
      icon: '📊'
    }, {
      id: 'all',
      label: 'Saare patte',
      icon: '⚠️'
    }]
  }, {
    id: 'cgls4',
    text: 'Daag ke aas-paas peela ring hai?',
    hint: 'Halo pattern check',
    options: [{
      id: 'yes',
      label: 'Haan, peela ring hai',
      icon: '🟡'
    }, {
      id: 'no',
      label: 'Nahi, sirf daag',
      icon: '❌'
    }, {
      id: 'wet',
      label: 'Geela aur dark border',
      icon: '💧'
    }, {
      id: 'dry',
      label: 'Sukha aur crispy',
      icon: '🍂'
    }]
  }, {
    id: 'cgls5',
    text: 'Pichle 2 hafte mein zyada nami ya baarish?',
    hint: 'Fungal trigger',
    options: [{
      id: 'heavy',
      label: 'Haan, bahut baarish',
      icon: '🌧️'
    }, {
      id: 'light',
      label: 'Thodi baarish',
      icon: '🌦️'
    }, {
      id: 'no',
      label: 'Bilkul sukha',
      icon: '☀️'
    }, {
      id: 'fog',
      label: 'Kohra/Fog tha',
      icon: '🌫️'
    }]
  }],
  'Gray Leaf Spot_multiple': [{
    id: 'cglsm1',
    text: 'Kitne patte ek saath prabhavit hain?',
    hint: 'Multiple infection scale',
    options: [{
      id: 'few',
      label: '3-5 patte',
      icon: '🔢'
    }, {
      id: 'many',
      label: '10+ patte',
      icon: '📊'
    }, {
      id: 'half',
      label: 'Adhi fasal',
      icon: '⚠️'
    }, {
      id: 'all',
      label: 'Poori fasal',
      icon: '💀'
    }]
  }, {
    id: 'cglsm2',
    text: 'Daag aapas mein jud rahe hain?',
    hint: 'Merging spots — serious sign',
    options: [{
      id: 'yes_merge',
      label: 'Haan, daag jud gaye',
      icon: '🔴'
    }, {
      id: 'spreading',
      label: 'Fail rahe hain',
      icon: '⚠️'
    }, {
      id: 'separate',
      label: 'Alag alag hain abhi',
      icon: '🟡'
    }, {
      id: 'no',
      label: 'Nahi',
      icon: '✅'
    }]
  }, {
    id: 'cglsm3',
    text: 'Infected patte gir bhi rahe hain?',
    hint: 'Defoliation check',
    options: [{
      id: 'yes',
      label: 'Haan, bahut gir rahe',
      icon: '🍂'
    }, {
      id: 'some',
      label: 'Kuch gir rahe',
      icon: '🌿'
    }, {
      id: 'no',
      label: 'Nahi gir rahe',
      icon: '✅'
    }, {
      id: 'dry',
      label: 'Sukh ke gir rahe',
      icon: '💀'
    }]
  }, {
    id: 'cglsm4',
    text: 'Pehle koi treatment kiya tha?',
    hint: 'Previous spray history',
    options: [{
      id: 'fungicide',
      label: 'Fungicide spray kiya',
      icon: '💊'
    }, {
      id: 'neem',
      label: 'Neem spray kiya',
      icon: '🌿'
    }, {
      id: 'nothing',
      label: 'Kuch nahi kiya',
      icon: '❌'
    }, {
      id: 'failed',
      label: 'Kiya par kaam nahi aaya',
      icon: '⚠️'
    }]
  }, {
    id: 'cglsm5',
    text: 'Aas-paas ke ped bhi prabhavit hain?',
    hint: 'Spread risk',
    options: [{
      id: 'yes_many',
      label: 'Haan, bahut pedo mein',
      icon: '⚠️'
    }, {
      id: 'yes_few',
      label: '1-2 pedo mein',
      icon: '🔢'
    }, {
      id: 'no',
      label: 'Sirf is ped mein',
      icon: '✅'
    }, {
      id: 'unknown',
      label: 'Check nahi kiya',
      icon: '❓'
    }]
  }],
  'Leaf rot': [{
    id: 'clr1',
    text: 'Patte kahan se galana shuru hua?',
    hint: 'Leaf rot starting point',
    options: [{
      id: 'tip',
      label: 'Nauk se shuru',
      icon: '📍'
    }, {
      id: 'edge',
      label: 'Kinare se',
      icon: '↔️'
    }, {
      id: 'middle',
      label: 'Beech se',
      icon: '🎯'
    }, {
      id: 'base',
      label: 'Base se',
      icon: '⬇️'
    }]
  }, {
    id: 'clr2',
    text: 'Gale hue hisse ka rang?',
    hint: 'Rot color identification',
    options: [{
      id: 'brown',
      label: 'Brown/Bhoora',
      icon: '🟤'
    }, {
      id: 'black',
      label: 'Kala',
      icon: '⚫'
    }, {
      id: 'gray',
      label: 'Bhoora-Gray',
      icon: '⬜'
    }, {
      id: 'yellow_brown',
      label: 'Peela-Bhoora',
      icon: '🟡'
    }]
  }, {
    id: 'clr3',
    text: 'Galane ki gati kaisi hai?',
    hint: 'Progression speed',
    options: [{
      id: 'fast',
      label: 'Tezi se fail raha',
      icon: '🏃'
    }, {
      id: 'slow',
      label: 'Dheere dheere',
      icon: '🐌'
    }, {
      id: 'stable',
      label: 'Ruka hua lagta hai',
      icon: '⏸️'
    }, {
      id: 'unknown',
      label: 'Pata nahi',
      icon: '❓'
    }]
  }, {
    id: 'clr4',
    text: 'Koi buri smell aa rahi hai?',
    hint: 'Bacterial vs fungal',
    options: [{
      id: 'yes_strong',
      label: 'Haan, bahut buri smell',
      icon: '👃'
    }, {
      id: 'yes_mild',
      label: 'Halki smell',
      icon: '😐'
    }, {
      id: 'no',
      label: 'Koi smell nahi',
      icon: '✅'
    }, {
      id: 'earthy',
      label: 'Mitti jaisi smell',
      icon: '🌍'
    }]
  }, {
    id: 'clr5',
    text: 'Patte mein paani ka bhoora daag hai?',
    hint: 'Water soaking check',
    options: [{
      id: 'yes',
      label: 'Haan, wet daag hain',
      icon: '💧'
    }, {
      id: 'dry_rot',
      label: 'Sukha rot hai',
      icon: '🍂'
    }, {
      id: 'both',
      label: 'Dono hain',
      icon: '🤔'
    }, {
      id: 'no',
      label: 'Normal lag raha',
      icon: '✅'
    }]
  }],
  'Leaf rot_multiple': [{
    id: 'clrm1',
    text: 'Kitni pattiyan ek saath gal rahi hain?',
    hint: 'Scale of infection',
    options: [{
      id: 'few',
      label: '3-5 patte',
      icon: '🔢'
    }, {
      id: 'many',
      label: '10+ patte',
      icon: '📊'
    }, {
      id: 'half',
      label: 'Adha ped',
      icon: '⚠️'
    }, {
      id: 'all',
      label: 'Poora ped',
      icon: '💀'
    }]
  }, {
    id: 'clrm2',
    text: 'Kali (bud) bhi prabhavit hai?',
    hint: 'Critical check — bud rot spreading?',
    options: [{
      id: 'yes_bud',
      label: 'Haan, kali bhi gal rahi',
      icon: '💀'
    }, {
      id: 'no_bud',
      label: 'Nahi, kali theek hai',
      icon: '✅'
    }, {
      id: 'unsure',
      label: 'Pata nahi',
      icon: '❓'
    }, {
      id: 'discolored',
      label: 'Kali ka rang badla',
      icon: '🟫'
    }]
  }, {
    id: 'clrm3',
    text: 'Pani bharav (waterlogging) ki problem hai?',
    hint: 'Root cause',
    options: [{
      id: 'yes',
      label: 'Haan, paani ruka rehta',
      icon: '🌊'
    }, {
      id: 'sometimes',
      label: 'Kabhi kabhi',
      icon: '🤔'
    }, {
      id: 'no',
      label: 'Nahi, drain ho jaata',
      icon: '✅'
    }, {
      id: 'recent',
      label: 'Haal hi mein baarish',
      icon: '🌧️'
    }]
  }, {
    id: 'clrm4',
    text: 'Trunk pe koi daag ya liquid?',
    hint: 'Check for stem bleeding too',
    options: [{
      id: 'yes_liquid',
      label: 'Haan, liquid nikal raha',
      icon: '🩸'
    }, {
      id: 'yes_spot',
      label: 'Haan, daag hain trunk pe',
      icon: '🟫'
    }, {
      id: 'no',
      label: 'Trunk theek lagta',
      icon: '✅'
    }, {
      id: 'bark',
      label: 'Chhal uth rahi hai',
      icon: '🌴'
    }]
  }, {
    id: 'clrm5',
    text: 'Yeh problem kab se hai?',
    hint: 'Duration',
    options: [{
      id: 'days',
      label: '2-3 din se',
      icon: '📅'
    }, {
      id: 'week',
      label: '1 hafte se',
      icon: '🗓️'
    }, {
      id: 'month',
      label: '1+ mahine se',
      icon: '📆'
    }, {
      id: 'old',
      label: '3+ mahine se',
      icon: '⏳'
    }]
  }],
  'bud rot': [{
    id: 'cbr1',
    text: 'Naariyal ki kali (growing tip) ka kya haal hai?',
    hint: 'Bud rot central symptom',
    options: [{
      id: 'brown_dead',
      label: 'Kali bhoori/kali ho gayi',
      icon: '💀'
    }, {
      id: 'wilting',
      label: 'Murjha rahi hai',
      icon: '🥀'
    }, {
      id: 'soft',
      label: 'Soft aur geeli lag rahi',
      icon: '💧'
    }, {
      id: 'smell',
      label: 'Buri smell aa rahi',
      icon: '👃'
    }]
  }, {
    id: 'cbr2',
    text: 'Kali khinchne pe kya hota hai?',
    hint: 'Pull test for bud rot',
    options: [{
      id: 'comes_out',
      label: 'Aasaani se nikal aati',
      icon: '⚠️'
    }, {
      id: 'stuck',
      label: 'Atki hui hai',
      icon: '✅'
    }, {
      id: 'broken',
      label: 'Toot jaati hai',
      icon: '💔'
    }, {
      id: 'not_tried',
      label: 'Check nahi kiya',
      icon: '❓'
    }]
  }, {
    id: 'cbr3',
    text: 'Inner leaves (andar ke patte) ka rang?',
    hint: 'Early bud rot sign',
    options: [{
      id: 'yellow',
      label: 'Peele pad rahe',
      icon: '🟡'
    }, {
      id: 'brown',
      label: 'Bhoore ho gaye',
      icon: '🟤'
    }, {
      id: 'normal',
      label: 'Normal hare',
      icon: '✅'
    }, {
      id: 'rotting',
      label: 'Galne lage',
      icon: '💀'
    }]
  }, {
    id: 'cbr4',
    text: 'Pichle mahine mein zyada baarish ya nami?',
    hint: 'Phytophthora trigger',
    options: [{
      id: 'heavy',
      label: 'Bahut zyada baarish',
      icon: '🌧️'
    }, {
      id: 'humid',
      label: 'Nami bahut thi',
      icon: '💧'
    }, {
      id: 'normal',
      label: 'Normal mausam',
      icon: '🌤️'
    }, {
      id: 'dry',
      label: 'Sukha mausam',
      icon: '☀️'
    }]
  }, {
    id: 'cbr5',
    text: 'Aas-paas ke naariyal pedo mein bhi?',
    hint: 'Spread check — urgent',
    options: [{
      id: 'yes_many',
      label: 'Haan, kai pedo mein',
      icon: '🚨'
    }, {
      id: 'yes_one',
      label: '1-2 pedo mein',
      icon: '⚠️'
    }, {
      id: 'no',
      label: 'Sirf is ped mein',
      icon: '✅'
    }, {
      id: 'check',
      label: 'Abhi check karunga',
      icon: '🔍'
    }]
  }],
  'stem bleeding': [{
    id: 'csb1',
    text: 'Trunk se kaunsa liquid nikal raha hai?',
    hint: 'Stem bleeding identification',
    options: [{
      id: 'dark_red',
      label: 'Dark red/maroon liquid',
      icon: '🩸'
    }, {
      id: 'brown',
      label: 'Brown ranga liquid',
      icon: '🟤'
    }, {
      id: 'black',
      label: 'Kala chipchipa',
      icon: '⚫'
    }, {
      id: 'clear',
      label: 'Transparent/Paani',
      icon: '💧'
    }]
  }, {
    id: 'csb2',
    text: 'Trunk pe daag ya kharaabi kahan hai?',
    hint: 'Location of bleeding',
    options: [{
      id: 'base',
      label: 'Neeche (jaad ke paas)',
      icon: '⬇️'
    }, {
      id: 'middle',
      label: 'Beech mein',
      icon: '🎯'
    }, {
      id: 'top',
      label: 'Upar (crown ke paas)',
      icon: '⬆️'
    }, {
      id: 'all',
      label: 'Poore trunk pe',
      icon: '🌴'
    }]
  }, {
    id: 'csb3',
    text: 'Trunk pe daag ka size kaisa hai?',
    hint: 'Size of affected area',
    options: [{
      id: 'small',
      label: 'Chota (10cm se kam)',
      icon: '🔵'
    }, {
      id: 'medium',
      label: 'Medium (10-30cm)',
      icon: '🟡'
    }, {
      id: 'large',
      label: 'Bada (30cm+)',
      icon: '🔴'
    }, {
      id: 'ring',
      label: 'Poore trunk ka chakkar',
      icon: '⭕'
    }]
  }, {
    id: 'csb4',
    text: 'Affected jagah ki chhal (bark) kaisi lag rahi?',
    hint: 'Bark condition',
    options: [{
      id: 'soft',
      label: 'Soft aur sunken',
      icon: '💧'
    }, {
      id: 'cracked',
      label: 'Phaati hui',
      icon: '💔'
    }, {
      id: 'peeling',
      label: 'Uth rahi hai',
      icon: '🌿'
    }, {
      id: 'normal',
      label: 'Theek lagti',
      icon: '✅'
    }]
  }, {
    id: 'csb5',
    text: 'Ped ka production prabhavit hua hai?',
    hint: 'Yield impact',
    options: [{
      id: 'less',
      label: 'Haan, naariyal kam aaye',
      icon: '📉'
    }, {
      id: 'none',
      label: 'Bilkul nahi aaye',
      icon: '❌'
    }, {
      id: 'normal',
      label: 'Normal production hai',
      icon: '✅'
    }, {
      id: 'young',
      label: 'Abhi young ped hai',
      icon: '🌱'
    }]
  }],
  'stem bleeding_multiple': [{
    id: 'csbm1',
    text: 'Trunk pe kitni jagah se bleeding ho rahi?',
    hint: 'Multiple sites = serious',
    options: [{
      id: 'two',
      label: '2 jagah',
      icon: '2️⃣'
    }, {
      id: 'three',
      label: '3-4 jagah',
      icon: '🔢'
    }, {
      id: 'many',
      label: '5+ jagah',
      icon: '⚠️'
    }, {
      id: 'ring',
      label: 'Poore trunk pe',
      icon: '🚨'
    }]
  }, {
    id: 'csbm2',
    text: 'Aas-paas ke pedo mein bhi yahi problem?',
    hint: 'Farm-wide assessment',
    options: [{
      id: 'yes_many',
      label: 'Haan, kai pedo mein',
      icon: '🚨'
    }, {
      id: 'yes_few',
      label: '1-2 pedo mein',
      icon: '⚠️'
    }, {
      id: 'no',
      label: 'Sirf is ped mein',
      icon: '✅'
    }, {
      id: 'new',
      label: 'Naye pedo mein bhi',
      icon: '⚡'
    }]
  }, {
    id: 'csbm3',
    text: 'Kya pehle bhi yeh ped beemar tha?',
    hint: 'Chronic vs new infection',
    options: [{
      id: 'yes_old',
      label: 'Haan, pehle bhi tha',
      icon: '📆'
    }, {
      id: 'first',
      label: 'Pehli baar dekha',
      icon: '📍'
    }, {
      id: 'partial',
      label: 'Pehle thoda tha',
      icon: '🤔'
    }, {
      id: 'unknown',
      label: 'Pata nahi',
      icon: '❓'
    }]
  }, {
    id: 'csbm4',
    text: 'Koi treatment pehle try kiya?',
    hint: 'Treatment history',
    options: [{
      id: 'yes_success',
      label: 'Haan, kuch faayda hua',
      icon: '✅'
    }, {
      id: 'yes_fail',
      label: 'Kiya par kaam nahi aaya',
      icon: '❌'
    }, {
      id: 'nothing',
      label: 'Kuch nahi kiya abhi tak',
      icon: '🚨'
    }, {
      id: 'natural',
      label: 'Natural/organic try kiya',
      icon: '🌿'
    }]
  }, {
    id: 'csbm5',
    text: 'Ped kitne saalon ka hai?',
    hint: 'Age affects treatment',
    options: [{
      id: 'young',
      label: '3-5 saal',
      icon: '🌱'
    }, {
      id: 'adult',
      label: '6-15 saal',
      icon: '🌴'
    }, {
      id: 'mature',
      label: '15-30 saal',
      icon: '🎋'
    }, {
      id: 'old',
      label: '30+ saal',
      icon: '🧓'
    }]
  }],
  'Healthy': [{
    id: 'ch1',
    text: 'Yeh check kyun kar rahe hain?',
    hint: 'Preventive monitoring reason',
    options: [{
      id: 'routine',
      label: 'Routine check',
      icon: '📋'
    }, {
      id: 'worried',
      label: 'Thoda worried hoon',
      icon: '😟'
    }, {
      id: 'nearby',
      label: 'Paas ke ped mein bimari',
      icon: '⚠️'
    }, {
      id: 'expert',
      label: 'Expert ne kaha',
      icon: '👨‍⚕️'
    }]
  }, {
    id: 'ch2',
    text: 'Naariyal ke patte kaisa dikhte hain?',
    hint: 'Leaf health check',
    options: [{
      id: 'bright',
      label: 'Chamakdar hare',
      icon: '✅'
    }, {
      id: 'dull',
      label: 'Matteese hare',
      icon: '🌿'
    }, {
      id: 'slight_yellow',
      label: 'Thoda peela',
      icon: '🟡'
    }, {
      id: 'very_green',
      label: 'Bahut ghane hare',
      icon: '🌳'
    }]
  }, {
    id: 'ch3',
    text: 'Production kaisi hai?',
    hint: 'Yield check',
    options: [{
      id: 'good',
      label: 'Bahut achhi',
      icon: '✅'
    }, {
      id: 'normal',
      label: 'Normal',
      icon: '👍'
    }, {
      id: 'less',
      label: 'Pehle se kam',
      icon: '📉'
    }, {
      id: 'none',
      label: 'Bilkul nahi',
      icon: '❌'
    }]
  }, {
    id: 'ch4',
    text: 'Last fertilizer kab diya?',
    hint: 'Nutrition check',
    options: [{
      id: 'recent',
      label: '1 mahine mein',
      icon: '📅'
    }, {
      id: 'months',
      label: '3-6 mahine mein',
      icon: '🗓️'
    }, {
      id: 'long',
      label: '6+ mahine pehle',
      icon: '📆'
    }, {
      id: 'never',
      label: 'Kabhi nahi',
      icon: '❌'
    }]
  }, {
    id: 'ch5',
    text: 'Koi keede ya pest dikhe hain?',
    hint: 'Pest monitoring',
    options: [{
      id: 'rhinoceros',
      label: 'Bada kala beetle',
      icon: '🦏'
    }, {
      id: 'red_palm',
      label: 'Lal rang ka ghun',
      icon: '🐛'
    }, {
      id: 'none',
      label: 'Koi nahi dikha',
      icon: '✅'
    }, {
      id: 'small',
      label: 'Chote keede',
      icon: '🔬'
    }]
  }]
};
const COCONUT_Q1_DATA = {
  id: 'cq1',
  text: 'Naariyal ke ped mein kya problem dikh rahi hai?',
  hint: 'Jo sabse pehle aapne notice kiya woh select karein',
  options: [{
    id: 'coconut_spots',
    label: 'Patte pe daag / Leaf Spots',
    icon: '🔴',
    desc: 'Gray Leaf Spot ya Leaf Rot ke symptoms'
  }, {
    id: 'coconut_wilt',
    label: 'Tanaa ya Bud mein pareshani',
    icon: '🥀',
    desc: 'Stem Bleeding ya Bud Rot ke signs'
  }, {
    id: 'coconut_yellow',
    label: 'Patte peele ho rahe hain',
    icon: '🟡',
    desc: 'Yellowing — Nutrient ya disease'
  }, {
    id: 'coconut_none',
    label: 'Koi dikhi problem nahi',
    icon: '✅',
    desc: 'Preventive check — healthy dikhta hai'
  }]
};
const Q1_DATA = {
  id: 'q1',
  text: 'Fasal mein kya problem dikhi hai?',
  hint: 'Sabse pehle jo problem dikha woh select karein',
  options: [{
    id: 'spots',
    label: 'Daag / Spots',
    icon: '🔴',
    desc: 'Patte ya tanay pe daag hain'
  }, {
    id: 'wilt',
    label: 'Murjhana / Wilting',
    icon: '🥀',
    desc: 'Paudha murjhaya hua hai'
  }, {
    id: 'yellow',
    label: 'Peele Patte',
    icon: '🟡',
    desc: 'Patte peele ho rahe hain'
  }, {
    id: 'none',
    label: 'Koi Problem Nahi',
    icon: '✅',
    desc: 'Preventive check chahiye'
  }]
};
const BRANCH_QS = {
  spots: [{
    id: 'q2',
    text: 'Daag kahan hain?',
    hint: 'Sabse zyada kahan dikhe?',
    options: [{
      id: 'lower',
      label: 'Neeche ke patte',
      icon: '⬇️'
    }, {
      id: 'upper',
      label: 'Upar ke patte',
      icon: '⬆️'
    }, {
      id: 'all',
      label: 'Poori fasal',
      icon: '🌿'
    }, {
      id: 'stem',
      label: 'Tanaa / Stem',
      icon: '🌵'
    }]
  }, {
    id: 'q3',
    text: 'Daag ka rang kya hai?',
    hint: 'Ek choose karein',
    options: [{
      id: 'brown',
      label: 'Bhoora / Brown',
      icon: '🟤'
    }, {
      id: 'yellow',
      label: 'Peela / Yellow',
      icon: '🟡'
    }, {
      id: 'black',
      label: 'Kaala / Black',
      icon: '⚫'
    }, {
      id: 'red',
      label: 'Laal / Red',
      icon: '🔴'
    }]
  }, {
    id: 'q4',
    text: 'Daag ka aakar kaisa hai?',
    hint: 'Shape describe karein',
    options: [{
      id: 'round',
      label: 'Gol / Round',
      icon: '⭕'
    }, {
      id: 'irreg',
      label: 'Irregular',
      icon: '🔶'
    }, {
      id: 'stripe',
      label: 'Dhabbedaar',
      icon: '🫧'
    }, {
      id: 'ring',
      label: 'Ring shape',
      icon: '🎯'
    }]
  }, {
    id: 'q9',
    text: 'Pichle hafte baarish aayi thi?',
    hint: 'Moisture level samajhna zaroori hai',
    options: [{
      id: 'heavy',
      label: 'Haan, zyada baarish',
      icon: '🌧️'
    }, {
      id: 'light',
      label: 'Thodi si baarish',
      icon: '🌦️'
    }, {
      id: 'no',
      label: 'Bilkul nahi',
      icon: '☀️'
    }, {
      id: 'fog',
      label: 'Fog / Kuhasa tha',
      icon: '🌫️'
    }]
  }, {
    id: 'q20',
    text: 'Pichli spray kab ki thi?',
    hint: 'Fungicide / pesticide schedule',
    options: [{
      id: 'recent',
      label: '3 din se kam',
      icon: '💊'
    }, {
      id: 'week',
      label: '1 hafte mein',
      icon: '📅'
    }, {
      id: 'old',
      label: '2+ hafte pehle',
      icon: '⏳'
    }, {
      id: 'never',
      label: 'Kabhi nahi',
      icon: '❌'
    }]
  }],
  wilt: [{
    id: 'q7',
    text: 'Paudha kab murjhata hai?',
    hint: 'Timing pattern batayein',
    options: [{
      id: 'morning',
      label: 'Subah theek, shaam murjhata',
      icon: '🌅'
    }, {
      id: 'always',
      label: 'Hamesha murjhaya',
      icon: '💧'
    }, {
      id: 'heat',
      label: 'Sirf tej dhoop mein',
      icon: '☀️'
    }, {
      id: 'water',
      label: 'Paani dene ke baad',
      icon: '💦'
    }]
  }, {
    id: 'q9',
    text: 'Pichle hafte baarish ya zyada paani?',
    hint: 'Waterlogging check',
    options: [{
      id: 'heavy',
      label: 'Zyada baarish',
      icon: '🌧️'
    }, {
      id: 'normal',
      label: 'Normal',
      icon: '🌤️'
    }, {
      id: 'no',
      label: 'Bilkul sukha',
      icon: '🏜️'
    }, {
      id: 'fog',
      label: 'Fog / Nami',
      icon: '🌫️'
    }]
  }, {
    id: 'q15',
    text: 'Paani kitni baar dete hain?',
    hint: 'Irrigation frequency',
    options: [{
      id: 'daily',
      label: 'Roz',
      icon: '💦'
    }, {
      id: 'alt',
      label: 'Ek din chhodkar',
      icon: '📆'
    }, {
      id: 'twice',
      label: 'Hafte mein 2 baar',
      icon: '🗓️'
    }, {
      id: 'once',
      label: 'Hafte mein 1 baar',
      icon: '📅'
    }]
  }, {
    id: 'q16',
    text: 'Khet mein paani ruka rehta hai?',
    hint: 'Drainage problem?',
    options: [{
      id: 'yes',
      label: 'Haan, bahut rukta hai',
      icon: '🌊'
    }, {
      id: 'sometimes',
      label: 'Kabhi kabhi',
      icon: '🤔'
    }, {
      id: 'no',
      label: 'Nahi, drain ho jaata',
      icon: '✅'
    }, {
      id: 'dk',
      label: 'Pata nahi',
      icon: '❓'
    }]
  }, {
    id: 'q30',
    text: 'Problem kab se shuru hui?',
    hint: 'Timeline',
    options: [{
      id: 'today',
      label: 'Aaj se / 1-2 din',
      icon: '📍'
    }, {
      id: 'week',
      label: 'Is hafte',
      icon: '📅'
    }, {
      id: 'twoweek',
      label: '2 hafte se',
      icon: '🗓️'
    }, {
      id: 'month',
      label: '1+ mahine se',
      icon: '⏳'
    }]
  }],
  yellow: [{
    id: 'q5',
    text: 'Patte girr bhi rahe hain?',
    hint: 'Leaf drop pattern',
    options: [{
      id: 'heavy',
      label: 'Bahut zyada gir rahe',
      icon: '🍂'
    }, {
      id: 'some',
      label: 'Thode gir rahe',
      icon: '🍁'
    }, {
      id: 'no',
      label: 'Nahi gir rahe',
      icon: '🌿'
    }, {
      id: 'tip',
      label: 'Sirf tip se peele',
      icon: '🌱'
    }]
  }, {
    id: 'q10',
    text: 'Humidity / Nami kaisi rehti hai?',
    hint: '',
    options: [{
      id: 'high',
      label: 'Bahut zyada nami',
      icon: '💧'
    }, {
      id: 'med',
      label: 'Normal',
      icon: '🌡️'
    }, {
      id: 'low',
      label: 'Bahut sukha',
      icon: '🏜️'
    }, {
      id: 'variable',
      label: 'Badlti rehti hai',
      icon: '🌤️'
    }]
  }, {
    id: 'q21',
    text: 'Pichla fertilizer konsa diya tha?',
    hint: 'Poshan deficiency check',
    options: [{
      id: 'urea',
      label: 'Urea (Neel)',
      icon: '🧪'
    }, {
      id: 'dap',
      label: 'DAP',
      icon: '⚗️'
    }, {
      id: 'npk',
      label: 'NPK Mix',
      icon: '🌱'
    }, {
      id: 'none',
      label: 'Kuch nahi diya',
      icon: '❌'
    }]
  }, {
    id: 'q22',
    text: 'Fertilizer kab diya tha?',
    hint: 'Timing matters',
    options: [{
      id: 'recent',
      label: '3 din mein',
      icon: '📍'
    }, {
      id: 'week',
      label: 'Is hafte',
      icon: '📅'
    }, {
      id: 'old',
      label: '2+ hafte pehle',
      icon: '⏳'
    }, {
      id: 'never',
      label: 'Kabhi nahi',
      icon: '❌'
    }]
  }, {
    id: 'q18',
    text: 'Soil / Mitti ka test karaya?',
    hint: 'pH aur NPK levels',
    options: [{
      id: 'recent',
      label: 'Haan, haal hi mein',
      icon: '✅'
    }, {
      id: 'old',
      label: 'Puraana test tha',
      icon: '📋'
    }, {
      id: 'no',
      label: 'Nahi karaya',
      icon: '❌'
    }, {
      id: 'plan',
      label: 'Plan hai',
      icon: '📝'
    }]
  }],
  none: [{
    id: 'q25',
    text: 'Fasal kitne din purani hai?',
    hint: 'Growth stage samajhna',
    options: [{
      id: 'seed',
      label: '0–30 din (Seedling)',
      icon: '🌱'
    }, {
      id: 'veg',
      label: '30–60 din (Vegetative)',
      icon: '🌿'
    }, {
      id: 'flower',
      label: '60–90 din (Flowering)',
      icon: '🌸'
    }, {
      id: 'fruit',
      label: '90+ din (Fruiting)',
      icon: '🍅'
    }]
  }, {
    id: 'q29',
    text: 'Fasal abhi kis stage mein hai?',
    hint: '',
    options: [{
      id: 'seed',
      label: 'Seedling',
      icon: '🌱'
    }, {
      id: 'veg',
      label: 'Vegetative',
      icon: '🌿'
    }, {
      id: 'flower',
      label: 'Flowering',
      icon: '🌸'
    }, {
      id: 'fruit',
      label: 'Fruiting / Maturing',
      icon: '🍎'
    }]
  }, {
    id: 'q11',
    text: 'Fasal ko poori dhoop milti hai?',
    hint: 'Sunlight requirement',
    options: [{
      id: 'full',
      label: 'Haan, 8+ ghante',
      icon: '☀️'
    }, {
      id: 'partial',
      label: '4–6 ghante',
      icon: '⛅'
    }, {
      id: 'shade',
      label: 'Zyada chhhav',
      icon: '🌥️'
    }, {
      id: 'vary',
      label: 'Badlta rehta hai',
      icon: '🔄'
    }]
  }, {
    id: 'q27',
    text: 'Konsa beej / variety use kiya?',
    hint: 'Disease resistance check',
    options: [{
      id: 'hybrid',
      label: 'Hybrid variety',
      icon: '🧬'
    }, {
      id: 'local',
      label: 'Local / Desi beej',
      icon: '🌾'
    }, {
      id: 'certified',
      label: 'Certified seed',
      icon: '📜'
    }, {
      id: 'dk',
      label: 'Pata nahi',
      icon: '🤷'
    }]
  }, {
    id: 'q28',
    text: 'Aas-paas ke khet mein bhi koi problem?',
    hint: 'Outbreak detection',
    options: [{
      id: 'yes',
      label: 'Haan, unhe bhi hai',
      icon: '⚠️'
    }, {
      id: 'no',
      label: 'Nahi, theek hain',
      icon: '✅'
    }, {
      id: 'dk',
      label: 'Check nahi kiya',
      icon: '🤔'
    }, {
      id: 'some',
      label: 'Kuch khet mein hai',
      icon: '🔍'
    }]
  }],
  /* ── COCONUT-ONLY QUESTIONS (30 total, 5 shown per session) ── */

  /* Branch A: Daag/Spots → Gray Leaf Spot ya Leaf Rot */
  coconut_spots: [{
    id: 'cq2',
    text: 'Daag kahan nazar aa rahe hain?',
    hint: 'Patte ka kaunsa hissa zyada prabhavit hai?',
    options: [{
      id: 'c_tip',
      label: 'Patte ki nauk pe',
      icon: '📍',
      desc: 'Tip se shuru hota hai'
    }, {
      id: 'c_middle',
      label: 'Beech mein',
      icon: '🎯',
      desc: 'Patte ke darmiyan'
    }, {
      id: 'c_all',
      label: 'Poore patte pe',
      icon: '🍃',
      desc: 'Pura patta prabhavit'
    }, {
      id: 'c_frond',
      label: 'Poori frond/shaakh pe',
      icon: '🌿',
      desc: 'Puri shaakh kharab'
    }]
  }, {
    id: 'cq3',
    text: 'Daag ka rang kaisa hai?',
    hint: 'Sabse sahi description choose karein',
    options: [{
      id: 'c_gray',
      label: 'Bhoora-Bhura / Grayish',
      icon: '⬜',
      desc: 'Gray Leaf Spot ka sign'
    }, {
      id: 'c_brown',
      label: 'Koyla Bhoora / Dark Brown',
      icon: '🟤',
      desc: 'Leaf rot ka sign'
    }, {
      id: 'c_yellow_halo',
      label: 'Peele ring ke saath daag',
      icon: '🟡',
      desc: 'Fungal infection'
    }, {
      id: 'c_black',
      label: 'Kaale daag',
      icon: '⚫',
      desc: 'Advanced infection'
    }]
  }, {
    id: 'cq4',
    text: 'Daag ka aakar kaisa hai?',
    hint: 'Shape closely dekho',
    options: [{
      id: 'c_oval',
      label: 'Oval / Lamba',
      icon: '⭕',
      desc: 'Pestalotiopsis sign'
    }, {
      id: 'c_irreg',
      label: 'Irregular shape',
      icon: '🔶',
      desc: 'Phytophthora sign'
    }, {
      id: 'c_stripe',
      label: 'Lambi pattiyan / Strips',
      icon: '🫧',
      desc: 'Bacterial lesion'
    }, {
      id: 'c_concentric',
      label: 'Gol rings mein',
      icon: '🎯',
      desc: 'Fungal bull-eye'
    }]
  }, {
    id: 'cq5',
    text: 'Patte ke aandar koi powder ya fungus dikhti hai?',
    hint: 'Patte palat ke dekho',
    options: [{
      id: 'c_powder_yes',
      label: 'Haan, safed powder',
      icon: '⬜',
      desc: 'Powdery mildew'
    }, {
      id: 'c_spores',
      label: 'Haan, kaale spore',
      icon: '⚫',
      desc: 'Fungal sporulation'
    }, {
      id: 'c_none',
      label: 'Nahi, sirf daag',
      icon: '✅',
      desc: 'Leaf spot only'
    }, {
      id: 'c_wet',
      label: 'Geela ya water-soaked',
      icon: '💧',
      desc: 'Bacterial/fungal'
    }]
  }, {
    id: 'cq6',
    text: 'Kitne patte prabhavit hain?',
    hint: 'Spread ka andaza lagao',
    options: [{
      id: 'c_few',
      label: '1-2 patte',
      icon: '🍃',
      desc: 'Abhi shuruat'
    }, {
      id: 'c_some',
      label: '5-10 patte',
      icon: '🌿',
      desc: 'Moderate spread'
    }, {
      id: 'c_half',
      label: 'Aadha ped',
      icon: '🌴',
      desc: 'Serious problem'
    }, {
      id: 'c_all_tree',
      label: 'Poora ped kharab',
      icon: '🆘',
      desc: 'Critical — turant karein'
    }]
  }],
  /* Branch B: Murjhana/Wilt → Bud Rot ya Stem Bleeding */
  coconut_wilt: [{
    id: 'cq7',
    text: 'Kaunsa hissa sabse zyada prabhavit hai?',
    hint: 'Dhyan se pehchano',
    options: [{
      id: 'c_topbud',
      label: 'Upar wali kali (Bud)',
      icon: '🌱',
      desc: 'Bud rot ka sign'
    }, {
      id: 'c_trunk',
      label: 'Tanaa / Trunk',
      icon: '🌵',
      desc: 'Stem bleeding'
    }, {
      id: 'c_roots',
      label: 'Jadein / Roots',
      icon: '🌳',
      desc: 'Root rot'
    }, {
      id: 'c_crown',
      label: 'Crown — ped ka dil',
      icon: '👑',
      desc: 'Crown rot'
    }]
  }, {
    id: 'cq8',
    text: 'Tanay ya bud se koi liquid (ras) nikalta hai?',
    hint: 'Tanaa closely check karo',
    options: [{
      id: 'c_dark_liquid',
      label: 'Haan, kaala/bhoora liquid',
      icon: '🖤',
      desc: 'Stem bleeding — serious'
    }, {
      id: 'c_sticky',
      label: 'Haan, chipchipa ras',
      icon: '🍯',
      desc: 'Fungal exudate'
    }, {
      id: 'c_none_liquid',
      label: 'Nahi, koi liquid nahi',
      icon: '❌',
      desc: 'Physical damage'
    }, {
      id: 'c_smell',
      label: 'Badboo aati hai',
      icon: '👃',
      desc: 'Bacterial rot'
    }]
  }, {
    id: 'cq9',
    text: 'Nayi pattiyaan kaise dikhti hain?',
    hint: 'Sabse upar ke naye patte dekho',
    options: [{
      id: 'c_brown_young',
      label: 'Bhoori ho gayi — paghlti hain',
      icon: '🟤',
      desc: 'Bud rot classic sign'
    }, {
      id: 'c_pale',
      label: 'Pale/Peeli aur kamzor',
      icon: '🟡',
      desc: 'Nutrient deficiency'
    }, {
      id: 'c_normal_but_wilt',
      label: 'Theek lagti hai phir bhi murjhati',
      icon: '🥀',
      desc: 'Root/vascular issue'
    }, {
      id: 'c_no_new',
      label: 'Nayi pattiyaan aa hi nahi raheen',
      icon: '⛔',
      desc: 'Severe bud damage'
    }]
  }, {
    id: 'cq10',
    text: 'Kab se yeh problem shuru hui?',
    hint: 'Timeline batao',
    options: [{
      id: 'c_days',
      label: '2-3 din pehle',
      icon: '📅',
      desc: 'Early stage'
    }, {
      id: 'c_week',
      label: '1-2 hafte pehle',
      icon: '🗓️',
      desc: 'Progressed'
    }, {
      id: 'c_month',
      label: '1 mahina pehle',
      icon: '📆',
      desc: 'Chronic issue'
    }, {
      id: 'c_sudden',
      label: 'Achanak hua',
      icon: '⚡',
      desc: 'Acute onset'
    }]
  }, {
    id: 'cq11',
    text: 'Kya aas-paas ke naariyal ke ped bhi prabhavit hain?',
    hint: 'Poora bagaan dekho',
    options: [{
      id: 'c_spread_yes',
      label: 'Haan, 2-3 ped aur bhi',
      icon: '⚠️',
      desc: 'Epidemic risk'
    }, {
      id: 'c_spread_many',
      label: 'Haan, bahut saare ped',
      icon: '🆘',
      desc: 'Critical spread'
    }, {
      id: 'c_only_one',
      label: 'Sirf yeh ek ped',
      icon: '🎯',
      desc: 'Isolated case'
    }, {
      id: 'c_nearby',
      label: 'Paas ka ek ped',
      icon: '👀',
      desc: 'Early spread'
    }]
  }],
  /* Branch C: Peele Patte → Yellowing / Nutrient Issues */
  coconut_yellow: [{
    id: 'cq12',
    text: 'Peele patte kahan se shuru ho rahe hain?',
    hint: 'Kaunse patte pehle peele hue?',
    options: [{
      id: 'c_old_first',
      label: 'Purane/Neeche ke patte pehle',
      icon: '⬇️',
      desc: 'Nitrogen deficiency'
    }, {
      id: 'c_young_first',
      label: 'Nayi pattiyaan pehle peeli',
      icon: '⬆️',
      desc: 'Iron/Zinc deficiency'
    }, {
      id: 'c_all_same',
      label: 'Sab ek saath peele',
      icon: '🌿',
      desc: 'Root/stem issue'
    }, {
      id: 'c_frond_tip',
      label: 'Frond ki nauk se shuru',
      icon: '📍',
      desc: 'Potassium deficiency'
    }]
  }, {
    id: 'cq13',
    text: 'Peelapan ka rang kaisa hai?',
    hint: 'Color closely dekho',
    options: [{
      id: 'c_bright_y',
      label: 'Chamkila Peela',
      icon: '💛',
      desc: 'Nutrient deficiency'
    }, {
      id: 'c_pale_green',
      label: 'Halkaa Peela-Hara',
      icon: '🟢',
      desc: 'Mild chlorosis'
    }, {
      id: 'c_bronze',
      label: 'Kaansi/Bronze rang',
      icon: '🥉',
      desc: 'Potassium/Mn deficiency'
    }, {
      id: 'c_necrotic',
      label: 'Peele mein bhoora/dead',
      icon: '🟤',
      desc: 'Advanced damage'
    }]
  }, {
    id: 'cq14',
    text: 'Kya pichle 3 mahine mein khad (fertilizer) diya tha?',
    hint: 'Nutrient management check',
    options: [{
      id: 'c_fert_yes',
      label: 'Haan, niyamit diya',
      icon: '✅',
      desc: 'Rule out deficiency'
    }, {
      id: 'c_fert_partial',
      label: 'Kabhi kabhi',
      icon: '⚠️',
      desc: 'Partial deficiency'
    }, {
      id: 'c_fert_no',
      label: 'Nahi, bilkul nahi',
      icon: '❌',
      desc: 'Likely deficiency'
    }, {
      id: 'c_fert_excess',
      label: 'Zyada diya shayad',
      icon: '⬆️',
      desc: 'Toxicity possible'
    }]
  }, {
    id: 'cq15',
    text: 'Zameen (soil) kaisi hai aapke naariyal ke bagaan ki?',
    hint: 'Soil type identify karo',
    options: [{
      id: 'c_sandy',
      label: 'Reti (Sandy)',
      icon: '🏖️',
      desc: 'Low nutrient retention'
    }, {
      id: 'c_loamy',
      label: 'Domatii (Loamy)',
      icon: '🌱',
      desc: 'Good soil'
    }, {
      id: 'c_clay',
      label: 'Chiknii Maitti (Clay)',
      icon: '🧱',
      desc: 'Drainage issue'
    }, {
      id: 'c_saline',
      label: 'Khaari (Saline/Salty)',
      icon: '🧂',
      desc: 'Salt stress — coastal'
    }]
  }, {
    id: 'cq16',
    text: 'Kya tree ki jadein (roots) ko paani mein rehna padta hai?',
    hint: 'Waterlogging check',
    options: [{
      id: 'c_water_yes',
      label: 'Haan, barish mein paani rukta hai',
      icon: '💧',
      desc: 'Root rot risk'
    }, {
      id: 'c_water_no',
      label: 'Nahi, achha drainage hai',
      icon: '✅',
      desc: 'Not waterlogging'
    }, {
      id: 'c_dry',
      label: 'Ulta — bahut sukha rehta hai',
      icon: '🏜️',
      desc: 'Drought stress'
    }, {
      id: 'c_seasonal',
      label: 'Mausam ke hisaab se',
      icon: '🌦️',
      desc: 'Seasonal issue'
    }]
  }],
  /* Branch D: Koi problem nahi / Preventive → General coconut health */
  coconut_none: [{
    id: 'cq17',
    text: 'Naariyal ka ped kitne saal purana hai?',
    hint: 'Age se disease risk samjha jaata hai',
    options: [{
      id: 'c_young',
      label: '1-3 saal (Navjaat)',
      icon: '🌱',
      desc: 'Young tree — more vulnerable'
    }, {
      id: 'c_mid',
      label: '4-10 saal',
      icon: '🌴',
      desc: 'Growing phase'
    }, {
      id: 'c_mature',
      label: '10-20 saal',
      icon: '🥥',
      desc: 'Productive age'
    }, {
      id: 'c_old',
      label: '20+ saal',
      icon: '🌳',
      desc: 'Old tree care needed'
    }]
  }, {
    id: 'cq18',
    text: 'Pichle season mein koi bimari thi?',
    hint: 'History check karo',
    options: [{
      id: 'c_hist_none',
      label: 'Bilkul nahi tha',
      icon: '✅',
      desc: 'Healthy history'
    }, {
      id: 'c_hist_minor',
      label: 'Thodi bimari thi',
      icon: '⚠️',
      desc: 'Watch carefully'
    }, {
      id: 'c_hist_major',
      label: 'Badi bimari thi',
      icon: '🆘',
      desc: 'High recurrence risk'
    }, {
      id: 'c_hist_spray',
      label: 'Spray kiya tha — theek hua',
      icon: '💊',
      desc: 'Treatment worked'
    }]
  }, {
    id: 'cq19',
    text: 'Aapka bagaan kahan hai?',
    hint: 'Location se disease risk samjha jaata hai',
    options: [{
      id: 'c_coastal',
      label: 'Samundar ke paas (Coastal)',
      icon: '🌊',
      desc: 'High humidity — fungal risk'
    }, {
      id: 'c_inland',
      label: 'Andar desh (Inland)',
      icon: '🏞️',
      desc: 'Moderate risk'
    }, {
      id: 'c_hilly',
      label: 'Pahadi ilaaka',
      icon: '⛰️',
      desc: 'Cool — different diseases'
    }, {
      id: 'c_backwater',
      label: 'Nadi/taalaab ke paas',
      icon: '🌊',
      desc: 'High moisture risk'
    }]
  }, {
    id: 'cq20',
    text: 'Abhi kaunsa mausam chal raha hai?',
    hint: 'Season disease pattern affect karta hai',
    options: [{
      id: 'c_monsoon',
      label: 'Barsaat / Monsoon',
      icon: '🌧️',
      desc: 'Highest fungal risk'
    }, {
      id: 'c_winter',
      label: 'Sardi / Winter',
      icon: '❄️',
      desc: 'Moderate risk'
    }, {
      id: 'c_summer',
      label: 'Garmi / Summer',
      icon: '☀️',
      desc: 'Drought stress risk'
    }, {
      id: 'c_preM',
      label: 'Barsaat se pehle',
      icon: '⛅',
      desc: 'Prevention time'
    }]
  }, {
    id: 'cq21',
    text: 'Aakhri baar spray kab kiya tha?',
    hint: 'Chemical history important hai',
    options: [{
      id: 'c_sp_week',
      label: 'Is hafte',
      icon: '✅',
      desc: 'Recent — good'
    }, {
      id: 'c_sp_month',
      label: 'Pichle mahine',
      icon: '⚠️',
      desc: 'Due soon'
    }, {
      id: 'c_sp_season',
      label: 'Pichle season mein',
      icon: '❌',
      desc: 'Overdue'
    }, {
      id: 'c_sp_never',
      label: 'Kabhi nahi kiya',
      icon: '🆘',
      desc: 'Immediate action needed'
    }]
  }],
  /* EXTRA 10 QUESTIONS (cq22-cq31) — Advanced coconut diagnostics */
  coconut_advanced: [{
    id: 'cq22',
    text: 'Naariyal (fruit) mein koi problem dikh rahi hai?',
    hint: 'Fruit quality check',
    options: [{
      id: 'c_fr_ok',
      label: 'Nahi, theek lag raha hai',
      icon: '✅',
      desc: 'No fruit issue'
    }, {
      id: 'c_fr_small',
      label: 'Naariyal chhota reh jaata hai',
      icon: '📉',
      desc: 'Poor nutrition'
    }, {
      id: 'c_fr_black',
      label: 'Naariyal pe kaale daag',
      icon: '⚫',
      desc: 'Fungal/bacterial'
    }, {
      id: 'c_fr_drop',
      label: 'Kacha naariyal girta hai',
      icon: '⬇️',
      desc: 'Physiological/pest'
    }]
  }, {
    id: 'cq23',
    text: 'Kya koi kida (insect) dikha hai naariyal ke ped pe?',
    hint: 'Pest + disease combo check',
    options: [{
      id: 'c_bug_mite',
      label: 'Haan, chhote kide/mite',
      icon: '🐛',
      desc: 'Mite damage'
    }, {
      id: 'c_bug_weevil',
      label: 'Haan, bada kaala kida (weevil)',
      icon: '🪲',
      desc: 'Rhinoceros beetle'
    }, {
      id: 'c_bug_scale',
      label: 'Haan, chipke hue kide',
      icon: '🔵',
      desc: 'Scale insect'
    }, {
      id: 'c_bug_no',
      label: 'Koi kida nahi dikha',
      icon: '❌',
      desc: 'Disease only'
    }]
  }, {
    id: 'cq24',
    text: 'Tanae pe koi daag ya ghav (wound) hai?',
    hint: 'Physical damage check',
    options: [{
      id: 'c_wound_cut',
      label: 'Haan, kataa hua daag',
      icon: '🔪',
      desc: 'Physical + infection risk'
    }, {
      id: 'c_wound_black',
      label: 'Haan, kaala hua hissa',
      icon: '⚫',
      desc: 'Stem bleeding sign'
    }, {
      id: 'c_wound_crack',
      label: 'Haan, daraar (crack) hai',
      icon: '🔓',
      desc: 'Borer entry point'
    }, {
      id: 'c_wound_no',
      label: 'Nahi, tanaa saaf hai',
      icon: '✅',
      desc: 'No physical damage'
    }]
  }, {
    id: 'cq25',
    text: 'Pani (irrigation) kitna aur kab dete ho?',
    hint: 'Water management crucial for coconut',
    options: [{
      id: 'c_irr_regular',
      label: 'Niyamit — hafte mein ek baar',
      icon: '💧',
      desc: 'Good practice'
    }, {
      id: 'c_irr_excess',
      label: 'Zyada paani deta/deti hoon',
      icon: '🌊',
      desc: 'Root rot risk'
    }, {
      id: 'c_irr_less',
      label: 'Kam paani milta hai',
      icon: '🏜️',
      desc: 'Drought stress'
    }, {
      id: 'c_irr_rain',
      label: 'Sirf barsaat pe dependent',
      icon: '🌧️',
      desc: 'Irregular moisture'
    }]
  }, {
    id: 'cq26',
    text: 'Naariyal ke ped ke aas-paas kaun si fasal lagaate ho?',
    hint: 'Intercrop pest transfer check',
    options: [{
      id: 'c_ic_arecanut',
      label: 'Supari (Arecanut)',
      icon: '🌰',
      desc: 'High bud rot transfer risk'
    }, {
      id: 'c_ic_banana',
      label: 'Kela (Banana)',
      icon: '🍌',
      desc: 'Leaf spot cross-infection'
    }, {
      id: 'c_ic_veg',
      label: 'Sabziyan (Vegetables)',
      icon: '🥦',
      desc: 'Generally safe'
    }, {
      id: 'c_ic_none',
      label: 'Kuch nahi — sirf naariyal',
      icon: '🥥',
      desc: 'Monocrop'
    }]
  }, {
    id: 'cq27',
    text: 'Zameen ki pH kaisi hai?',
    hint: 'Soil acidity affects disease',
    options: [{
      id: 'c_ph_acid',
      label: 'Khaati (Acidic < 6)',
      icon: '🔴',
      desc: 'Nutrient lockout'
    }, {
      id: 'c_ph_normal',
      label: 'Theek (6-7.5)',
      icon: '✅',
      desc: 'Ideal for coconut'
    }, {
      id: 'c_ph_alkali',
      label: 'Khaari (Alkaline > 7.5)',
      icon: '🔵',
      desc: 'Mg/Mn deficiency risk'
    }, {
      id: 'c_ph_unknown',
      label: 'Pata nahi',
      icon: '❓',
      desc: 'Test recommended'
    }]
  }, {
    id: 'cq28',
    text: 'Kya poshan (fertilizer) mein NPK + micronutrients dete ho?',
    hint: 'Complete nutrition check',
    options: [{
      id: 'c_npk_complete',
      label: 'Haan, NPK + Mg + Zn + B',
      icon: '✅',
      desc: 'Complete nutrition'
    }, {
      id: 'c_npk_basic',
      label: 'Sirf NPK deta/deti hoon',
      icon: '⚠️',
      desc: 'Micronutrient gap'
    }, {
      id: 'c_npk_organic',
      label: 'Sirf compost/organic',
      icon: '🌱',
      desc: 'May be deficient'
    }, {
      id: 'c_npk_no',
      label: 'Bilkul fertilizer nahi',
      icon: '❌',
      desc: 'High deficiency risk'
    }]
  }, {
    id: 'cq29',
    text: 'Kya pedon mein se mushroom jaisa kuch ugaa hai (basidiomycete)?',
    hint: 'Root/trunk rot sign',
    options: [{
      id: 'c_mush_yes',
      label: 'Haan, jadein/tanay ke paas',
      icon: '🍄',
      desc: 'Ganoderma/root rot critical'
    }, {
      id: 'c_mush_no',
      label: 'Nahi, kuch nahi',
      icon: '✅',
      desc: 'No basal rot'
    }, {
      id: 'c_mush_check',
      label: 'Check nahi kiya abhi tak',
      icon: '🔍',
      desc: 'Check immediately'
    }, {
      id: 'c_mush_soil',
      label: 'Zameen mein hai, ped pe nahi',
      icon: '🌱',
      desc: 'Environmental only'
    }]
  }, {
    id: 'cq30',
    text: 'Ped ka upar wala hissa (canopy) kaisa dikhta hai?',
    hint: 'Overall health indicator',
    options: [{
      id: 'c_can_full',
      label: 'Ghana — poora hara',
      icon: '🌴',
      desc: 'Good health'
    }, {
      id: 'c_can_sparse',
      label: 'Virla — patte kam hain',
      icon: '🌿',
      desc: 'Moderate stress'
    }, {
      id: 'c_can_dead',
      label: 'Upar se sukh raha hai',
      icon: '💀',
      desc: 'Severe — bud/trunk rot'
    }, {
      id: 'c_can_lean',
      label: 'Ped jhuk raha hai ek taraf',
      icon: '↗️',
      desc: 'Root/structural issue'
    }]
  }]
};

/* ════════════════════════════════════════════════════════════════
   QUESTION FLOW STYLES
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   CONSULT PAGE  — Crop → Upload → Question Flow → Processing
════════════════════════════════════════════════════════════════ */
/* ════════════════════════════════════════════════════════════════
   FARMER DASHBOARD
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   EXPERT DASHBOARD
════════════════════════════════════════════════════════════════ */
/* ════════════════════════════════════════════════════════════════
   EXPERTS PAGE
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   BOOKING PAGE
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   CHAT PAGE
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   CASE DETAIL (Expert)
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   NOTIFICATIONS PAGE
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   SUPPORT PAGE
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   PROFILE PAGE
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   SETTINGS PAGE
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   2. SATELLITE FIELD MONITOR 🛰️
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   3. PREDICTIVE DISEASE FORECAST 📊
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   4. IoT SOIL SENSORS 🌱
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   5. B2B DATA INTELLIGENCE 💼
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   6. INPUT MARKETPLACE 📦
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   7. INSURANCE CLAIM 🏦
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   8. GOVT DISEASE SURVEILLANCE MAP 📈
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   ROBOT DASHBOARD 🤖
════════════════════════════════════════════════════════════════ */
const ROBOTS = [{
  id: 'R01',
  name: 'DroneBot Alpha',
  type: 'Drone',
  model: 'DJI Agras T40',
  status: 'online',
  battery: 87,
  signal: 94,
  field: 'Field 1',
  task: 'Spray in progress',
  spray: 68,
  flights: 124,
  area: '2.1 Acres',
  lastSeen: 'Just now',
  emoji: '🚁'
}, {
  id: 'R02',
  name: 'GroundBot Beta',
  type: 'Ground',
  model: 'TartanSense TG-1',
  status: 'busy',
  battery: 62,
  signal: 88,
  field: 'Field 2',
  task: 'Soil scanning',
  spray: 0,
  flights: 0,
  area: '1.5 Acres',
  lastSeen: '3 min ago',
  emoji: '🤖'
}, {
  id: 'R03',
  name: 'DroneBot Gamma',
  type: 'Drone',
  model: 'ideaForge RYNO',
  status: 'offline',
  battery: 12,
  signal: 0,
  field: 'Charging',
  task: 'Charging station',
  spray: 0,
  flights: 89,
  area: '—',
  lastSeen: '2 hrs ago',
  emoji: '🚁'
}, {
  id: 'R04',
  name: 'SensorBot Delta',
  type: 'Sensor',
  model: 'Custom IoT v2',
  status: 'online',
  battery: 91,
  signal: 99,
  field: 'All Fields',
  task: 'Monitoring',
  spray: 0,
  flights: 0,
  area: '4.5 Acres',
  lastSeen: 'Live',
  emoji: '📡'
}];
/* ════════════════════════════════════════════════════════════════
   LIVE CAMERA FEED 📡
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   AUTO FIELD NAVIGATION & MAPPING 🗺️
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   MANUAL ROBOT CONTROL 🎮
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   BATTERY & MAINTENANCE 🔋
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   ROBOT ANALYTICS 📊
════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════
   ROOT APP
════════════════════════════════════════════════════════════════ */

export default function SafeBeejHealth() {
  return <ErrorBoundary>
      <BeejHealthApp />
    </ErrorBoundary>;
}
