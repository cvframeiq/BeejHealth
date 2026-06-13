import React, { useState, useEffect, useRef, useCallback } from "react";
import { useTranslation } from 'react-i18next';
import APP_CSS from '../styles/App.css?raw';
import { API } from '../services/api.js';
import { resolveConsultPhotoUrls } from '../utils/helpers.jsx';
import { CROPS, DISEASE_DB, getReportQuestionRows, buildReportAiRows, buildReportSnapshot, LOGO_FULL_B64 } from '../utils/constants.jsx';
import { cropName, reportLabel, reportValue, rt, tx } from '../utils/localize.jsx';

function openPrintableReport(reportRoot, fileName = 'BeejHealth-Report') {
  if (!reportRoot) return false;
  const clone = reportRoot.cloneNode(true);
  clone.querySelectorAll('[data-no-export="true"]').forEach(el => el.remove());
  const win = window.open('', '_blank', 'noopener,noreferrer,width=980,height=1200');
  if (!win) return false;
  win.document.open();
  win.document.write(`<!doctype html><html><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>${fileName}</title><style>${APP_CSS}</style><style>@page{size:A4 portrait;margin:12mm}html,body{margin:0;padding:0;background:#fff}body{-webkit-print-color-adjust:exact;print-color-adjust:exact}.wrap-sm{max-width:760px;margin:0 auto;padding:0}.rep-sheet{box-shadow:none;border-radius:0}.rep-actions,.report-print-title{display:none!important}</style></head><body>${clone.outerHTML}</body></html>`);
  win.document.close();
  const triggerPrint = () => {
    try { win.focus(); } catch {}
    try { win.print(); } catch {}
  };
  setTimeout(triggerPrint, 400);
  win.onafterprint = () => {
    try { win.close(); } catch {}
  };
  return true;
}

async function waitForReportAssets(root) {
  try {
    if (document.fonts?.ready) await document.fonts.ready;
  } catch {}
  const imgs = Array.from(root?.querySelectorAll('img') || []);
  await Promise.all(imgs.map(img => img.complete ? Promise.resolve() : new Promise(resolve => {
    img.addEventListener('load', resolve, { once: true });
    img.addEventListener('error', resolve, { once: true });
  })));
}

export default /* ════════════════════════════════════════════════════════════════
   AI REPORT PAGE  — Full Branded Report with Logo
════════════════════════════════════════════════════════════════ */
function AIReportPage({
  selCrop,
  nav,
  toast,
  qAnswers,
  viewConsultId
}) {
  const { i18n } = useTranslation();
  const crop = selCrop || CROPS[0];
  const reportWrapRef = useRef(null);
  const [dbConsult, setDbConsult] = useState(null);
  const [consultPhotos, setConsultPhotos] = useState([]);
  const [loadingConsult, setLoadingConsult] = useState(false);
  const savedReportRef = useRef(null);
  const consultId = viewConsultId || localStorage.getItem('bh_view_consult') || localStorage.getItem('bh_latest_consult');
  // Coconut AI result (from real model)
  const [aiResult, setAiResult] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('bh_ai_result') || 'null');
    } catch {
      return null;
    }
  });
  useEffect(() => {
    // Load real consultation from DB
    if (!consultId) return;
    setDbConsult(null);
    setConsultPhotos([]);
    savedReportRef.current = null;
    setLoadingConsult(true);
    API.get('/api/consultations/' + consultId).then(d => {
      if (d.consultation) setDbConsult(d.consultation);
    }).catch(() => {}).finally(() => setLoadingConsult(false));
  }, [consultId]);
  useEffect(() => {
    if (!consultId) {
      setConsultPhotos([]);
      return;
    }
    let cancelled = false;
    API.get(`/api/photos?consultationId=${encodeURIComponent(consultId)}`).then(d => {
      if (cancelled) return;
      const urls = Array.isArray(d?.photos) ? d.photos.map(photo => photo?.url).filter(Boolean) : [];
      setConsultPhotos(urls);
    }).catch(() => {
      if (!cancelled) setConsultPhotos([]);
    });
    return () => {
      cancelled = true;
    };
  }, [consultId]);

  // Use coconut AI result first, then DB, then fallback
  const isCoconutReport = dbConsult?.cropId === 'coconut' || crop?.id === 'coconut' || !!aiResult;
  const effectiveCropId = dbConsult?.cropId || crop.id;
  const effectiveAnswers = dbConsult?.answers || qAnswers || {};
  const effectiveDisease = dbConsult?.disease ?? aiResult?.disease;
  const effectiveConf = dbConsult?.confidence ?? aiResult?.confidence;
  const effectiveSev = dbConsult?.severity ?? aiResult?.severity;
  const effectiveHindi = (dbConsult?.disease_hindi ?? aiResult?.disease_hindi) || '';
  const effectiveTreatments = dbConsult?.ai_treatments ?? aiResult?.treatments ?? [];
  const effectiveTop3 = dbConsult?.ai_top3 ?? aiResult?.top3 ?? [];
  const effectiveUrgency = dbConsult?.urgency ?? aiResult?.urgency ?? 'medium';
  const reportCrop = {
    ...crop,
    id: dbConsult?.cropId || crop.id,
    name: cropName(dbConsult?.cropId || crop.id, i18n) || dbConsult?.cropName || cropName(crop, i18n) || crop.name,
    emoji: dbConsult?.cropEmoji || crop.emoji
  };
  const reportPhotoUrls = Array.from(new Set([...resolveConsultPhotoUrls(dbConsult), ...consultPhotos].filter(Boolean)));
  const reportPhotoCount = dbConsult?.photoCount || reportPhotoUrls.length || 0;
  const answers = effectiveAnswers;
  const reportIdSeed = dbConsult?._id || consultId || `preview-${reportCrop.id}`;
  const reportId = dbConsult?.reportSnapshot?.reportId || `BH-${String(reportIdSeed).replace(/[^a-zA-Z0-9]/g, '').slice(-8).toUpperCase() || 'REPORT'}`;
  const d = DISEASE_DB[effectiveCropId] || DISEASE_DB.default;
  const reportD = {
    ...d,
    disease: effectiveDisease || d.disease,
    conf: effectiveConf || d.conf,
    sev: effectiveSev || d.sev,
    sevLabel: effectiveSev ? `Stage ${effectiveSev}/5` : d.sevLabel
  };
  const today = new Date().toLocaleDateString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  });
  /* Q-derived values */
  const q2 = answers.q2;
  const q3 = answers.q3;
  const q9 = answers.q9;
  const q20 = answers.q20;
  const triggerText = q9 ? q9.id === 'heavy' || q9.id === 'light' ? 'Baarish + humidity 68–75%' : 'Dry conditions — monitoring needed' : d.humidity;
  const sprayText = q20 ? q20.id === 'never' || q20.id === 'old' ? 'Spray nahi kiya tha lamba time' : 'Recent spray tha — phir bhi infection' : d.humidity;
  const storedReportSnapshot = dbConsult?.reportSnapshot || null;
  const questionRows = storedReportSnapshot?.questionRows?.length ? storedReportSnapshot.questionRows : getReportQuestionRows({
    cropId: effectiveCropId,
    answers,
    aiResult
  });
  const aiRows = storedReportSnapshot?.aiRows?.length ? storedReportSnapshot.aiRows : buildReportAiRows({
    reportD,
    effectiveTop3,
    effectiveTreatments,
    effectiveUrgency
  });
  const R = key => rt(i18n, key);
  const RV = value => reportValue(value, i18n);
  const RL = label => reportLabel(label, i18n);
  const localizedQuestionRows = questionRows.map(row => ({
    ...row,
    question: RV(row.question),
    answer: RV(row.answer)
  }));
  const localizedAiRows = aiRows.map(row => ({
    ...row,
    label: RL(row.label),
    value: RV(row.value)
  }));
  const reportSnapshot = storedReportSnapshot || buildReportSnapshot({
    reportId,
    reportSeed: reportIdSeed,
    crop: reportCrop,
    reportD,
    dbConsult,
    photoUrlsOverride: reportPhotoUrls,
    answers,
    questionRows,
    aiRows,
    effectiveTop3,
    effectiveTreatments,
    effectiveUrgency,
    triggerText,
    sprayText,
    aiResult
  });
  const persistReportSnapshot = async (extra = {}) => {
    if (!dbConsult?._id) return null;
    const mergedPhotos = reportPhotoUrls.map((url, index) => ({
      photoId: dbConsult?.photoIds?.[index] || dbConsult?.photoId || null,
      url,
      index: index + 1
    }));
    const payload = {
      ...reportSnapshot,
      photos: mergedPhotos.length ? mergedPhotos : reportSnapshot.photos || [],
      photoCount: mergedPhotos.length || reportSnapshot.photoCount || 0,
      ...extra,
      downloadedAt: extra.downloadedAt || reportSnapshot.downloadedAt || null,
      savedAt: new Date().toISOString()
    };
    const res = await API.patch('/api/consultations/' + dbConsult._id + '/report', {
      reportSnapshot: payload,
      reportSummary: payload.summary
    });
    if (res?.consultation) {
      setDbConsult(res.consultation);
    }
    return res?.consultation || payload;
  };
  useEffect(() => {
    if (!dbConsult?._id) return;
    const storedSnapshotPhotos = Array.isArray(dbConsult.reportSnapshot?.photos) ? dbConsult.reportSnapshot.photos.filter(photo => photo?.url || photo?.photoId) : [];
    const snapshotNeedsRefresh = storedSnapshotPhotos.length < reportPhotoUrls.length || !dbConsult.reportSnapshot?.questionRows?.length || !dbConsult.reportSnapshot?.aiRows?.length;
    if (dbConsult.reportSnapshot?.reportId === reportSnapshot.reportId && !snapshotNeedsRefresh) {
      savedReportRef.current = reportSnapshot.reportId;
      return;
    }
    if (savedReportRef.current === reportSnapshot.reportId) return;
    let cancelled = false;
    persistReportSnapshot().then(() => {
      if (!cancelled) savedReportRef.current = reportSnapshot.reportId;
    }).catch(err => {
      console.warn('Report snapshot save failed:', err.message);
    });
    return () => {
      cancelled = true;
    };
  }, [dbConsult?._id, reportSnapshot.reportId]);
  const handleDownloadReport = async () => {
    const reportIdSafe = `BeejHealth-Report-${reportId}`.replace(/\s+/g, '_');
    const reportNode = reportWrapRef.current?.querySelector('.rep-sheet');
    try {
      if (!reportNode) throw new Error('Report content not found');
      await waitForReportAssets(reportNode);
      const {
        default: html2pdf
      } = await import('html2pdf.js');
      const opt = {
        filename: `${reportIdSafe}.pdf`,
        margin: 0,
        image: {
          type: 'jpeg',
          quality: 0.98
        },
        html2canvas: {
          scale: Math.max(2, Math.min(3, window.devicePixelRatio || 2)),
          useCORS: true,
          backgroundColor: '#ffffff',
          scrollX: 0,
          scrollY: 0,
          windowWidth: reportNode.scrollWidth,
          windowHeight: reportNode.scrollHeight,
          ignoreElements: element => {
            if (!element) return false;
            return element.matches?.('[data-no-export="true"]') || element.closest?.('[data-no-export="true"]');
          },
          onclone: clonedDoc => {
            clonedDoc.documentElement.style.background = '#ffffff';
            clonedDoc.body.style.background = '#ffffff';
            const sheet = clonedDoc.querySelector('.rep-sheet');
            if (sheet) {
              sheet.style.boxShadow = 'none';
              sheet.style.borderRadius = '0';
            }
            clonedDoc.querySelectorAll('[data-no-export="true"]').forEach(el => {
              el.style.display = 'none';
              el.setAttribute('aria-hidden', 'true');
            });
            clonedDoc.querySelectorAll('.rep-actions').forEach(el => {
              el.style.display = 'none';
              el.setAttribute('aria-hidden', 'true');
            });
            clonedDoc.querySelectorAll('.report-no-export').forEach(el => {
              el.style.display = 'none';
              el.setAttribute('aria-hidden', 'true');
            });
          }
        },
        jsPDF: {
          unit: 'mm',
          format: 'a4',
          orientation: 'portrait'
        },
        pagebreak: {
          mode: ['css', 'legacy']
        }
      };
      await html2pdf().set(opt).from(reportNode).save();
      persistReportSnapshot({
        downloadedAt: new Date().toISOString()
      }).catch(err => {
        console.warn('Report snapshot download save failed:', err.message);
      });
      toast('PDF download start ho gaya ✅', 'inf');
    } catch (err) {
      console.error('PDF download failed:', err);
      const opened = openPrintableReport(reportWrapRef.current, reportIdSafe);
      if (!opened) toast('PDF export fail hua. Please popup allow karein.', 'err');else toast('PDF export fail hua, print view khul rahi hai...', 'warn');
    }
  };
  return <div className="wrap-sm" ref={reportWrapRef}>
      <button className="btn btn-ghost btn-sm report-no-export" data-no-export="true" style={{
      marginBottom: 18
    }} onClick={() => nav('consultation')}>← {tx(i18n, 'newDiagnosis')}</button>

      <div className="rep-sheet">
        {/* ── HEADER ── */}
        <div className="rep-header">
          <div className="rh-top">
            <div className="rh-logo">
              <img className="rh-logo-img" src={`data:image/png;base64,${LOGO_FULL_B64}`} alt="FrameIQ" />
              <div className="rh-logo-sep" />
              <div className="rh-platform">{tx(i18n, 'aiReport')}</div>
            </div>
            <div className="rh-meta">
              <div className="rh-badge"><span className="rh-badge-dot" />&nbsp;{tx(i18n, 'analysisComplete')}</div>
              <div className="rh-id">{tx(i18n, 'report')} #{reportId}</div>
            </div>
          </div>
          <div className="rh-main">
            <div className="rh-crop-line">{reportCrop.emoji} {reportCrop.name.toUpperCase()} · {today}</div>
            <div className="rh-disease">{RV(reportD.disease)} {tx(i18n, 'diseaseDetected')}</div>
            <div className="rh-sci">{reportD.sci} · {RV(reportD.hindi)}</div>
            <div className="rh-scores">
              <div className="rhs"><div className="rhs-val">{reportD.conf}%</div><div className="rhs-lbl">{tx(i18n, 'confidence')}</div><div className="rhs-bar"><div className="rhs-fill" style={{
                  width: `${reportD.conf}%`
                }} /></div></div>
              <div className="rhs"><div className="rhs-val">{reportD.sev}/5</div><div className="rhs-lbl">{tx(i18n, 'severity')}</div><div className="rhs-bar"><div className="rhs-fill" style={{
                  width: `${reportD.sev * 20}%`
                }} /></div></div>
              <div className="rhs"><div className="rhs-val">{reportD.aff}%</div><div className="rhs-lbl">{tx(i18n, 'areaAffected')}</div><div className="rhs-bar"><div className="rhs-fill" style={{
                  width: `${reportD.aff}%`
                }} /></div></div>
              <div className="rhs"><div className="rhs-val">{reportD.sev >= 3 ? R('high') : d.sev === 2 ? R('mediumShort') : R('low')}</div><div className="rhs-lbl">{tx(i18n, 'riskLevel')}</div><div className="rhs-bar"><div className="rhs-fill" style={{
                  width: d.sev >= 3 ? '80%' : d.sev === 2 ? '50%' : '25%'
                }} /></div></div>
            </div>
          </div>
        </div>

        {/* ── BODY ── */}
        <div className="rep-body">

          {/* S1: Disease ID */}
          <div className="r-sec">
            <div className="r-sec-hd"><div className="r-sec-num">1</div><div className="r-sec-title">{tx(i18n, 'diseaseId')}</div><span className="r-sec-tag r-tag-b">{R('photo')} + Q1</span></div>
            {false && reportPhotoUrls.length > 0 && <div style={{
            marginBottom: 14
          }}>
                <div style={{
              fontSize: 12,
              fontWeight: 700,
              color: 'var(--tx3)',
              marginBottom: 8,
              textTransform: 'uppercase',
              letterSpacing: '.5px'
            }}>
                  📸 Uploaded Photos ({reportPhotoCount || reportPhotoUrls.length})
                </div>
                <div style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${Math.min(reportPhotoUrls.length, 3)},1fr)`,
              gap: 8
            }}>
                  {reportPhotoUrls.map((url, i) => <div key={i} style={{
                borderRadius: 8,
                overflow: 'hidden',
                border: '1.5px solid var(--br)',
                aspectRatio: '1'
              }}>
                      <img src={url} alt={`Photo ${i + 1}`} style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  display: 'block'
                }} />
                    </div>)}
                </div>
                <div style={{
              marginTop: 6,
              padding: '5px 10px',
              background: 'var(--gp)',
              borderRadius: 7,
              fontSize: 11.5,
              color: 'var(--tx3)',
              fontWeight: 600
            }}>
                  📸 AI Analysis in {reportPhotoCount || 1} photo{(reportPhotoCount || 1) > 1 ? 's' : ''} se ki gayi
                </div>
              </div>}
            <div className="r-card gl">
              <div className="r-fr"><div className="r-fl">{tx(i18n, 'disease')}</div><div className="r-fv b">{RV(reportD.disease)} <span className="r-pill r-pill-p">{R('photo')}</span></div></div>
              <div className="r-fr"><div className="r-fl">{tx(i18n, 'scientificName')}</div><div className="r-fv"><em>{reportD.sci}</em></div></div>
              <div className="r-fr"><div className="r-fl">{tx(i18n, 'localName')}</div><div className="r-fv">{RV(reportD.hindi)}</div></div>
              <div className="r-fr"><div className="r-fl">{tx(i18n, 'severityStage')}</div><div className="r-fv b">{RV(reportD.sevLabel)}</div></div>
              <div className="r-conf-row"><span>{tx(i18n, 'confidence')}</span><span style={{
                color: 'var(--g4)',
                fontWeight: 700
              }}>{reportD.conf}%</span></div>
              <div className="r-conf-bar"><div className="r-conf-fill" style={{
                width: `${reportD.conf}%`
              }} /></div>
            </div>
          </div>

          {/* Questionnaire Summary */}
          {(questionRows.length > 0 || aiRows.length > 0) && <div style={{
          marginBottom: 14
        }}>
              <div className="r-sec" style={{
            marginBottom: 14
          }}>
                <div className="r-sec-hd">
                  <div className="r-sec-num">Q</div>
                  <div className="r-sec-title">{tx(i18n, 'userQuestions')}</div>
                  <span className="r-sec-tag r-tag-q">{R('savedInDb')}</span>
                </div>
                <div className="r-card" style={{
              background: 'linear-gradient(180deg,#ffffff 0%, #f8fbf8 100%)'
            }}>
                  {localizedQuestionRows.length > 0 ? localizedQuestionRows.map((row, i) => <div key={row.id || i} style={{
                padding: '10px 0',
                borderBottom: i < localizedQuestionRows.length - 1 ? '1px solid var(--br)' : 'none'
              }}>
                      <div style={{
                  fontSize: 11,
                  textTransform: 'uppercase',
                  letterSpacing: '.45px',
                  color: 'var(--tx3)',
                  fontWeight: 700,
                  marginBottom: 4
                }}>{tx(i18n, 'question')} {i + 1}</div>
                      <div style={{
                  fontSize: 13.2,
                  fontWeight: 700,
                  color: 'var(--tx)',
                  lineHeight: 1.55
                }}>{row.question}</div>
                      <div style={{
                  marginTop: 5,
                  fontSize: 12.6,
                  color: 'var(--g2)',
                  lineHeight: 1.55
                }}><span style={{
                    fontWeight: 700
                  }}>{tx(i18n, 'answer')}:</span> {row.answer}</div>
                    </div>) : <div style={{
                padding: '10px 0',
                fontSize: 12.8,
                color: 'var(--tx3)',
                lineHeight: 1.6
              }}>
                      {tx(i18n, 'noAnswers')}
                    </div>}
            {false && reportPhotoUrls.length > 0 && <div style={{
                paddingTop: questionRows.length > 0 ? 14 : 0,
                marginTop: questionRows.length > 0 ? 10 : 0,
                borderTop: questionRows.length > 0 ? '1px solid var(--br)' : 'none'
              }}>
                      <div style={{
                  fontSize: 11,
                  textTransform: 'uppercase',
                  letterSpacing: '.45px',
                  color: 'var(--tx3)',
                  fontWeight: 700,
                  marginBottom: 8
                }}>
                        Uploaded Images ({reportPhotoCount || reportPhotoUrls.length})
                      </div>
                      <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3,1fr)',
                  gap: 8
                }}>
                        {reportPhotoUrls.map((url, i) => <div key={`question-photo-${i}`} style={{
                    borderRadius: 8,
                    overflow: 'hidden',
                    border: '1.5px solid var(--br)',
                    aspectRatio: '1',
                    background: 'var(--gp)'
                  }}>
                            <img src={url} alt={`Uploaded ${i + 1}`} style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      display: 'block'
                    }} />
                          </div>)}
                      </div>
                    </div>}
                </div>
              </div>
              <div className="r-sec" style={{
            marginBottom: 14,
            pageBreakInside: 'avoid',
            breakInside: 'avoid'
          }}>
                <div className="r-sec-hd">
                  <div className="r-sec-num" style={{
                background: '#1565C0'
              }}>AI</div>
                  <div className="r-sec-title">{R('aiExpertAnswers')}</div>
                  <span className="r-sec-tag" style={{
                background: '#E3F0FB',
                color: '#1565C0'
              }}>{R('savedInDb')}</span>
                </div>
                <div className="r-card" style={{
              background: 'linear-gradient(180deg,#ffffff 0%, #f7fbff 100%)',
              marginBottom: 14,
              pageBreakInside: 'avoid',
              breakInside: 'avoid'
            }}>
                  {localizedAiRows.map((row, i) => <div key={row.label || i} style={{
                display: 'flex',
                gap: 12,
                justifyContent: 'space-between',
                padding: '8px 0',
                borderBottom: i < localizedAiRows.length - 1 ? '1px solid var(--br)' : 'none'
              }}>
                      <div style={{
                  fontSize: 12,
                  color: 'var(--tx3)',
                  fontWeight: 700,
                  flexShrink: 0
                }}>{row.label}</div>
                      <div style={{
                  fontSize: 12.7,
                  color: 'var(--tx)',
                  fontWeight: 700,
                  textAlign: 'right',
                  lineHeight: 1.55,
                  flex: 1
                }}>{row.value}</div>
                    </div>)}
                </div>
                
                <div className="r-card" style={{
              background: 'var(--gp)'
            }}>
                  <div style={{
                fontSize: 12,
                fontWeight: 700,
                color: 'var(--tx3)',
                marginBottom: 10,
                textTransform: 'uppercase',
                letterSpacing: '.5px'
              }}>
                    {R('farmerUploadedPhotos')} ({reportPhotoCount || reportPhotoUrls.length || 1})
                  </div>
                  <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                gap: 12,
                alignItems: 'start'
              }}>
                    {(reportPhotoUrls.length > 0 ? reportPhotoUrls : ['https://placehold.co/400x400/e6f4ea/1e7e42?text=Sample+Upload']).map((url, i) => <div key={`report-photo-${i}`} style={{
                  borderRadius: 10,
                  overflow: 'hidden',
                  border: '1.5px solid var(--br)',
                  background: '#fff',
                  padding: 6
                }}>
                        <img src={url} alt={`Photo ${i + 1}`} style={{
                    width: '100%',
                    aspectRatio: '4/3',
                    objectFit: 'cover',
                    display: 'block',
                    borderRadius: 6,
                    background: 'white'
                  }} crossOrigin="anonymous" />
                      </div>)}
                  </div>
                  <div style={{
                marginTop: 10,
                padding: '6px 10px',
                background: '#fff',
                border: '1px solid var(--br)',
                borderRadius: 7,
                fontSize: 11.5,
                color: 'var(--tx3)',
                fontWeight: 600
              }}>
                    {R('farmerPhotoNote')}
                  </div>
                </div>
              </div>
            </div>}

          {/* COCONUT AI: Real Model Results */}
          {isCoconutReport && effectiveTop3 && effectiveTop3.length > 0 && <div className="r-sec" style={{
          marginBottom: 14
        }}>
              <div className="r-sec-hd">
                <div className="r-sec-num" style={{
              background: '#1565C0'
            }}>AI</div>
                <div className="r-sec-title">{R('modelResults')}</div>
                <span className="r-sec-tag" style={{
              background: '#E3F0FB',
              color: '#1565C0'
            }}>{R('realModel')}</span>
              </div>
              <div style={{
            padding: '14px 16px',
            background: '#E3F0FB',
            borderRadius: 10,
            marginBottom: 10,
            border: '1.5px solid #1565C0'
          }}>
                <div style={{
              fontSize: 12,
              fontWeight: 700,
              color: '#1565C0',
              marginBottom: 8,
              textTransform: 'uppercase',
              letterSpacing: '.5px'
            }}>
                  🤖 {R('top3Predictions')}
                </div>
                {effectiveTop3.map((p, i) => <div key={i} style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '7px 0',
              borderBottom: i < 2 ? '1px solid rgba(21,101,192,.15)' : 'none'
            }}>
                    <div style={{
                display: 'flex',
                gap: 9,
                alignItems: 'center'
              }}>
                      <div style={{
                  width: 22,
                  height: 22,
                  borderRadius: '50%',
                  background: i === 0 ? '#1565C0' : 'rgba(21,101,192,.2)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 11,
                  fontWeight: 800,
                  color: i === 0 ? 'white' : '#1565C0'
                }}>#{p.rank}</div>
                      <span style={{
                  fontSize: 13,
                  fontWeight: i === 0 ? 800 : 600,
                  color: i === 0 ? '#1565C0' : 'var(--tx2)'
                }}>{RV(p.disease)}</span>
                    </div>
                    <div style={{
                display: 'flex',
                gap: 8,
                alignItems: 'center'
              }}>
                      <span style={{
                  fontSize: 13,
                  fontWeight: 800,
                  color: i === 0 ? '#1565C0' : 'var(--tx3)'
                }}>{p.confidence}%</span>
                      <div style={{
                  width: 60,
                  height: 6,
                  background: 'rgba(21,101,192,.15)',
                  borderRadius: 3
                }}>
                        <div style={{
                    width: `${p.confidence}%`,
                    height: '100%',
                    background: i === 0 ? '#1565C0' : 'rgba(21,101,192,.4)',
                    borderRadius: 3
                  }} />
                      </div>
                    </div>
                  </div>)}
              </div>
              {effectiveHindi && <div style={{
            fontSize: 12.5,
            color: 'var(--tx2)',
            padding: '8px 12px',
            background: 'var(--gp)',
            borderRadius: 8
          }}>
                  🌐 {R('hindi')}: <strong>{RV(effectiveHindi)}</strong>
                  {effectiveUrgency === 'critical' && <span style={{
              marginLeft: 10,
              background: '#FEE2E2',
              color: '#DC2626',
              padding: '2px 8px',
              borderRadius: 100,
              fontSize: 11,
              fontWeight: 700
            }}>⚠️ {R('critical')}</span>}
                  {effectiveUrgency === 'high' && <span style={{
              marginLeft: 10,
              background: '#FEF3C7',
              color: '#D97706',
              padding: '2px 8px',
              borderRadius: 100,
              fontSize: 11,
              fontWeight: 700
            }}>⚡ {R('high')}</span>}
                </div>}
            </div>}

          {/* COCONUT AI: Treatment Plan from Model */}
          {isCoconutReport && effectiveTreatments && effectiveTreatments.length > 0 && <div className="r-sec" style={{
          marginBottom: 14
        }}>
              <div className="r-sec-hd">
                <div className="r-sec-num" style={{
              background: '#16a34a'
            }}>Rx</div>
                <div className="r-sec-title">{tx(i18n, 'aiTreatmentPlan')}</div>
                <span className="r-sec-tag r-tag-b">{R('modelBased')}</span>
              </div>
              <div style={{
            background: 'var(--gp)',
            borderRadius: 10,
            padding: 14
          }}>
                {effectiveTreatments.map((t, i) => <div key={i} style={{
              display: 'flex',
              gap: 10,
              padding: '8px 0',
              borderBottom: i < effectiveTreatments.length - 1 ? '1px solid var(--br)' : 'none',
              alignItems: 'flex-start'
            }}>
                    <div style={{
                width: 22,
                height: 22,
                borderRadius: '50%',
                background: 'var(--g4)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 11,
                fontWeight: 800,
                color: 'white',
                flexShrink: 0,
                marginTop: 1
              }}>{i + 1}</div>
                    <span style={{
                fontSize: 13,
                color: 'var(--tx)',
                lineHeight: 1.55
              }}>{RV(t)}</span>
                  </div>)}
              </div>
            </div>}

          {/* S2 + S3 */}
          <div className="r-two">
            <div className="r-sec">
              <div className="r-sec-hd"><div className="r-sec-num">2</div><div className="r-sec-title">{R('sevenDaySpread')}</div><span className="r-sec-tag r-tag-b">{R('photo')} + Q2,Q3</span></div>
              <div className="r-card">
                <div className="r-fr"><div className="r-fl">{R('affectedNow')}</div><div className="r-fv b">{reportD.aff}% {R('leaves')}</div></div>
                {q2 && <div className="r-fr"><div className="r-fl">{R('spotLocation')}</div><div className="r-fv">{RV(q2.label)} <span className="r-pill r-pill-q">Q2</span></div></div>}
                {q3 && <div className="r-fr" style={{
                borderBottom: 'none'
              }}><div className="r-fl">{R('spotColor')}</div><div className="r-fv">{RV(q3.label)} <span className="r-pill r-pill-q">Q3</span></div></div>}
                {!q2 && !q3 && <div className="r-fr" style={{
                borderBottom: 'none'
              }}><div className="r-fl">{R('status')}</div><div className="r-fv">{R('photoAnalysisComplete')}</div></div>}
                <div className="r-spread">
                  <div className="r-sc un"><div className="r-sc-num">{reportD.utr}%</div><div className="r-sc-lbl">{R('untreated')}</div><div style={{
                    fontSize: 10,
                    opacity: .65
                  }}>{R('after7Days')}</div></div>
                  <div className="r-sc tr"><div className="r-sc-num">{reportD.tr}%</div><div className="r-sc-lbl">{R('treated')}</div><div style={{
                    fontSize: 10,
                    opacity: .65
                  }}>{R('in5Days')}</div></div>
                </div>
              </div>
            </div>
            <div className="r-sec">
              <div className="r-sec-hd"><div className="r-sec-num">3</div><div className="r-sec-title">{R('rootCause')}</div><span className="r-sec-tag r-tag-q">Q9, Q20</span></div>
              <div className="r-card">
                <div className="r-fr"><div className="r-fl">{R('primaryCause')}</div><div className="r-fv b">{RV(reportD.cause)}</div></div>
                <div className="r-fr"><div className="r-fl">{R('trigger')} {q9 && <span className="r-pill r-pill-q">Q9</span>}</div><div className="r-fv">{RV(triggerText)}</div></div>
                <div className="r-fr"><div className="r-fl">{R('contributing')} {q20 && <span className="r-pill r-pill-q">Q20</span>}</div><div className="r-fv">{RV(sprayText)}</div></div>
                <div className="r-fr" style={{
                borderBottom: 'none'
              }}><div className="r-fl">{R('risk')}</div><div className="r-fv r">{RV(reportD.risk)}</div></div>
              </div>
            </div>
          </div>

          {/* S4: Treatment */}
          <div className="r-sec" style={{
          pageBreakInside: 'avoid',
          breakInside: 'avoid'
        }}>
            <div className="r-sec-hd"><div className="r-sec-num">4</div><div className="r-sec-title">{tx(i18n, 'treatmentPlan3')}</div><span className="r-sec-tag r-tag-ai">AI</span></div>
            <div className="r-phases" style={{
            pageBreakInside: 'avoid',
            breakInside: 'avoid'
          }}>
              {[['t', tx(i18n, 'todayDo'), '🔴'], ['w', tx(i18n, 'thisWeek'), '🟡'], ['s', tx(i18n, 'thisSeason'), '🟢']].map(([cls, lbl, ic], i) => <div key={cls} className={`r-ph ${cls}`}>
                  <div className="r-ph-dot" />
                  <div><div className="r-ph-tag">{ic} {lbl}</div><div className="r-ph-txt">{RV(reportD.phases[i])}</div></div>
                </div>)}
            </div>
          </div>

          {/* S5 + S6+S7 */}
          <div className="r-two">
            <div className="r-sec">
              <div className="r-sec-hd"><div className="r-sec-num">5</div><div className="r-sec-title">{R('medicines')}</div><span className="r-sec-tag r-tag-ai">{R('aiMatched')}</span></div>
              <div className="r-med-list">
                {reportD.meds.map((m, i) => <div key={m.nm} className={`r-med${m.top ? ' top' : ''}`}>
                    <div className="r-med-rank" style={!m.top ? {
                  background: 'var(--tx4)'
                } : {}}>{i + 1}</div>
                    <div style={{
                  flex: 1
                }}>
                      <div className="r-med-nm">{m.nm}{m.top && <span className="r-rec-badge">{R('best')}</span>}</div>
                      <div className="r-med-ty">{RV(m.ty)}</div>
                    </div>
                    <div className="r-med-pr">{m.pr}</div>
                  </div>)}
              </div>
              <div style={{
              marginTop: 9,
              padding: '8px 11px',
              background: 'var(--gp)',
              borderRadius: 8,
              fontSize: 12,
              color: 'var(--g2)',
              fontWeight: 600
            }}>
                📍 {R('agriStore')}
              </div>
            </div>
            <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 12
          }}>
              <div className="r-sec">
                <div className="r-sec-hd"><div className="r-sec-num">6</div><div className="r-sec-title">{tx(i18n, 'weatherRisk')}</div><span className="r-sec-tag r-tag-q">Q9</span></div>
                <div className="r-alert">
                  <div className="r-alert-title">⚠️ {R('highRiskActToday')}</div>
                  <div className="r-fr"><div className="r-fl">{R('humidity')}</div><div className="r-fv r">{RV('68–75% Dangerous')}</div></div>
                  <div className="r-fr"><div className="r-fl">{R('sprayWindow')}</div><div className="r-fv">{RV('Aaj 4–6 PM')}</div></div>
                  <div className="r-fr" style={{
                  borderBottom: 'none'
                }}><div className="r-fl">{R('nearbyCases')}</div><div className="r-fv r">{RV('8 within 5km')}</div></div>
                </div>
              </div>
              <div className="r-sec">
                <div className="r-sec-hd"><div className="r-sec-num">7</div><div className="r-sec-title">{R('expertAdvice')}</div><span className="r-sec-tag r-tag-b">{R('stageBased')}</span></div>
                <div className="r-card">
                  <div className="r-fr"><div className="r-fl">{R('consult')}</div><div className="r-fv g">{R('recommended')}</div></div>
                  <div className="r-fr"><div className="r-fl">{R('expertName')}</div><div className="r-fv">Dr. Rajesh Kumar</div></div>
                  <div className="r-fr" style={{
                  borderBottom: 'none'
                }}><div className="r-fl">{R('followUp')}</div><div className="r-fv">{R('followUpPhoto')}</div></div>
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="rep-actions" data-no-export="true" style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 9,
          marginTop: 20
        }}>
            <button className="btn btn-g btn-md" onClick={() => nav('experts')}>👨‍⚕️ {R('expertConfirm')}</button>
            <button className="btn btn-out btn-md" onClick={handleDownloadReport}>📄 {tx(i18n, 'pdfDownload')}</button>
            <button className="btn btn-ghost btn-md" onClick={() => toast('Treatment reminder set ✅')}>🔔 {tx(i18n, 'reminder')}</button>
            <button className="btn btn-ghost btn-md" onClick={() => nav('marketplace')}>🛒 {R('medicineOrder')}</button>
          </div>
        </div>

        {/* ── FOOTER ── */}
        <div className="rep-footer">
          <div className="rf-logo">
            <img className="rf-logo-img" src={`data:image/png;base64,${LOGO_FULL_B64}`} alt="FrameIQ" />
            <div className="rf-txt">{R('footerBrand')}</div>
          </div>
          <div className="rf-disc">{R('footerDisclaimer')}</div>
        </div>
      </div>
    </div>;
}

/* ════════════════════════════════════════════════════════════════
   MY CONSULTATIONS
════════════════════════════════════════════════════════════════ */
