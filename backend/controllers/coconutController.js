import { prisma } from '../db.js';
import { callAI } from '../utils/helpers.js';
import { AI_SERVER } from '../config/env.js';

export const checkAIStatus = async (req, res) => {
  try {
    const r = await fetch(`${AI_SERVER}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await r.json();
    res.json({ ai_online: true, ...data });
  } catch {
    res.json({ ai_online: false, message: 'Python AI server band hai. Port 8000 check karein.' });
  }
};

export const getQuestions = (req, res) => {
  res.json({
    crop: 'coconut', model: 'EfficientNetV2-S Transfer Learning', total_questions: 30, questions_per_session: 5,
    classes: [
      { id: 'Gray Leaf Spot', hindi: 'Dhusrit Patti Dhabb', severity: 2, urgency: 'medium' },
      { id: 'Gray Leaf Spot_multiple', hindi: 'Bahut Dhusrit Dhabb', severity: 3, urgency: 'high' },
      { id: 'Healthy', hindi: 'Swasth Ped', severity: 0, urgency: 'none' },
      { id: 'Leaf rot', hindi: 'Patti Galaav', severity: 3, urgency: 'high' },
      { id: 'Leaf rot_multiple', hindi: 'Bahut Patti Galaav', severity: 4, urgency: 'critical'},
      { id: 'bud rot', hindi: 'Kali Galaav', severity: 5, urgency: 'critical'},
      { id: 'stem bleeding', hindi: 'Tane se Khoon', severity: 4, urgency: 'high' },
      { id: 'stem bleeding_multiple', hindi: 'Bahut Tane se Khoon', severity: 5, urgency: 'critical'},
    ],
    branches: {
      coconut_spots: { desc: 'Patti pe daag — Gray Leaf Spot ya Leaf Rot', q_ids: ['cq2','cq3','cq4','cq5'] },
      coconut_wilt: { desc: 'Stem Bleeding ya Bud Rot symptoms', q_ids: ['cq7','cq8','cq9','cq10'] },
      coconut_yellow: { desc: 'Yellowing — nutrient ya disease', q_ids: ['cq12','cq13','cq14','cq15'] },
      coconut_none: { desc: 'Preventive check — healthy tree', q_ids: ['cq17','cq18','cq19','cq20'] },
    },
  });
};

export const getQuestionSession = (req, res) => {
  const { q1_answer } = req.body;
  const branchMap = { spots:'coconut_spots', wilt:'coconut_wilt', yellow:'coconut_yellow', none:'coconut_none' };
  const branch = branchMap[q1_answer] || 'coconut_none';

  const Q1 = {
    id:'cq1', question:'Aapke naariyal ke ped mein sabse pehle kya symptom dikha?', type:'single',
    options:[
      {id:'spots', label:'🟤 Patti pe daag ya dhabba', desc:'Brown, gray ya kala daag'},
      {id:'wilt', label:'🔴 Tana ya kali mein problem', desc:'Trunk se liquid ya kali gal rahi'},
      {id:'yellow', label:'🟡 Patti peeli ho rahi hai', desc:'Leaves yellow ya dry'},
      {id:'none', label:'✅ Koi symptom nahi, preventive check', desc:'Tree healthy lagta hai'},
    ],
  };

  const ALL_BRANCH_QS = {
    coconut_spots: [
      {id:'cq2', question:'Daag ka rang kaisa hai?', options:[{id:'gray',label:'⬜ Gray/Ash rang'},{id:'brown',label:'🟫 Brown/Dark Brown'},{id:'yellow_edge',label:'🟡 Brown with yellow border'},{id:'black',label:'⬛ Kala daag'}]},
      {id:'cq3', question:'Daag patti ke kahan par hai?', options:[{id:'tip',label:'🔝 Patti ki nauk pe'},{id:'middle',label:'↔️ Beech mein'},{id:'base',label:'⬇️ Neeche jaad ke paas'},{id:'all',label:'📍 Poori patti pe'}]},
      {id:'cq4', question:'Kitni pattiyon pe daag hai?', options:[{id:'one',label:'1️⃣ Ek ya do patti'},{id:'few',label:'🔢 3-5 pattiyan'},{id:'many',label:'📊 Adhi se zyada'},{id:'all',label:'⚠️ Saari pattiyan'}]},
      {id:'cq5', question:'Daag ka size kaisa hai?', options:[{id:'small',label:'🔵 Chota (1-2mm)'},{id:'medium',label:'🟡 Medium (5-10mm)'},{id:'large',label:'🔴 Bada (>1cm)'},{id:'merge',label:'💀 Daag aapas mein jud gaye'}]},
    ],
    coconut_wilt: [
      {id:'cq7', question:'Problem mainly kahan hai?', options:[{id:'trunk',label:'🌴 Tane (trunk) pe'},{id:'bud',label:'🌱 Nai kali/growth pe'},{id:'roots',label:'🌿 Neeche jaad ke paas'},{id:'both',label:'⚠️ Trunk aur kali dono'}]},
      {id:'cq8', question:'Tane se koi liquid nikal raha hai?', options:[{id:'dark_liquid',label:'🩸 Haan, dark brown/red liquid'},{id:'clear',label:'💧 Haan, transparent liquid'},{id:'no',label:'❌ Nahi, koi liquid nahi'},{id:'smell',label:'👃 Liquid hai + buri smell'}]},
      {id:'cq9', question:'Nai patti ya kali ka kya haal hai?', options:[{id:'normal',label:'✅ Normal badi ho rahi hai'},{id:'brown',label:'🟫 Kali brown/black ho gayi'},{id:'small',label:'📉 Patti choti reh gayi'},{id:'dead',label:'💀 Kali/patti mar gayi'}]},
      {id:'cq10', question:'Yeh problem kitne samay se hai?', options:[{id:'week',label:'📅 1 hafte se kam'},{id:'month',label:'🗓️ 1-2 mahine'},{id:'old',label:'📆 3-6 mahine'},{id:'very_old',label:'⏳ 6 mahine se zyada'}]},
    ],
    coconut_yellow: [
      {id:'cq12', question:'Yellowing kahan se shuru hua?', options:[{id:'old_leaves',label:'⬇️ Neeche ki purani pattiyon se'},{id:'new_leaves',label:'⬆️ Upar ki nai pattiyon se'},{id:'all_at_once',label:'📊 Saari pattiyan ek saath'},{id:'patches',label:'🗺️ Kuch jagah kuch jagah'}]},
      {id:'cq13', question:'Patti ki nasubon (veins) ka rang kaisa hai?', options:[{id:'green_vein',label:'🟢 Nasuben hari, baaki peeli'},{id:'all_yellow',label:'🟡 Sab kuch peela'},{id:'brown_vein',label:'🟫 Nasuben bhi brown'},{id:'normal',label:'✅ Nasuben theek hain'}]},
      {id:'cq14', question:'Paani dene ka schedule kya hai?', options:[{id:'too_much',label:'💧 Bahut zyada paani'},{id:'less',label:'🏜️ Kam paani milta hai'},{id:'regular',label:'✅ Regular schedule pe'},{id:'rain',label:'🌧️ Sirf baarish pe depend'}]},
      {id:'cq15', question:'Khaad (fertilizer) kab diya tha?', options:[{id:'recent',label:'📅 1 mahine ke andar'},{id:'months',label:'🗓️ 2-6 mahine pehle'},{id:'long',label:'📆 6+ mahine nahi diya'},{id:'never',label:'❌ Kabhi nahi diya'}]},
    ],
    coconut_none: [
      {id:'cq17', question:'Aap kaun sa coconut variety ughate hain?', options:[{id:'tall',label:'🌴 Tall variety (Tiptur, West Coast)'},{id:'dwarf',label:'🌿 Dwarf variety'},{id:'hybrid',label:'🔬 Hybrid variety'},{id:'unknown',label:'❓ Pata nahi'}]},
      {id:'cq18', question:'Aapka farm kaisi jagah pe hai?', options:[{id:'coastal',label:'🌊 Coastal (samundar ke paas)'},{id:'inland',label:'🏔️ Inland (andar ki taraf)'},{id:'humid',label:'💧 Bahut nami wali jagah'},{id:'dry',label:'☀️ Sukhha ilaqa'}]},
      {id:'cq19', question:'Last spray kab kiya tha?', options:[{id:'recent',label:'📅 1 mahine ke andar'},{id:'months',label:'🗓️ 2-4 mahine pehle'},{id:'long',label:'📆 6+ mahine nahi kiya'},{id:'never',label:'❌ Kabhi nahi'}]},
      {id:'cq20', question:'Aas paas ke pedo mein koi bimari dikhi?', options:[{id:'yes',label:'⚠️ Haan, bahut pedo mein disease hai'},{id:'no',label:'✅ Nahi, sab theek hain'},{id:'some',label:'🔢 Kuch pedo mein thodi problem'},{id:'unknown',label:'❓ Maine check nahi kiya'}]},
    ],
  };

  const branchQs = ALL_BRANCH_QS[branch] || ALL_BRANCH_QS.coconut_none;
  res.json({ success:true, branch, q1_answer, total_questions:5, questions:[Q1, ...branchQs] });
};

export const quickScan = async (req, res) => {
  try {
    const { photoBase64 } = req.body;
    if (!photoBase64) return res.status(400).json({ error: 'photoBase64 required' });
    const aiResult = await callAI('/predict', { image_base64: photoBase64, consultation_id: null, question_answers: {} });
    res.json({ disease: aiResult.disease, confidence: aiResult.confidence, severity: aiResult.severity, is_healthy: aiResult.is_healthy, urgency: aiResult.urgency, top3: aiResult.top3 });
  } catch(e) {
    if (e.message === 'AI_SERVER_DOWN') return res.status(503).json({ error: 'AI server band hai' });
    res.status(500).json({ error: e.message });
  }
};

export const analyze = async (req, res) => {
  try {
    const { photoBase64, photoId, photoIds = [], photoUrls = [], photoCount = 0, consultationId, questionAnswers } = req.body;
    if (!photoBase64 && !photoId) return res.status(400).json({ error: 'photoBase64 ya photoId dena hoga' });

    let base64ToSend = photoBase64;
    const resolvedPhotoIds = Array.isArray(photoIds) && photoIds.length ? photoIds.filter(Boolean) : (photoId ? [photoId] : []);
    let resolvedPhotoUrls = Array.isArray(photoUrls) ? photoUrls.filter(Boolean) : [];
    if (!resolvedPhotoUrls.length && resolvedPhotoIds.length) {
      resolvedPhotoUrls = resolvedPhotoIds.map((id) => `/api/photos/${id}`);
    }
    const resolvedPhotoCount = Number(photoCount) || resolvedPhotoUrls.length || resolvedPhotoIds.length || (photoId ? 1 : 0);

    if (photoId && !photoBase64) {
      const photoDoc = await prisma.photo.findFirst({ where: { photoId, userId: req.user.id } });
      if (!photoDoc) return res.status(404).json({ error: 'Photo nahi mili' });
      base64ToSend = photoDoc.base64;
    }

    console.log(`🥥 Coconut AI analyze — user: ${req.user.id}`);
    let aiResult;
    try {
      aiResult = await callAI('/predict', { image_base64: base64ToSend, consultation_id: consultationId || null, question_answers: questionAnswers || {} });
    } catch(aiErr) {
      if (aiErr.message === 'AI_SERVER_DOWN' || aiErr.message.includes('fetch failed') || aiErr.message.includes('ECONNREFUSED')) {
        console.warn('⚠️ AI Server is down! Proceeding with fallback dummy data so photos are saved.');
        aiResult = { disease: "AI Analysis Pending", disease_hindi: "AI Analysis Pending", confidence: 0, severity: 0, is_healthy: false, urgency: "medium", description: "Python AI server running nahi hai. Lekin aapki photos save ho gayi hain.", treatments: ["AI server on karein aur try karein."], top3: [], model_version: "offline" };
      } else {
        throw aiErr;
      }
    }

    let consultation;
    if (consultationId) {
      await prisma.consultation.updateMany({
        where: { id: consultationId, farmerId: req.user.id },
        data: {
          disease: aiResult.disease, disease_hindi: aiResult.disease_hindi, confidence: aiResult.confidence, severity: aiResult.severity, is_healthy: aiResult.is_healthy, urgency: aiResult.urgency, ai_description: aiResult.description, ai_treatments: aiResult.treatments, ai_top3: aiResult.top3, ai_model: aiResult.model_version, photoUploaded: resolvedPhotoCount > 0, photoId: photoId || resolvedPhotoIds[0] || null, photoUrl: resolvedPhotoUrls[0] || (photoId ? `/api/photos/${photoId}` : null), photoIds: resolvedPhotoIds, photoUrls: resolvedPhotoUrls, photoCount: resolvedPhotoCount, coconut_only: true, updatedAt: new Date()
        }
      });
      consultation = await prisma.consultation.findUnique({ where: { id: consultationId } });
    } else {
      const experts = await prisma.user.findMany({ where: { type: 'expert', available: true } });
      const expert = experts.length > 0 ? experts[Math.floor(Math.random() * experts.length)] : null;
      consultation = await prisma.consultation.create({
        data: {
          farmerId: req.user.id, expertId: expert?.id || null, expertName: expert?.name || 'Pending Assignment', cropId: 'coconut', cropName: 'Naariyal (Coconut)', cropEmoji: '🥥', method: 'photo', photoUploaded: resolvedPhotoCount > 0, photoId: photoId || resolvedPhotoIds[0] || null, photoUrl: resolvedPhotoUrls[0] || (photoId ? `/api/photos/${photoId}` : null), photoIds: resolvedPhotoIds, photoUrls: resolvedPhotoUrls, photoCount: resolvedPhotoCount, disease: aiResult.disease, disease_hindi: aiResult.disease_hindi, confidence: aiResult.confidence, severity: aiResult.severity, is_healthy: aiResult.is_healthy, urgency: aiResult.urgency, ai_description: aiResult.description, ai_treatments: aiResult.treatments, ai_top3: aiResult.top3, ai_model: aiResult.model_version, coconut_only: true, answers: questionAnswers || {}, status: expert ? 'expert_assigned' : 'pending', report: null
        }
      });
      if (resolvedPhotoIds.length) {
        await prisma.photo.updateMany({ where: { photoId: { in: resolvedPhotoIds }, userId: req.user.id }, data: { consultationId: consultation.id } }).catch(() => {});
      }
      await prisma.notification.create({
        data: {
          userId: req.user.id, type: 'consultation', icon: '🥥', title: `AI Report Ready — ${aiResult.disease}`, body: `${aiResult.disease} detected (${aiResult.confidence}% confidence). ${aiResult.is_healthy ? '🌴 Tree healthy hai!' : aiResult.urgency === 'critical' ? '⚠️ Turant action zaroor!' : '📋 Treatment plan ready.'}`, read: false, consultationId: consultation.id
        }
      });
      if (expert) {
        await prisma.notification.create({
          data: {
            userId: expert.id, type: 'new_case', icon: '🥥', title: 'Naya Coconut Case', body: `${aiResult.disease} detected — farmer ka case review karein.`, read: false, consultationId: consultation.id
          }
        });
        await prisma.user.update({ where: { id: expert.id }, data: { totalCases: { increment: 1 } } });
      }
    }

    console.log(`✅ Coconut AI: ${aiResult.disease} (${aiResult.confidence}%) → ${consultation.id}`);
    res.json({ success: true, consultation_id: consultation.id, ai_result: aiResult, consultation });

  } catch(e) {
    console.error('Coconut analyze error:', e);
    res.status(500).json({ error: e.message });
  }
};

export const getCoconutConsultations = async (req, res) => {
  try {
    const q = { cropId: 'coconut', ...(req.user.type === 'expert' ? { expertId: req.user.id } : { farmerId: req.user.id }) };
    const list = await prisma.consultation.findMany({ where: q, orderBy: { createdAt: 'desc' } });
    res.json({ consultations: list, count: list.length });
  } catch(e) { res.status(500).json({ error: 'Coconut consultations nahi mile' }); }
};
