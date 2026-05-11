/**
 * BeejHealth — Seed Script
 * Dummy farmers + experts + consultations + chats create karta hai
 * Run: npm run seed   (from root)
 * OR:  node seed.js   (from backend/)
 */
import bcrypt from 'bcryptjs';
import {
  usersDB, consultationsDB, notificationsDB, chatsDB,
  dbFind, dbFindOne, dbInsert, dbRemove,
} from './db.js';

const ts   = (daysAgo = 0) => new Date(Date.now() - daysAgo * 86400000).toISOString();
const wait = (ms) => new Promise(r => setTimeout(r, ms));

console.log('\n🌱 ════════════════════════════════════════');
console.log('   BeejHealth Seed Script');
console.log('🌱 ════════════════════════════════════════\n');

await wait(600); /* DB load hone do */

/* ── Clear existing data ─────────────────────────────────────── */
await Promise.all([
  dbRemove(usersDB,         {}, { multi: true }),
  dbRemove(consultationsDB, {}, { multi: true }),
  dbRemove(notificationsDB, {}, { multi: true }),
  dbRemove(chatsDB,         {}, { multi: true }),
]);
console.log('🗑️  Purana data clear ho gaya\n');

/* ═══════════════════════════════════════════════════════════════
   FARMERS
═══════════════════════════════════════════════════════════════ */
const FARMERS_DATA = [
  { name: 'Ramesh Patil',   mobile: '9876543210', district: 'Pune',      taluka: 'Haveli',    village: 'Wagholi',     soil: 'Black Cotton',  crops: ['tomato','wheat'],   email: 'ramesh@demo.com'  },
  { name: 'Suresh Kumar',   mobile: '9876543211', district: 'Nashik',    taluka: 'Dindori',   village: 'Manur',       soil: 'Red Laterite',  crops: ['grape','onion'],    email: 'suresh@demo.com'  },
  { name: 'Kavita Devi',    mobile: '9876543212', district: 'Solapur',   taluka: 'Pandharpur',village: 'Shetphal',    soil: 'Sandy Loam',    crops: ['cotton','soybean'], email: 'kavita@demo.com'  },
  { name: 'Mohan Yadav',    mobile: '9876543213', district: 'Nagpur',    taluka: 'Kamptee',   village: 'Hudkeshwar',  soil: 'Alluvial',      crops: ['potato','corn'],    email: 'mohan@demo.com'   },
  { name: 'Priya Shinde',   mobile: '9876543214', district: 'Aurangabad',taluka: 'Paithan',   village: 'Chhatrapati', soil: 'Medium Black',  crops: ['wheat','onion'],    email: 'priya@demo.com'   },
];

console.log('👨‍🌾 Farmers create ho rahe hain...');
const farmers = [];
for (const f of FARMERS_DATA) {
  const h = await bcrypt.hash('farmer123', 10);
  const u = await dbInsert(usersDB, {
    ...f, password: h, type: 'farmer',
    initials:   f.name.split(' ').map(w => w[0]).join('').toUpperCase(),
    verified:   true,
    available:  false,
    rating:     0,
    totalCases: 0,
    bio:        '',
    langs:      'Hindi, Marathi',
    spec:       '',
    fee:        0,
    university: '',
    createdAt:  ts(Math.floor(Math.random() * 30)),
    lastLogin:  ts(),
  });
  farmers.push(u);
  console.log(`  ✅ ${f.name.padEnd(16)} | 📱 ${f.mobile} | 🔑 farmer123`);
}

/* ═══════════════════════════════════════════════════════════════
   EXPERTS
═══════════════════════════════════════════════════════════════ */
const EXPERTS_DATA = [
  { name: 'Dr. Rajesh Kumar',  mobile: '9000000001', spec: 'Plant Pathologist',   fee: 800,  rating: 4.9, totalCases: 1240, university: 'IARI New Delhi',    langs: 'Hindi, English, Punjabi',   district: 'Delhi',         available: true,  bio: '15+ saal ka anubhav. Tomato, wheat, cotton specialist.' },
  { name: 'Dr. Priya Sharma',  mobile: '9000000002', spec: 'Horticulture Expert',  fee: 600,  rating: 4.8, totalCases: 892,  university: 'PAU Ludhiana',      langs: 'English, Hindi, Tamil',     district: 'Punjab',        available: true,  bio: 'Fruit aur vegetable crops mein expert. 10+ saal.' },
  { name: 'Prof. Amit Verma',  mobile: '9000000003', spec: 'Soil Scientist',       fee: 1000, rating: 5.0, totalCases: 2100, university: 'BHU Varanasi',      langs: 'English, Hindi',            district: 'Uttar Pradesh', available: false, bio: 'Soil health aur nutrition ka expert. PhD aur research.' },
  { name: 'Dr. Manoj Desai',   mobile: '9000000004', spec: 'Crop Scientist',       fee: 500,  rating: 4.7, totalCases: 675,  university: 'MPUAT Udaipur',     langs: 'Hindi, Marathi',            district: 'Maharashtra',   available: true,  bio: 'Cereals aur pulses specialist. Affordable consultation.' },
  { name: 'Dr. Kavita Patel',  mobile: '9000000005', spec: 'Horticulture Expert',  fee: 600,  rating: 4.8, totalCases: 755,  university: 'AAU Anand',         langs: 'Hindi, Gujarati',           district: 'Gujarat',       available: false, bio: 'Spice crops aur vegetables. Gujarat aur Maharashtra.' },
  { name: 'Dr. Arun Singh',    mobile: '9000000006', spec: 'Crop Scientist',       fee: 450,  rating: 4.6, totalCases: 430,  university: 'GBPUAT Pantnagar', langs: 'Hindi, English',            district: 'Uttarakhand',   available: true,  bio: 'Young expert. Modern farming techniques.' },
];

console.log('\n👨‍⚕️  Experts create ho rahe hain...');
const experts = [];
for (const e of EXPERTS_DATA) {
  const h = await bcrypt.hash('expert123', 10);
  const u = await dbInsert(usersDB, {
    ...e, password: h, type: 'expert',
    initials: e.name.replace('Dr. ','').replace('Prof. ','').split(' ').map(w => w[0]).join('').toUpperCase(),
    verified: true,
    email:    `${e.name.toLowerCase().replace(/[^a-z]/g,'').slice(0,10)}@expert.com`,
    village:  '',
    taluka:   '',
    soil:     '',
    crops:    [],
    createdAt: ts(Math.floor(Math.random() * 60)),
    lastLogin: ts(),
  });
  experts.push(u);
  const status = e.available ? '🟢 Online' : '🔴 Offline';
  console.log(`  ✅ ${e.name.padEnd(20)} | 📱 ${e.mobile} | 🔑 expert123 | ${status}`);
}

/* ═══════════════════════════════════════════════════════════════
   SAMPLE CONSULTATIONS
═══════════════════════════════════════════════════════════════ */
const CONSULT_SAMPLES = [
  {
    fi: 0, ei: 0, cropId: 'tomato', cropName: 'Tomato', cropEmoji: '🍅',
    disease: 'Early Blight', confidence: 94, severity: 2, status: 'completed',
    answers: { q1:{id:'spots',label:'Daag/Spots'}, q2:{id:'lower',label:'Neeche ke patte'}, q3:{id:'brown',label:'Bhoora'}, q9:{id:'heavy',label:'Zyada baarish'}, q20:{id:'old',label:'2+ hafte pehle'} },
    report: 'Early Blight (Alternaria solani) confirm hua hai. Mancozeb 75% WP — 2.5g/L spray, hafte mein 2 baar. Infected patte hatao. Waterlogging avoid karo.',
    daysAgo: 5,
    chat: [
      { from: 'expert', text: 'Maine aapki tomato ki photo dekhi. Early Blight ke clear symptoms hain. Kab se shuru hua?' },
      { from: 'farmer', text: '4-5 din pehle se daag dikhe. Bahut tezi se fail raha hai.' },
      { from: 'expert', text: 'Achha. Pichle 2 hafte mein spray nahi kiya? Kyunki Q20 mein apne pehle se kaha tha.' },
      { from: 'farmer', text: 'Haan doctor sahab, spray khatam ho gaya tha.' },
      { from: 'expert', text: 'Theek hai. Maine full report bhej di. Aaj hi Mancozeb spray shuru karo. 7 din mein control ho jaayega. 💚' },
    ],
  },
  {
    fi: 1, ei: 1, cropId: 'wheat',  cropName: 'Wheat',  cropEmoji: '🌾',
    disease: 'Leaf Rust',     confidence: 89, severity: 1, status: 'expert_assigned',
    answers: { q1:{id:'spots',label:'Daag/Spots'}, q2:{id:'upper',label:'Upar ke patte'}, q3:{id:'red',label:'Laal'}, q9:{id:'light',label:'Thodi baarish'}, q20:{id:'week',label:'1 hafte mein'} },
    report: null,
    daysAgo: 2,
    chat: [
      { from: 'expert', text: 'Wheat Leaf Rust ke symptoms dekhe. Ye early stage hai. Propiconazole spray karo. Aur photos bhejo.' },
      { from: 'farmer', text: 'Theek hai doctor. Aur kya karna chahiye?' },
    ],
  },
  {
    fi: 2, ei: 3, cropId: 'cotton', cropName: 'Cotton', cropEmoji: '🌸',
    disease: 'Bacterial Blight', confidence: 86, severity: 3, status: 'pending',
    answers: { q1:{id:'spots',label:'Daag/Spots'}, q2:{id:'all',label:'Poori fasal'}, q3:{id:'black',label:'Kaala'}, q9:{id:'heavy',label:'Zyada baarish'}, q20:{id:'never',label:'Kabhi nahi'} },
    report: null, daysAgo: 1, chat: [],
  },
  {
    fi: 3, ei: 0, cropId: 'potato', cropName: 'Potato', cropEmoji: '🥔',
    disease: 'Late Blight',   confidence: 91, severity: 3, status: 'completed',
    answers: { q1:{id:'spots',label:'Daag/Spots'}, q2:{id:'lower',label:'Neeche'}, q3:{id:'brown',label:'Bhoora'}, q9:{id:'heavy',label:'Baarish'}, q20:{id:'old',label:'2 hafte'} },
    report: 'Late Blight (Phytophthora infestans) — HIGH SEVERITY. TURANT KAREIN: Metalaxyl + Mancozeb spray. Infected plants hatao. Drainage improve karo. 7 din baad dobara check.',
    daysAgo: 8,
    chat: [
      { from: 'expert', text: 'Yeh Late Blight hai aur ye bahut serious hai. Turant action chahiye.' },
      { from: 'farmer', text: 'Ghabhrana chahiye kya doctor? Poori fasal kharab ho jaayegi?' },
      { from: 'expert', text: 'Ghabrao mat. Abhi spray shuru karo toh fasal bach sakti hai. Report mein sab detail hai. Follow karo.' },
      { from: 'farmer', text: 'Bahut bahut shukriya doctor sahab! Kal se spray karunga.' },
    ],
  },
  {
    fi: 4, ei: 1, cropId: 'onion',  cropName: 'Onion',  cropEmoji: '🧅',
    disease: 'Purple Blotch',  confidence: 88, severity: 2, status: 'expert_assigned',
    answers: { q1:{id:'yellow',label:'Peele Patte'}, q5:{id:'some',label:'Thode gir rahe'}, q10:{id:'high',label:'Zyada nami'}, q21:{id:'urea',label:'Urea'}, q22:{id:'week',label:'1 hafte mein'} },
    report: null, daysAgo: 3, chat: [
      { from: 'expert', text: 'Purple Blotch fungal infection hai. Mancozeb ya Chlorothalonil spray try karo.' },
    ],
  },
];

console.log('\n📋 Consultations create ho rahe hain...');
const createdConsults = [];
for (const s of CONSULT_SAMPLES) {
  const farmer = farmers[s.fi];
  const expert = experts[s.ei];
  const c = await dbInsert(consultationsDB, {
    farmerId:      farmer._id,
    expertId:      expert._id,
    expertName:    expert.name,
    cropId:        s.cropId,
    cropName:      s.cropName,
    cropEmoji:     s.cropEmoji,
    method:        'photo',
    photoUploaded: true,
    answers:       s.answers,
    disease:       s.disease,
    confidence:    s.confidence,
    severity:      s.severity,
    status:        s.status,
    report:        s.report,
    createdAt:     ts(s.daysAgo),
    updatedAt:     ts(s.status === 'completed' ? s.daysAgo - 1 : s.daysAgo),
  });
  createdConsults.push(c);

  /* Chat messages */
  let chatDelta = s.daysAgo * 3600;
  for (const msg of (s.chat || [])) {
    const senderId   = msg.from === 'expert' ? expert._id   : farmer._id;
    const senderName = msg.from === 'expert' ? expert.name  : farmer.name;
    await dbInsert(chatsDB, {
      consultationId: c._id,
      senderId,
      senderName,
      senderType: msg.from,
      text:       msg.text,
      createdAt:  new Date(Date.now() - chatDelta * 1000).toISOString(),
    });
    chatDelta -= 1800;
  }

  /* Notifications */
  await dbInsert(notificationsDB, {
    userId: farmer._id, type: 'consultation', icon: '🔬',
    title:  `AI Report Ready — ${s.cropName}`,
    body:   `${s.disease} detected (${s.confidence}% confidence). ${expert.name} assigned.`,
    read:   s.status === 'completed',
    consultationId: c._id, createdAt: ts(s.daysAgo),
  });
  if (s.status === 'completed') {
    await dbInsert(notificationsDB, {
      userId: farmer._id, type: 'report_ready', icon: '✅',
      title:  `Expert Report Ready — ${s.cropName}!`,
      body:   `${expert.name} ne aapki ${s.cropName} ki report bhej di.`,
      read:   false, consultationId: c._id,
      createdAt: ts(s.daysAgo - 1),
    });
  }
  await dbInsert(notificationsDB, {
    userId: expert._id, type: 'new_case', icon: '📋',
    title:  `Naya Case — ${s.cropName}`,
    body:   `${s.disease} suspected from ${farmer.name}.`,
    read:   s.status !== 'pending',
    consultationId: c._id, createdAt: ts(s.daysAgo),
  });

  const status = s.status.replace('_', ' ');
  console.log(`  ✅ ${s.cropName.padEnd(10)} | ${s.disease.padEnd(20)} | ${status}`);
}

/* ═══════════════════════════════════════════════════════════════
   SUMMARY
═══════════════════════════════════════════════════════════════ */
console.log('\n🌱 ════════════════════════════════════════');
console.log('   SEED COMPLETE!\n');
console.log('👨‍🌾 FARMER CREDENTIALS:');
console.log('   ─────────────────────────────────────────');
FARMERS_DATA.forEach(f => {
  console.log(`   📱 ${f.mobile}  🔑 farmer123  👤 ${f.name}`);
});
console.log('\n👨‍⚕️  EXPERT CREDENTIALS:');
console.log('   ─────────────────────────────────────────');
EXPERTS_DATA.forEach(e => {
  const dot = e.available ? '🟢' : '🔴';
  console.log(`   📱 ${e.mobile}  🔑 expert123  👤 ${e.name} ${dot}`);
});
console.log('\n🔑 OTP LOGIN: Koi bhi 6-digit number chalega (e.g. 123456)');
console.log('🌱 ════════════════════════════════════════\n');

process.exit(0);
