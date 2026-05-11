import { prisma } from '../db.js';

export const getNotifications = async (req, res) => {
  try {
    const list = await prisma.notification.findMany({ where: { userId: req.user.id }, orderBy: { createdAt: 'desc' } });
    res.json({ notifications: list });
  } catch (e) { res.status(500).json({ error: 'Notifications nahi mile' }); }
};

export const readAllNotifications = async (req, res) => {
  await prisma.notification.updateMany({ where: { userId: req.user.id, read: false }, data: { read: true } });
  res.json({ success: true });
};

export const readNotification = async (req, res) => {
  await prisma.notification.updateMany({ where: { id: req.params.id, userId: req.user.id }, data: { read: true } });
  res.json({ success: true });
};

export const rateExpert = async (req, res) => {
  try {
    const { rating, consultationId } = req.body;
    if(!rating || rating < 1 || rating > 5) return res.status(400).json({ error: 'Rating 1-5 ke beech honi chahiye' });

    const expert = await prisma.user.findFirst({ where: { id: req.params.id, type: 'expert' } });
    if(!expert) return res.status(404).json({ error: 'Expert nahi mila' });

    const oldRating = expert.rating || 4.5;
    const totalCases = expert.totalCases || 1;
    const newRating = ((oldRating * totalCases) + rating) / (totalCases + 1);

    await prisma.user.update({ where: { id: req.params.id }, data: { rating: Math.round(newRating * 10) / 10 } });

    await prisma.notification.create({
      data: {
        userId: req.params.id, type: 'rating', icon: '⭐', title: `Naya Rating Mila — ${rating}/5`, body: `Ek farmer ne aapko ${rating} star diye consultation ke baad.`, read: false
      }
    });

    console.log(`⭐ Rating: expert ${req.params.id} → ${newRating.toFixed(1)}`);
    res.json({ success: true, newRating: Math.round(newRating * 10) / 10 });
  } catch(e) { console.error('rate:', e); res.status(500).json({ error: 'Rating save nahi hua' }); }
};

export const getEarnings = async (req, res) => {
  try {
    const consults = await prisma.consultation.findMany({ where: { expertId: req.user.id } });
    const expert = await prisma.user.findUnique({ where: { id: req.user.id } });
    const fee = expert?.fee || 500;
    const completed = consults.filter(c => c.status === 'completed');
    const now = new Date();
    const thisMonth = completed.filter(c => {
      const d = new Date(c.createdAt);
      return d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
    });
    const pending = consults.filter(c => c.status !== 'completed');
    res.json({
      total: completed.length * fee, thisMonth: thisMonth.length * fee, pending: pending.length * fee,
      completed: completed.length, totalCases: consults.length, feePerCase: fee,
      recentCases: consults.slice(0, 10).map(c => ({
        id: c.id, crop: c.cropName, status: c.status, date: c.createdAt, amount: c.status === 'completed' ? fee : 0,
      })),
    });
  } catch(e) { res.status(500).json({ error: 'Earnings nahi mile' }); }
};

export const getHealth = (_, res) => res.json({ status: 'ok', app: 'BeejHealth API', version: '3.0.0', uptime: process.uptime() });

export const getStats = async (_, res) => {
  try {
    const [total, farmers, experts, consultations, pendingConsults] = await Promise.all([
      prisma.user.count(), prisma.user.count({ where: { type: 'farmer' } }), prisma.user.count({ where: { type: 'expert' } }),
      prisma.consultation.count(), prisma.consultation.count({ where: { status: 'pending' } }),
    ]);
    res.json({ total, farmers, experts, consultations, pendingConsults });
  } catch (e) { res.status(500).json({ error: 'Stats nahi mile' }); }
};

export const getAdminUsers = async (_, res) => {
  const users = await prisma.user.findMany();
  res.json({
    count: users.length,
    users: users.map(u => ({
      id: u.id, name: u.name, mobile: u.mobile, type: u.type,
      district: u.district, available: u.available, totalCases: u.totalCases,
    })),
  });
};

export const resetData = async (_, res) => {
  await Promise.all([
    prisma.user.deleteMany(), prisma.consultation.deleteMany(),
    prisma.notification.deleteMany(), prisma.chat.deleteMany(),
  ]);
  console.log('🗑️  All data cleared');
  res.json({ success: true, message: 'All data cleared. Ab seed karo.' });
};

export const createSupportTicket = async (req, res) => {
  try {
    const { name, mobile, issue, desc } = req.body;
    if (!issue?.trim() || !desc?.trim()) return res.status(400).json({ error: 'Issue aur description required hai' });
    const ticket = await prisma.consultation.create({
      data: {
        farmerId: req.user?.id || 'guest', expertId: null, expertName: 'Support Team', cropId: 'support', cropName: `Support: ${issue}`, cropEmoji: '🆘',
        method: 'support', disease: issue, confidence: 100, severity: 1, status: 'pending', report: null,
        answers: { name: name || '', mobile: mobile || '', desc }
      }
    });
    console.log(`🆘 Support ticket: ${issue} from ${name || 'guest'}`);
    res.json({ success: true, ticketId: ticket.id, message: '24 ghante mein reply milegi' });
  } catch (e) { res.status(500).json({ error: 'Ticket save nahi hua' }); }
};
