import { prisma } from '../db.js';
import { safe } from '../utils/helpers.js';

export const getExperts = async (req, res) => {
  try {
    const q = { type: 'expert' };
    if (req.query.spec) q.spec = req.query.spec;
    if (req.query.available === 'true') q.available = true;
    const list = await prisma.user.findMany({ where: q });
    res.json({ experts: list.map(safe) });
  } catch (e) { res.status(500).json({ error: 'Experts nahi mile' }); }
};

export const getExpertById = async (req, res) => {
  try {
    const expert = await prisma.user.findFirst({ where: { id: req.params.id, type: 'expert' } });
    if (!expert) return res.status(404).json({ error: 'Expert nahi mila' });
    res.json({ expert: safe(expert) });
  } catch (e) { res.status(500).json({ error: 'Server error' }); }
};

export const updateAvailability = async (req, res) => {
  try {
    const available = !!req.body.available;
    await prisma.user.update({ where: { id: req.user.id }, data: { available } });
    console.log(`👨‍⚕️ Expert ${req.user.id}: available=${available}`);
    res.json({ success: true, available });
  } catch (e) { res.status(500).json({ error: 'Availability update fail' }); }
};

