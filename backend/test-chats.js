import { prisma } from './db.js';

async function test() {
  const chats = await prisma.chat.findMany();
  console.log('Chats in DB:', chats.length);
  console.log(chats.slice(-5));
}

test().finally(() => prisma.$disconnect());
