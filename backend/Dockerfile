# ── BeejHealth Backend (Node + Express + Prisma) ──────────────
FROM node:22-alpine

# Prisma needs OpenSSL at runtime
RUN apk add --no-cache openssl

WORKDIR /app

# Install dependencies first (better layer caching)
COPY package*.json ./
RUN npm ci --omit=dev

# Copy source
COPY . .

# Generate the Prisma client (does NOT need a live DB connection)
RUN npx prisma generate

ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

# DATABASE_URL is provided at runtime (external database — see note below)
CMD ["node", "server.js"]
