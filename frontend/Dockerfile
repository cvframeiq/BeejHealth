# ── BeejHealth Frontend (Vite + React) ───────────────────────
FROM node:22-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# Optional: point the app at a production API at build time
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

ENV PORT=4173
EXPOSE 4173

# Serve the built static files (Dokploy handles routing/proxy)
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "4173"]
