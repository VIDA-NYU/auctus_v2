# syntax=docker/dockerfile:1

# ---- Stage 1: build the static bundle ----
FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies against the committed lockfile first so this layer is
# cached and only re-runs when package.json / package-lock.json change.
COPY package.json package-lock.json ./
RUN npm ci

# Copy the rest of the source and produce the optimized production build.
# Vite emits to /app/dist (default outDir).
COPY . .

# Bake the backend API URL into the bundle at build time. Vite exposes env vars
# prefixed with VITE_ to import.meta.env, so `npm run build` picks this up. When
# unset it resolves to an empty string, and the app falls back to
# http://localhost:8000 (see src/hooks/*.js), keeping local builds zero-config.
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ---- Stage 2: serve with nginx ----
FROM nginx:alpine AS runtime

# Replace the stock server block with our SPA-aware config (history fallback).
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Ship only the compiled assets from the build stage.
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
