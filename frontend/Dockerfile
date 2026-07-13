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
RUN npm run build

# ---- Stage 2: serve with nginx ----
FROM nginx:alpine AS runtime

# Replace the stock server block with our SPA-aware config (history fallback).
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Ship only the compiled assets from the build stage.
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
