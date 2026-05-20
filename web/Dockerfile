# syntax=docker/dockerfile:1.7

# Build the static SvelteKit bundle.
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Serve the static output with Caddy. The outer reverse-proxy handles TLS.
FROM caddy:2-alpine
COPY --from=builder /app/build /srv
COPY Caddyfile /etc/caddy/Caddyfile
EXPOSE 80
