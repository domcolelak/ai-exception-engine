/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // API_BASE_URL is deliberately NOT declared under `env`. That key inlines the
  // value at build time, which would bake the build machine's URL into the
  // image and silently ignore the runtime setting -- so the same image could
  // never point at a different backend. Server components read
  // process.env.API_BASE_URL directly instead, which resolves at request time.
};

export default nextConfig;
