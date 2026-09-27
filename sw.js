// Minimal service worker: caches the app shell so it works offline once
// visited once. Required (along with manifest.json) for Chrome on Android
// to offer "Install app" / "Add to Home Screen" with standalone display.
const CACHE_NAME = "gusset-predictor-v1";
const ASSETS = ["./", "./index.html", "./model_data.js", "./manifest.json",
                "./icon-192.png", "./icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
