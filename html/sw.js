// TradeForge SW kill-switch
// Purpose: remove stale service workers and caches that can cause blank/old pages.

self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    try {
      const keys = await caches.keys();
      await Promise.all(keys.map((k) => caches.delete(k)));
    } catch (_) {}

    try {
      await self.registration.unregister();
    } catch (_) {}

    try {
      const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
      for (const c of clients) {
        try { c.navigate(c.url); } catch (_) {}
      }
    } catch (_) {}
  })());
});

// Never intercept network during cleanup mode.
self.addEventListener('fetch', () => {});
