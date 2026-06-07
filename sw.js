// Self-destroying service worker.
//
// This site previously shipped an sw-precache service worker. Its generator
// (Gulp) is no longer functional, so the precache manifest froze and the worker
// began serving stale CSS/JS — to visitors after a deploy, and locally during
// development. This replacement worker exists only to undo that: it takes
// control, deletes every cache, unregisters itself, and reloads open pages so
// they fall back to plain network fetches.
//
// The browser always revalidates sw.js itself (it is not served from the
// worker's own cache), so any visitor still controlled by the old worker will
// pick this up and clean themselves out. Once enough time has passed for that to
// happen, sw.js can be deleted entirely.

self.addEventListener('install', function () {
  self.skipWaiting();
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys()
      .then(function (keys) {
        return Promise.all(keys.map(function (key) { return caches.delete(key); }));
      })
      .then(function () { return self.registration.unregister(); })
      .then(function () { return self.clients.matchAll(); })
      .then(function (clients) {
        clients.forEach(function (client) {
          if ('navigate' in client) { client.navigate(client.url); }
        });
      })
  );
});
