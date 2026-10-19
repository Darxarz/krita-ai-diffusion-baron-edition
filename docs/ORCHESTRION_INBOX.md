# Orchestrion inbox ("Send to Krita")

Pictures sent from the website with **Send to Krita** (gallery, comic panel picture) open in Krita as new documents. **Return to website** sends the edited picture back.

## How it works

- Active only when connected to Orchestrion and `GET /api/krita/info` lists the `inbox:1` feature.
- The plugin long-polls `GET /api/krita/inbox?since=<cursor>&wait=25`, re-reads the whole list after any error and every 30 minutes, and applies `removed` ids.
- New items are claimed with `POST /api/krita/inbox/:id/ack` (stable per-installation `deviceId`, computer name as `deviceName`), downloaded from the same website only, verified by size and SHA-256, and opened as a new document. Items already open on another device are not opened automatically; choose them in the **Inbox** menu to take them over.
- The document stores the item id, return target and website origin in the `ai_diffusion/orchestrion_inbox` annotation, so returning still works after saving and reopening the `.kra` (until the item expires after 7 days).
- **Return to website** flattens the document and posts it as raw PNG with `?width=&height=` and an `Idempotency-Key` derived from the item id and the picture bytes: retrying the same picture never creates duplicates, a changed picture becomes a new version. Comic panels are filled by the website (`fit: cover`).
- The device token is sent only as `Authorization: Bearer` to the website origin, never in URLs.

Automatic opening can be turned off with the `orchestrion_inbox_auto_open` setting.
