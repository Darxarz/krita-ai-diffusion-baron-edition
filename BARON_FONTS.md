# Expressive fonts in Baron Edition

Version 1.53.0-baron.8 includes 96 original publicly licensed font families,
with per-family licenses and source hashes in `ai_diffusion/fonts/manifest.json`.
90 use SIL OFL 1.1 and six use Apache 2.0. Thirty contain the complete Russian
alphabet including Ё/ё. The selection spans horror, fantasy, blackletter,
calligraphy, vintage display, comic, stencil and other distinct styles.

The extension registers these fonts with Qt and places missing font files in
Krita's font resource folder, preserving existing files. Restart Krita after
updating. They are available to the text tool; prompt fields still use the UI font.

To register the bundled fonts for other Windows applications, run:

```
python scripts/install_baron_fonts.py ai_diffusion/fonts build/fonts-installed.json
```

The installer uses the current user's Windows font directory and registry,
validates font/license hashes, preserves existing registrations, broadcasts the
font change and writes a receipt. No administrator privileges are required.
Already running programs may need a restart.

The bundle contains no Chiller, Papyrus or other proprietary Windows font files.
Its original license notices accompany both the plugin and the public Android APK.

This PC's installation receipt is `build/fonts-installed.json`; the plugin backup
is `build/plugin-backup-before-fonts-008`. The installed Krita text engine's
Fontconfig library successfully loaded all 96 resource font files in a separate
verification process. This is not a visual test of the running Krita text panel.
