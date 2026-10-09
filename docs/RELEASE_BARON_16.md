# Krita AI Diffusion Baron Edition 1.53.0-baron.16

On Windows, updating while Krita was running could fail with `WinError 5` when
replacing a registered `.ttf` file. The plugin rolled back instead of completing.

- Packaged fonts now live in `fonts/bundled`, allowing the existing updater to
  install this release without overwriting font files held by older versions.
- Qt registers fonts from memory, so future updates can replace the bundled
  files while Krita remains open.
- The updater skips files whose content is unchanged.
- All 96 public font families and their licenses remain included.
- Includes Baron .15's Android .kra history and generation/edit prompt recovery.

Restart Krita after installation to load the new plugin. If an already open
update dialog still targets .15, restart Krita before checking again.

Validation includes font registration and Cyrillic coverage, actual Windows
file replacement while Qt holds a registered font, updater rollback and
preservation of settings, and the existing history recovery tests.
