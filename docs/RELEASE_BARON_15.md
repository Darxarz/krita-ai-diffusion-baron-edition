# Krita AI Diffusion Baron Edition 1.53.0-baron.15

Opening a project saved on Android could abort loading AI state with
`factor was '2', but expected <class 'float'>`, leaving the result history and
prompts unavailable even though the drawing layers still opened.

- Accept integer JSON numbers for persisted float settings without accepting
  booleans or strings as numbers.
- Preserve integer-valued unblur settings instead of replacing them with defaults.
- Recover generation and edit prompt banks from older Android annotations.
  Desktop edits are retained when reopening a subsequently saved project.
- Existing files can be reopened without recreating their history or modifying
  their drawing layers. Install the plugin update and restart Krita first.

Validation covers numeric loading, actual ModelSync restoration of result
images, transparency, placement, applied flags, prompts, and re-saving. A local
Android project containing 285 history batches was loaded read-only with all
285 images decoded; its original file hash remained unchanged. Test images
were reduced in memory to limit the validation process's RAM usage.

The companion Android 0.1.36 release writes decimal float values and current
prompt banks directly into Python-compatible annotations.
