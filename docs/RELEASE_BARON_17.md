# Krita AI Diffusion Baron Edition 1.53.0-baron.17

Orchestrion connections now use https://orchestrion.ink. Saved official website
addresses migrate automatically and retain the encrypted device credential;
custom ComfyUI servers remain unchanged. The existing device credential was
verified against the new domain without starting a paid generation.

Choosing a LoRA in the model gallery now appends `<lora:folder/name>` to the
current generation, edit or regional prompt. It preserves the selected model,
style, existing prompt and explicit weights, and avoids duplicate tags.
Checkpoints retain their existing style creation behavior.

The LoRA catalog refreshes when opening the gallery and every minute while
connected, so models uploaded after login appear in prompt completion. Expired
entries are removed on refresh. The server has a separate one-minute catalog
cache, so a newly uploaded file may need another refresh cycle.

Updates use the new domain. A separate legacy feed on the old domain lets
existing installations download this bridge release before that domain expires.
Restart Krita after installing.

Validation: Python available fast suite 509 passed, 111 skipped; final focused
connection/gallery tests 88 passed; Pyright zero errors; formatting checked.
The repository's pre-existing Ruff warnings remain. Android companion: 0.1.37.
