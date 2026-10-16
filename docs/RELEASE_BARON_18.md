# Krita AI Diffusion Baron Edition 1.53.0-baron.18

Preserve the opacity of manually feathered selections when inpaint Blending is
zero. Previously every nonzero mask pixel became fully opaque when placing the
result, exposing VAE/color drift around the edge. Both inpaint and region refine
now reuse the soft compositing mask. Positive Blending retains the original
threshold/shrink behavior; binary selections remain binary.

No opacity amplification or gamma adjustment is introduced. A wide feather can
still mix old and new content in the transition zone. The reported tenfold fringe
reduction came from an external experiment, not a new visual acceptance test.

Validation: 515 available fast Python tests passed, 111 skipped; mask regression
covers direct alpha, preserving model alpha, and four nonzero blending widths.
Pyright passes. Full Ruff retains the existing unrelated warnings. Inference suite
was attempted: 8 passed before its first server-dependent test failed because
127.0.0.1:8188 was unavailable. Real Krita/ComfyUI visual validation remains open.

Restart Krita after installing. The Android Orchestrion compiler receives the
same patch on the server; an APK update is not required for that connection.
