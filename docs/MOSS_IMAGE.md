# MOSS-Image

MOSS-Image (OpenMOSS, 5B DiT, Apache-2.0) is strong at legible text and object layout. No weights are bundled with Baron Edition.

## Server requirements

- ComfyUI with MOSS support (master after 0.41.2, `CLIPLoader` type `moss_image`). Older servers hide MOSS checkpoints instead of failing at generation time.
- comfyui-tooling-nodes must map the model class `MOSSImage` to `moss-image`, otherwise `/api/etn/model_info` reports `unknown` and the checkpoint is not listed.
- Files:
  - `diffusion_models/moss/moss_image_sft_int8_convrot.safetensors` (`UNETLoader`)
  - `text_encoders/qwen3.5_9b_w4a8_gptq.safetensors` or `qwen3.5_9b_int8_convrot.safetensors` (`CLIPLoader`, type `moss_image`)
  - `vae/flux2-vae.safetensors`

The Qwen Image 2.1 prompt enhancer (`qwen3.5_9b_qwen_image_2.1_pe_*`) shares the encoder prefix and is never selected for MOSS.

## Workflow

- `EmptyFlux2LatentImage`, sides aligned to 16, between 512 and 2048 px, at most 2048² pixels.
- `SkipLayerGuidanceDiTSimple` with single block `22` for the unconditional pass.
- No `ModelSamplingSD3` / `ModelSamplingFlux` / `ModelSamplingAuraFlow`: the shift is baked into the model and these nodes produce noise.
- LoRAs are applied to the model only (`LoraLoaderModelOnly`); the quantized encoder is never patched.
- Inpaint, outpaint and refine use the generic DifferentialDiffusion + `SetLatentNoiseMask` path (no inpaint model, no ControlNet).
- Style `MOSS Image` with sampler preset `MOSS - Euler uniform`: euler / sgm_uniform, 30 steps (minimum 20), CFG 3. The negative prompt is used.

Write prompts as sentences, not tags. Quoted text is rendered, e.g. `a sign that says "OPEN"`.

## Regions

MOSS does not use attention-couple masks. With two or more regions the positive prompt becomes a layout JSON string, built like ComfyUI's `MOSSImageLayoutPrompt`:

```json
{"summary":{"caption":"a fantasy street at dusk"},"items":[{"label":"character","caption":"a red-haired girl in a green cloak","bbox":[-412,-472,388,-52]},{"label":"object","caption":"a wooden tavern sign that says \"TAVERN\"","bbox":[-462,48,-162,448]},{"label":"text","caption":"a wooden tavern sign that says \"TAVERN\"","text":"TAVERN","bbox":[-462,48,-162,448]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}
```

- `summary.caption` is the global prompt (with the style prompt merged in).
- One item per region, `bbox` is the bounding box of the region mask (pixels at least half covered) as `[y0, x0, y1, x1]` in pixels of the image which is actually sampled, origin at its centre; inner edges are floored, outer edges ceiled. For inpaint, the high-resolution refinement and upscale tiles, the box is cropped and scaled like the mask.
- Label: `character` by default, as on the website (with `object` MOSS draws people as creatures or things). Start the region prompt with `object:` for props and signs, `text:` for a text-only box, or `character:` to be explicit. Quoted text (`"…"`, `“…”`, `«…»`) adds a `text` item with the exact text and the same box, whatever the label. A region without a prefix whose prompt quotes text (a sign, a label) is an `object` rather than a `character`, so it does not add an extra figure; an explicit prefix always wins (`character: a girl in a shirt that says "HELLO"`).
- With two or more `character` regions, words like "on the left", "to the right", "слева", "справа" are removed from their captions and the position of the box is appended instead: "on the left of the frame", "in the center" or "on the right of the frame" (box centre in the left, middle or right third). Otherwise MOSS tends to swap the characters. Body parts ("left hand") are kept.
- The global prompt Krita appends to region prompts is removed from the captions; the background region is left out.
- The negative prompt stays plain text. Regional LoRAs are rejected, use global LoRAs. Prompts are not translated into the layout, the Qwen3.5 encoder reads most languages directly.

## Not supported yet

- ControlNet, IP-Adapter and instruction edit: the model has none.
