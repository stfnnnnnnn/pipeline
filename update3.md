# Update 3 Comprehensive Repository Report

Generated: 2026-04-23

## 1) Scope and Source of Truth

This report consolidates repository evolution, active working-tree changes, and conversation-level technical findings using the following in-repo artifacts:

- `update.md` (prior migration and stabilization summary)
- `README.md` (current operational documentation and setup instructions)
- `setup_windows.ps1`, `requirements.txt` (environment and dependency behavior)
- `scripts/prefetch_sd15_inpaint_assets.py`, `scripts/run_sd1.5_controlnet_inpaint.py` (new SD1.5/ControlNet colorization flow)
- `configs/model/sd15_controlnet.yaml`, `data/annotations/object_color_prompts.json` (model/runtime/prompt settings)
- `src/diffusion/color_only_diffusion.py`, `src/reconstruction/photometric_stabilizer.py` (core diffusion and photometric utility behavior)
- `tests/unit/test_color_only_lock.py` (new unit coverage)
- `0001-lora-added.patch` and `allfiles.txt` (historical snapshot artifact)
- Current changed-files metadata and generated outputs in `data/intermediate/*` and `data/output_colorized/*`

Note on constraints:
- The host shell does not currently expose `git` CLI, so commit-log chronology could not be pulled directly in this session.
- This report therefore uses repository files and built-in changed-file metadata as primary evidence.

## 2) Executive Summary

The repository has progressed from a detector-centric perception stack to a broader end-to-end video colorization platform with:

1. A GroundingDINO-first perception pipeline (with SAM2, CoTracker, and optional XMem), while retaining YOLO compatibility.
2. A new SD1.5 inpainting + ControlNet-based per-instance colorization subsystem, including model prefetch tooling and runtime config.
3. Expanded setup automation for robust Conda/Torch/GPU handling and automatic model asset preparation.
4. Large-scale generated/updated artifact output across masks, metadata, and debug renders from recent inference runs.
5. New test coverage around luminance locking in the diffusion subsystem.

## 3) Cumulative Architectural and Feature Changes

### 3.1 Perception Stack Migration and Stabilization (from prior updates)

Based on `update.md` and `README.md`, the major platform direction is:

- Detector migration: legacy YOLO-first flow to GroundingDINO open-vocabulary detection as primary path.
- Production pipeline phases:
	- Phase A: GroundingDINO + SAM2 on keyframes.
	- Phase B: CoTracker for temporal point continuity.
	- Phase C: mask completion/refresh over all frames.
	- Optional: XMem propagation for long-range memory stability.
- Fusion and semantics tuning:
	- Depth-aware detection and mask fusion updates.
	- Duplicate suppression refinements (IoM/IoU behavior by case).
	- Hierarchy and overlap behavior improvements (for example traffic light depth handling, accessory-human mask safety rules).

### 3.2 Environment and Installer Hardening

`setup_windows.ps1` now implements a robust setup workflow:

- Conda executable discovery across environment variable, PATH, and common install locations.
- Env create/update semantics via `conda env ... --environment-spec environment.yml`.
- CUDA-aware Torch installation with fallback strategy:
	- Detect NVIDIA GPU and driver CUDA version.
	- Attempt cu121/cu128 wheel installs in preferred order.
	- Validate via CUDA kernel probe.
	- Fall back to CPU Torch if needed.
- Dedicated GroundingDINO installation path with custom-op verification and fallback messaging.
- Automatic installation of XMem runtime dependencies.
- Automatic prefetch of SD1.5 inpainting + ControlNet assets during setup.
- Continued installation of YOLO-related package(s) for compatibility workflows.

### 3.3 SD1.5 ControlNet Colorization Subsystem (new functional area)

The repository now contains a dedicated per-instance colorization path:

- `src/diffusion/color_only_diffusion.py`
	- Defines `ColorOnlyConfig` and `InstanceColorizer`.
	- Uses `StableDiffusionControlNetInpaintPipeline` with canny control.
	- Converts to grayscale guidance and applies luminance lock in LAB space.
- `scripts/run_sd1.5_controlnet_inpaint.py`
	- Loads frame/mask/metadata artifacts.
	- Parses object color hints JSON.
	- Builds prompts per instance and applies sequential inpainting per frame.
	- Writes colorized outputs and per-instance debug control/mask images.
- `scripts/prefetch_sd15_inpaint_assets.py`
	- Prefetches base inpainting and ControlNet model repos into local Hugging Face cache.
- `configs/model/sd15_controlnet.yaml`
	- Centralizes model IDs, dtype/device, inference parameters, and IO paths.
- `data/annotations/object_color_prompts.json`
	- Provides object class to color/style hints for prompt conditioning.

### 3.4 Validation Additions

- `tests/unit/test_color_only_lock.py` adds unit validation ensuring luminance lock preserves LAB L-channel behavior.

## 4) Current Working-Tree Change Inventory (This Snapshot)

Current changed-file metadata indicates very large run-generated churn:

- At least 2063 additional changed files were identified in the saved changed-file listing.
- Top-level distribution from that listing:
	- `data`: 2054 paths
	- `src`: 3 paths
	- repository root: 3 paths
	- `scripts`: 2 paths
	- `tests`: 1 path

### 4.1 High-Signal Non-Data Files Changed

The following non-data files are explicitly listed as changed in this snapshot:

- `requirements.txt`
- `setup_windows.ps1`
- `scripts/prefetch_sd15_inpaint_assets.py`
- `scripts/run_sd1.5_controlnet_inpaint.py`
- `src/diffusion/color_only_diffusion.py`
- `src/reconstruction/photometric_stabilizer.py`
- `tests/unit/test_color_only_lock.py`
- `update3.md`

Additionally, built-in diff output in this session showed changes for:

- `README.md`
- `configs/model/sd15_controlnet.yaml`
- `data/annotations/object_color_prompts.json`

### 4.2 Data and Output Artifact Churn

Large artifact updates include:

- Many mask PNG changes under `data/intermediate/masks/frame_*/instance_*.png`.
- Metadata updates in `data/intermediate/metadata/segmentation_tracking_metadata.json`.
- Large debug/render output under `data/output_colorized/debug/*`.
- Colorized frame outputs under `data/output_colorized/frame_*.png`.

Interpretation:
- Most of this churn appears to be generated runtime output from recent segmentation/colorization runs rather than hand-authored code edits.

## 5) Important Upstream Repositories and External Components

The repository currently depends on, integrates, or references the following major external projects/models:

1. GroundingDINO (IDEA-Research)
	 - Installed in dedicated `gdino310` env and used as primary open-vocabulary detector.
2. Segment Anything 2 (Meta)
	 - Installed via pip source and used for segmentation mask generation.
3. CoTracker (Meta)
	 - Installed via pip source and used for temporal correspondence/tracking.
4. XMem
	 - Optional memory-based temporal mask propagation and consistency.
5. Hugging Face Diffusers ecosystem
	 - `runwayml/stable-diffusion-inpainting`
	 - `lllyasviel/sd-controlnet-canny`
	 - Used by new colorization subsystem.
6. Ultralytics/YOLO
	 - Retained as compatibility path, not the primary detector path.

## 6) Conversation-Level Findings (This Session)

A focused debug review was done on `scripts/run_sd1.5_controlnet_inpaint.py` and `src/diffusion/color_only_diffusion.py` to answer why output looked uncolorized.

Key findings:

1. Primary behavior issue:
	 - Sequential per-instance processing re-enters grayscale conversion on each call.
	 - This can effectively wash out prior instance colorization, leaving final frames near-monochrome.
2. Prompt-hint matching is exact by normalized label key.
	 - Labels not present in `object_color_prompts.json` fall back to generic natural color prompts.
	 - Label mismatches (for example variants like `facade` versus `building facade`, `skyline horizon` versus `sky`) reduce explicit color guidance coverage.
3. LoRA scene styling is not active in this execution path.
	 - The reviewed SD1.5 ControlNet runner does not load LoRA adapters.
	 - LoRA utilities exist elsewhere in repo but are not wired into this specific script path.

Conclusion for that question:
- The dominant reason for weak/no visible colorization in this path is logic/conditioning flow, not solely missing user hints.
- Missing or mismatched hints can reduce color intent strength, but they are secondary to the grayscale reset behavior.

## 7) Risk and Operational Impact

### 7.1 High impact

- Large generated artifact churn can obscure meaningful source diffs and slow review cycles.
- Colorization path currently risks low visible effect due to sequential grayscale reconditioning.

### 7.2 Medium impact

- Exact-label prompt mapping creates brittle behavior when detection labels vary semantically.
- Mixed compatibility and primary paths (YOLO retained, GroundingDINO default) require clear operational discipline to avoid accidental path confusion.

### 7.3 Low impact

- Presence of historical artifact files (`0.0.25`, `0001-lora-added.patch`, `allfiles.txt`) is useful for auditing but may confuse day-to-day change tracking if undocumented.

## 8) Recommended Next Actions

1. Colorization correctness
	 - Update the sequential inpaint flow so prior colorized regions are preserved across instance passes.
	 - Add regression tests for multi-instance color persistence.

2. Prompt resilience
	 - Add alias/synonym mapping for labels (for example `facade` -> `building facade`, `skyline horizon` -> `sky`).
	 - Consider confidence-based prompt fallback policy with controlled color priors.

3. Repository hygiene
	 - Separate generated artifacts from source control scope (or isolate to dedicated run-output branches).
	 - Keep concise changelog entries for high-signal code changes versus generated data updates.

4. Traceability
	 - Restore/verify shell `git` availability in dev environment for richer in-session historical reporting and faster audits.

## 9) File-by-File Importance Map

- Core platform docs:
	- `README.md`
	- `update.md`
- Setup and env stability:
	- `setup_windows.ps1`
	- `requirements.txt`
- SD1.5 ControlNet colorization features:
	- `scripts/prefetch_sd15_inpaint_assets.py`
	- `scripts/run_sd1.5_controlnet_inpaint.py`
	- `src/diffusion/color_only_diffusion.py`
	- `configs/model/sd15_controlnet.yaml`
	- `data/annotations/object_color_prompts.json`
	- `tests/unit/test_color_only_lock.py`
- Large generated outputs and run artifacts:
	- `data/intermediate/masks/*`
	- `data/intermediate/metadata/segmentation_tracking_metadata.json`
	- `data/output_colorized/*`

## 10) Final Status

This repository currently combines:

- A production-oriented perception stack migration (GroundingDINO + SAM2 + tracking/memory options),
- A newly integrated SD1.5/ControlNet colorization subsystem,
- Significant generated artifact churn from recent execution,
- And a clearly identified colorization flow issue requiring targeted correction for stable visible results.

This document is intended to serve as a complete operational snapshot for current repo state, recent feature additions, and critical technical findings from the latest debugging conversation.
