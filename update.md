# Executive Summary: Perception Pipeline Migration and Stabilization

## 1) Strategic tech-stack change

The perception stack has been migrated from a YOLOv26-oriented detection flow to a GroundingDINO-first architecture.

- Previous posture: fixed-label YOLO detection upstream of mask generation.
- Current posture: open-vocabulary GroundingDINO detection feeding SAM2, with CoTracker and optional XMem for temporal continuity.
- Compatibility commitment: YOLO installation support is retained in setup automation for rollback and legacy experiments.

## 2) Core architecture now in production

- Phase A (keyframes): GroundingDINO detection, association, SAM2 mask extraction.
- Phase B: CoTracker persistent point tracking.
- Phase C: per-frame mask completion using temporal memory and selective refresh.
- Optional path: XMem propagation for long-range temporal stability.

## 3) Detection and fusion policy updates

- Depth-aware detection fusion was introduced and iterated to avoid over-merging.
- Prompt grouping issues were reduced by shifting toward discrete urban prompts.
- Same-label duplicate suppression now uses IoM matching behavior intended for nested/padded boxes.
- Different-label overlap suppression remains IoU-based for safer coexistence of semantically distinct objects.

## 4) Semantic hierarchy and mask interaction improvements

- Semantic depth map expanded/refined for foreground-background ordering.
- `traffic light` is now explicitly represented in depth hierarchy.
- Accessory-vs-human crossfire mitigation added: accessory negatives are blocked from directly collapsing human masks during generation.
- Depth-aware mask subtraction remains responsible for final front-to-back carving.

## 5) Large-background handling and stability

- `mask_guard` configuration is now active and aligned with active GroundingDINO vocabulary.
- Large-stuff labels (skyline/building/road/sidewalk/tree) are preserved against oversize-penalty false rejections.
- Background/large-stuff classes are handled to avoid unstable foreground tracking behavior.

## 6) Prompting and threshold tuning

- Urban preset thresholds were tuned across low/medium/high clutter levels.
- Prompt list updated to include distant/scene-critical entities (for example traffic lights).
- Prompt-cap adjusted to support discrete prompt strategies without truncation.

## 7) Setup and environment changes

- Main app environment: `vidcolor`.
- Detector environment: `gdino310`.
- GroundingDINO installation path is now part of setup automation.
- YOLO package installation remains intentionally present in setup for compatibility support.

## 8) File-level impact highlights

- Detection runtime and config resolution:
	- `src/perception/grounding_dino_detector.py`
	- `scripts/gdino_worker.py`
	- `configs/perception/grounding_dino.yaml`
- Segmentation orchestration, fusion, depth logic, and temporal policy:
	- `src/perception/segmentation.py`
- SAM2 mask candidate behavior and no-box-fallback posture:
	- `src/perception/sam_segmenter.py`
- Setup and dependency footprint:
	- `setup.ps1`
	- `setup_windows.ps1`
	- `requirements.txt`
	- `requirements-gdino310.txt`

## 9) Current operational intent

- GroundingDINO is the default detector in the active pipeline.
- YOLO remains installed as a compatibility option, not the primary execution path.
- The system is tuned to prioritize stable silhouettes, cleaner occlusion carving, and stronger distant-object recall while controlling duplicate fragmentation.

## 10) XMem installation and setup updates

- `setup_windows.ps1` now installs XMem dependencies directly into `vidcolor`.
- Added a base-package check/install step for `torch`, `torchvision`, `opencv-python`, `pillow`, and `tqdm` before XMem runtime dependency installation.
- Added `XMem/requirements.txt` installation in setup flow to cover repo-specific runtime dependencies.
- README setup guidance was updated so `setup_windows.ps1` is the primary setup command and includes explicit XMem install instructions for manual setup.
- `requirements.txt` now reflects XMem-applicable dependencies used by this repository's XMem workflow.
