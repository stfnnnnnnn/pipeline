# Pipeline Update Report

Date: 2026-04-13
Workspace: C:\Users\Administrator\Documents\pipeline
OS: Windows

## 1) Executive Summary

This project was upgraded from a YOLO11 + CoTracker2 baseline to a YOLO26 + CoTracker3 pipeline, with additional hardening for Windows/Conda, CUDA compatibility, GroundingDINO reliability, segmentation quality, and UI observability.

Major outcomes:

- Active detector/tracker defaults now target YOLO26s and CoTracker3.
- GroundingDINO execution is now more portable across machines and more resilient when custom ops are unavailable.
- CoTracker initialization now adapts to checkpoint/window-length mismatches.
- Segmentation quality controls were expanded to reduce full-frame false masks, label confusion, duplicate IDs, and merged person masks.
- UI now shows processing context, compute backend, and live/final processing time.
- README and dependency guidance were updated to reflect the migrated stack.

## 2) What Changed (High Level)

### Detector/Tracker Migration

- YOLO11s -> YOLO26s for active defaults and runtime config.
- CoTracker2 -> CoTracker3 for active defaults and tests.
- New runtime config added: configs/perception/cotracker3.yaml.
- New runtime config added: configs/perception/yolo26_s.yaml.

### Reliability and Portability

- setup_windows.ps1 now does GPU detection, CUDA wheel selection (cu128/cu121), kernel verification, and CPU fallback when needed.
- GroundingDINO wrapper now auto-resolves conda executable and Windows DLL paths when config values are null.
- GroundingDINO worker now falls back to PyTorch deformable attention if custom C++ ops are unavailable.

### Segmentation/Tracking Quality

- Added stronger detection filtering, dedupe, ID matching, adaptive point sampling, and static-scene handling.
- Added scene mask shaping and geometric relabel checks for sky/building/road consistency.
- Added dynamic mask-to-box constraint for people/vehicles/accessories to avoid merging two persons into one instance.
- Added GroundingDINO keyframe interval control to cut overhead while preserving scene signal.

### UX and Observability

- Object detection page now shows current video name.
- Added compute backend display (GPU vs CPU).
- Added live elapsed minutes during processing and final runtime after completion.

## 3) Detailed File-by-File Changes

### Core Pipeline

- src/perception/segmentation.py
	- Active defaults switched to:
		- models/checkpoints/yolo/yolo26s.pt
		- models/checkpoints/cotracker/cotracker3.pth
		- configs/perception/yolo26_s.yaml
		- configs/perception/cotracker3.yaml
	- Added/expanded:
		- detection dedupe by IoU and label
		- ID matching with IoU + center-distance gating
		- adaptive tracking points by mask area
		- static-scene fallback boxes for missing tracked boxes
		- scene mask shaping (sky/road/building/tree)
		- geometry-based scene relabel logic (sky <-> building correction)
		- scene area safeguards to reject implausible near-full-frame masks
		- dynamic mask box clipping and center-component selection for person class
		- runtime_device in return payload
		- optional GDINO throttling via gdino_keyframe_interval

- src/perception/yolo_detector.py
	- Default model changed to yolo26s.pt.
	- Added clear runtime guidance when ultralytics is too old for newer model blocks.

- src/perception/cotracker_wrapper.py
	- Added adaptive constructor logic for different CoTracker signatures.
	- Added runtime retry strategy for window_len based on time_emb mismatch errors.
	- Improved compatibility when checkpoint/runtime expectation differs.

- src/perception/grounding_dino_detector.py
	- Added allow_cpu_fallback config support.
	- Added auto-resolution for conda path, torch lib dir, CUDA bin dir.
	- Added repo-relative path resolution and file existence checks.
	- Added CUDA incompatibility detection and fallback/error guidance.

- scripts/gdino_worker.py
	- Added safer DLL search path setup.
	- Added fallback patch using PyTorch deformable attention when custom ops are missing.

### Configs

- configs/perception/yolo26_s.yaml (new runtime config)
	- model_path switched to yolo26s checkpoint.
	- Added tuned detection_filter values for precision and speed:
		- tighter confidence/area filters
		- gdino_keyframe_interval: 2
		- static scene quality gates
		- dynamic mask clipping controls

- configs/perception/cotracker3.yaml (new)
	- checkpoint: cotracker3.pth
	- v2: false
	- offline: true
	- window_len: 60
	- points_per_instance tuned to 40 for improved throughput.

- configs/perception/grounding_dino.yaml
	- conda_exe, torch_lib_dir, cuda_bin_dir moved to auto-detect (null) defaults.
	- Added allow_cpu_fallback flag.
	- Expanded prompts for building/road variants.
	- Tuned thresholds and max_detections for less noise and lower overhead.

### UI

- pages/object_detection.py
	- Added Processing (<filename>) heading.
	- Added compute backend caption.
	- Added live elapsed processing minutes and persisted final processing time.

- pages/upload.py
	- Minor cleanup: removed redundant Mask Generation heading.

### Dependencies / Docs / Tests

- requirements.txt
	- ultralytics changed from fixed 8.2.0 to >=8.4.0,<9 to support YOLO26-era modules.

- tests/test_backend.py
	- Updated checkpoint paths to yolo26s + cotracker3.
	- Note: current test script still imports segmentation as a top-level module; package import path should be used when running from repo root.

- README.md
	- Updated stack references to YOLO26s and CoTracker3.
	- Added adaptive CUDA wheel guidance and fallback behavior.
	- Added GroundingDINO custom-op validation instructions.
	- Updated key config file references.

## 4) PC/Environment Configuration on This Machine

Configured environments:

- vidcolor
	- Python 3.10
	- Uses requirements.txt
	- Torch install strategy: prefer cu128 when supported, fallback to cu121, then CPU if required.

- gdino310
	- Python 3.10
	- Uses requirements-gdino310.txt
	- GroundingDINO installed from source.
	- transformers pinned to 4.26.1 in this env for compatibility.

Conda path strategy in setup and runtime wrappers:

- Primary: CONDA_EXE from environment.
- Fallback candidates include:
	- C:\ProgramData\miniconda3\condabin\conda.bat
	- C:\ProgramData\anaconda3\condabin\conda.bat
	- user-profile miniconda/anaconda paths.

CUDA/DLL strategy:

- NVIDIA presence is checked via nvidia-smi.
- Driver CUDA version guides cu128 vs cu121 preference.
- Runtime includes verification with a real CUDA tensor operation.
- GroundingDINO DLL directories can auto-detect from env/CUDA when config values are null.

## 5) Error History and Resolution Log

### A) GroundingDINO custom op / DLL issues

Symptoms:

- failures importing groundingdino._C or custom attention op missing
- platform-specific DLL loading issues on Windows

Resolution:

- Added auto DLL directory handling in worker.
- Added fallback to PyTorch deformable attention when custom ops are unavailable.
- Added explicit guidance and checks in README and runtime error paths.

### B) CUDA wheel/device compatibility instability

Symptoms:

- torch install success but CUDA runtime probe failing on some machines.

Resolution:

- setup_windows.ps1 now verifies CUDA with actual tensor kernel execution.
- Added install fallback chain: cu128 -> cu121 -> CPU.

### C) CoTracker mismatch (time_emb shape mismatch, e.g. 60 vs 16)

Symptoms:

- predictor initialization fails with state/shape mismatch when checkpoint and runtime defaults disagree.

Resolution:

- cotracker_wrapper now retries with inferred/common window_len values and adapts constructor usage.
- Introduced cotracker3 runtime config to make parameters explicit.

### D) Segmentation quality errors

Reported issues:

- full-frame false building masks
- building/sky confusion
- road not fully masked in parts
- two people merged into one instance
- duplicate/unstable instance identity

Resolution:

- added scene mask shaping + geometric relabeling guards
- added static mask area limits and sky/road band checks
- added dynamic mask clipping to detection boxes
- for person, keep nearest meaningful component near detection center
- strengthened ID association and dedupe logic
- reduced noisy GDINO frequency via interval setting

### E) Runtime/latency concern (YOLO26 expected faster but observed slower)

Actions taken:

- tightened detection filters
- reduced CoTracker points_per_instance
- reduced GDINO frequency and capped detections
- benchmarked with a generated 30-frame smoke input

Observed benchmark in this workspace:

- YOLO26 config run: 114.74 sec, instances: 38
- YOLO11 config run: 236.64 sec, instances: 50

Note:

- This benchmark includes full pipeline costs (YOLO + SAM2 + CoTracker + GDINO). It is not a pure detector-only benchmark.

## 6) What Is New (Net New Artifacts)

New runtime configs:

- configs/perception/yolo26_s.yaml
- configs/perception/cotracker3.yaml

Updated setup and docs emphasize:

- adaptive, device-aware installation
- robust fallback behavior
- clearer operational checks

UI enhancements:

- runtime backend display
- live and final timing visibility

## 7) Important Clarification: yolo26.yaml vs yolo26_s.yaml

- configs/perception/yolo26.yaml is an Ultralytics architecture definition (model graph/scales/modules).
- configs/perception/yolo26_s.yaml is the app runtime policy config used by segmentation.py (model_path, thresholds, labels, filtering, mask behavior).

Both are valid files, but they serve different layers and are not interchangeable in this app.

## 8) Recommendations for the Stack (Next Steps)

### Recommended current stack

- Keep YOLO26s + CoTracker3 as active defaults.
- Keep ultralytics >=8.4.0,<9.
- Keep GroundingDINO in separate gdino310 env with source install and fallback enabled.

### Recommended tuning workflow

- Tune in this order:
	1) yolo26_s.yaml detection_filter
	2) grounding_dino.yaml thresholds/prompts/max_detections
	3) cotracker3.yaml points_per_instance/window_len
- For speed-first runs, consider:
	- increasing gdino_keyframe_interval to 3
	- lowering points_per_instance to 32 if quality remains acceptable.

### Recommended repository hygiene

- Add/confirm ignore rules for generated outputs:
	- data/intermediate/**
	- __pycache__/**
	- ad-hoc files like 8.3.0
- Keep legacy YOLO11/CoTracker2 config files for comparison only, clearly marked as legacy.

### Recommended test fix

- tests/test_backend.py should import package path (src.perception.segmentation) or run with explicit PYTHONPATH to avoid import errors from repo root.

## 9) Current Known Non-Blocking Noise

- Large numbers of modified/generated mask images and metadata under data/intermediate are expected from pipeline runs and can obscure code diffs.
- Static-analysis warnings about cv2 members are mostly stub/type-check noise; runtime py_compile checks on edited modules passed.

## 10) Final Status

The migration and hardening work is complete at code/config/doc level:

- YOLO11 -> YOLO26: completed for active pipeline.
- CoTracker2 -> CoTracker3: completed for active pipeline.
- Setup/runtime robustness: completed.
- UI timing/backend observability: completed.
- Quality controls for previously reported segmentation issues: implemented and tuned.

Remaining practical work is iterative validation on your real videos and optional cleanup of generated artifacts.

