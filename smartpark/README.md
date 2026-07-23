# SmartPark — Parking Spot Object Detection

A working, tested, end-to-end system: a from-scratch object detector finds
and classifies parking spots (`empty_spot` / `occupied_spot`) directly in
any full parking-lot photo, served live through a REST API and a browser
upload demo. A second pipeline (occupancy classifier + geometric
misparking check) is kept for cameras with known, calibrated spot
boundaries.

The live checkpoint (`trained_detector_multibox_mixed/spot_detector.keras`)
predicts 3 boxes per grid cell (not 1) and was fine-tuned on synthetic
**and** real PKLot data together. Getting here took four iterations, each
exposing a real failure mode worth knowing about before you trust any of
this on your own photos -- see "The fine-tuning journey" below.

For the full writeup (background, dataset, architecture, training,
evaluation, hyperparameter tuning, benchmarking, discussion, and both the
zero-shot cross-domain test and the real-data fine-tuning results), see
[`report_assets/SmartPark_Report.docx`](report_assets/SmartPark_Report.docx)
and the slide deck
[`report_assets/SmartPark_Slides.pptx`](report_assets/SmartPark_Slides.pptx).
This README is the practical "how to run it" companion.

## What's built and tested

| Component | File | Status |
|---|---|---|
| Synthetic detection dataset generator | `data/generate_synthetic_data.py` | ✅ 300 full-lot images, 4,422 labeled spot instances, randomized layouts |
| Object detector (primary deliverable) | `models/detector_model.py`, `train_detector.py` | ✅ trained from scratch, 98.71% mAP@0.5 |
| Detector evaluation (P/R/F1/AP/mAP) | `evaluate_detector.py` | ✅ tested |
| Hyperparameter ablations | `hyperparameter_ablation.py` | ✅ grid resolution + coordinate-loss-weight experiments |
| Baseline benchmarking | `benchmark_baseline.py` | ✅ majority-class / classic-CV / trained-detector comparison |
| Zero-shot cross-domain test | `real_photo_test.py` | ✅ qualitative test on real CNRPark-EXT photos (unfamiliar domain) |
| Real-data detector fine-tuning | `train_detector.py --resume_from`, `convert_coco_to_detection.py` | ✅ 47.71% mAP@0.5 on real PKLot photos |
| Crop/scale augmentation | `augmentation.py`, `train_detector.py --augment_crops` | ✅ fixes scale/density generalization; caused catastrophic forgetting alone |
| Rehearsal-based mixed fine-tuning (single-box, superseded) | `train_detector_mixed.py` | ✅ 93.31% synthetic mAP + 51.39% real mAP |
| Multi-box-per-cell architecture (**live default**) | `models/detector_model.py` (`NUM_BOXES_PER_CELL=3`) | ✅ 90.60% synthetic mAP + **68.76% real mAP** -- eliminates grid-cell collisions |
| Occupancy classifier (secondary pipeline) | `models/cnn_model.py`, `train.py` | ✅ 97.5% val accuracy (synthetic), 98.69% (real PKLot) |
| Improper-parking geometry check | `improper_parking.py` | ✅ IoU/overflow logic |
| Full inference pipeline (classifier path) | `inference.py` | ✅ image + video support |
| REST API | `api/app.py` | ✅ serves both pipelines live |
| Upload demo UI | `demo/park_check.html` | ✅ drag-and-drop photo upload, live detection |
| Streamlit demo | `streamlit_app.py` | ✅ same demo, presentation-friendly UI, runs in its own venv |

## Results

**Object detector, live checkpoint** (`trained_detector_multibox_mixed`: 3
boxes per grid cell, fine-tuned on synthetic + real PKLot data together):

```
Synthetic val mAP@0.5:  90.60%   (45 images, 721 spot instances)
Real PKLot val mAP@0.5: 68.76%   (186 images, 10,320 spot instances)
Classification accuracy given correct localization: 94.8%
```

Benchmarked against two non-learned baselines on synthetic validation spots:
a majority-class guess (53.3%) and this project's own original classic-CV
heuristic (82.7%) — the trained detector reaches 99.9% classification
accuracy given correct localization on that set.

### The fine-tuning journey (why this specific checkpoint)

Four variants were actually trained and compared -- this wasn't the first
thing tried, and the failures along the way are as informative as the final
number:

| Variant | Synthetic mAP | Real mAP | Notes |
|---|---|---|---|
| Base: synthetic only, 1 box/cell | 98.71% | -- (never tested on real boxes until fine-tuned) | The original OD deliverable |
| v1: real-only fine-tune, 1 box/cell | not re-tested | 47.71% | Worked on the standard real val split, but only 3-5 detections on two arbitrary real stock photos (way denser/higher-res than any training data) |
| v2: real + crop augmentation, 1 box/cell, no rehearsal | 6.50% (collapsed) | 49.36% | Augmentation fixed the stress-test photos dramatically, but **catastrophic forgetting**: fine-tuning on real data alone overwrote synthetic performance |
| v3: synthetic + real + augmentation (rehearsal), 1 box/cell | 93.31% | 51.39% | Recovered synthetic performance *and* improved real performance further -- but still under-detected badly on the densest stress-test photo (17 detections) |
| **v4: same rehearsal recipe, 3 boxes/cell (adopted) — live default** | **90.60%** | **68.76%** | Directly fixed v3's residual gap: real mAP +17 points, stress-test detections 17→38, by removing the one-box-per-cell recall ceiling itself |

v3's residual gap traced back to the detector's core design, not the
training recipe: **one predicted box per grid cell**. Real PKLot lots
average ~57 spots/image against a 16×16 = 256-cell grid, so two spot
centers sharing a cell forced one to be dropped during training, capping
recall no matter how the model was fine-tuned. Measuring collision rate
directly (not assuming) across grid sizes *and* boxes-per-cell counts
showed 3 boxes/cell at the *same* 16×16 resolution eliminates it entirely:

```
grid=16, 1 box/cell:  28.25% of real ground-truth boxes dropped to collisions
grid=16, 2 box/cell:   3.10%
grid=16, 3 box/cell:   0.00%   <- adopted; cheaper than the finer-grid alternative (32x32 alone: 3.10%)
```

The zero-shot cross-domain test (115 real CNRPark-EXT photos -- a totally
different camera angle with no painted lines) is a separate, harder result
that the architecture fix does not address: the detector carries over a
coarse "cars are roughly here" signal but does not reliably localize per
spot or recognize `empty_spot` in that domain. Domain transfer needs the
model to actually see data resembling the target camera; it does not
happen for free.

Classification itself stays reliable throughout (94.8% given correct
localization, vs. 99.86% synthetic) -- every stage of this journey confirms
the real-data gap is about *finding* spots in a dense scene, not
misclassifying them once found. See Sections 5.2-5.4 of the report for the
full story, including all four before/after stress-test images -- this is
the single most useful thing to read before trusting this on a real
deployment.

**Occupancy classifier** (secondary pipeline): 97.5% validation accuracy on
synthetic crops; **98.69%** on real PKLot data (freshly retrained on the same
real export used for the detector fine-tuning above).

## Project structure

```
smartpark/
├── data/
│   └── generate_synthetic_data.py   # synthetic detection + classifier dataset
├── models/
│   ├── detector_model.py            # object detector architecture (NUM_BOXES_PER_CELL=3)
│   └── cnn_model.py                 # occupancy classifier architecture
├── api/
│   └── app.py                       # Flask REST API, serves both pipelines
├── demo/
│   └── park_check.html              # upload demo UI (served at "/")
├── train_detector.py                # train/fine-tune the object detector (--resume_from, --augment_crops)
├── train_detector_mixed.py          # rehearsal fine-tune: synthetic + real + augmented together (live checkpoint recipe)
├── augmentation.py                  # random crop/zoom augmentation (scale/density generalization)
├── evaluate_detector.py             # precision/recall/F1/AP/mAP
├── hyperparameter_ablation.py       # grid-size + lambda_coord experiments
├── benchmark_baseline.py            # baseline comparison
├── real_photo_test.py               # zero-shot cross-domain test (CNRPark-EXT)
├── streamlit_app.py                 # presentation-friendly demo UI (own venv, see below)
├── detector_utils.py                # target encoding, loss, decode+NMS, drawing (multi-box aware)
├── train.py                         # train the occupancy classifier
├── inference.py                     # classifier + geometry inference pipeline
├── improper_parking.py              # IoU/overflow misparking check
├── validate.py                      # classifier pipeline validation
├── convert_coco_to_crops.py         # real PKLot COCO export -> classifier crops
├── convert_coco_to_detection.py     # real PKLot COCO export -> detector annotations.json
├── generate_report_samples.py       # sample outputs for submission
├── generate_report_charts.py        # training curve, PR curves, confusion matrix
├── generate_architecture_diagram.py # system architecture figure
├── report_assets/
│   ├── SmartPark_Report.docx
│   ├── SmartPark_Slides.pptx
│   ├── sample_outputs/              # 8 synthetic annotated examples + summary.json
│   ├── sample_outputs_real/         # 8 real PKLot annotated examples (v3) + summary.json
│   ├── sample_outputs_multibox/     # 8 real PKLot annotated examples (v4, adopted) + summary.json
│   ├── charts/                      # synthetic-data figures used in the report/slides
│   ├── charts_real/                 # real-data figures (v3 fine-tuning results)
│   ├── charts_multibox/             # real-data figures (v4 multi-box results, collision-rate chart)
│   ├── build_report_docx.py         # regenerates the report from these numbers
│   └── build_slides_pptx.py         # regenerates the slide deck
└── requirements.txt
```

Not tracked in git (large and/or reproducible from a source you provide):
`pklot_raw/` (the raw PKLot COCO export), `real_pklot_dataset/` and
`real_dataset_v2/` (converted from it), `real_cnrpark_sample/` (the
CNRPark-EXT zero-shot test images). Regenerate them with the conversion
scripts above once you have the raw PKLot export (see "How to run
everything").

## How to run everything

```bash
# 0. Set up environment (Python 3.9+; a venv keeps this isolated)
python3 -m venv .venv && source .venv/bin/activate
pip install tensorflow opencv-python flask numpy matplotlib python-docx python-pptx
# (use "tensorflow" not "tensorflow-cpu" on macOS -- the latter has no macOS wheels)

# 1. Generate the synthetic detection + classifier dataset
python data/generate_synthetic_data.py --out ./synthetic_dataset --n_lots 300

# 2. Train the object detector (the primary deliverable)
python train_detector.py --dataset ./synthetic_dataset --epochs 40 --out ./trained_detector

# 3. Evaluate it (precision/recall/F1/AP/mAP)
python evaluate_detector.py --dataset ./synthetic_dataset --model ./trained_detector/spot_detector.keras

# 4. (optional) Hyperparameter ablations + baseline benchmark
python hyperparameter_ablation.py --epochs 15
python benchmark_baseline.py

# 5. (optional) Zero-shot cross-domain test (CNRPark-EXT, unfamiliar camera)
python real_photo_test.py --n 6

# 6. Train the occupancy classifier (secondary pipeline)
python train.py --dataset ./synthetic_dataset --epochs 10 --out ./trained_model

# 7. (optional, needs a raw PKLot COCO export -- see note below) Fine-tune
# the detector and retrain the classifier on real data
python convert_coco_to_detection.py --images_dir ./pklot_raw/test \
    --annotations ./pklot_raw/test/_annotations.coco.json --out ./real_pklot_dataset
python train_detector.py --dataset ./real_pklot_dataset --epochs 25 \
    --resume_from ./trained_detector/spot_detector.keras --lr 1e-4 --out ./trained_detector_real
python evaluate_detector.py --dataset ./real_pklot_dataset --model ./trained_detector_real/spot_detector.keras

python convert_coco_to_crops.py --images_dir ./pklot_raw/test \
    --annotations ./pklot_raw/test/_annotations.coco.json --out ./real_dataset_v2
python train.py --dataset ./real_dataset_v2 --epochs 12 --out ./trained_model_real

# 8. (recommended) Fine-tune with rehearsal instead of step 7's real-only
# fine-tune -- real-only fine-tuning generalizes poorly to arbitrary real
# photos; fixing that with crop augmentation alone causes catastrophic
# forgetting of synthetic performance (see Results above). Rehearsal (mixing
# synthetic data back into the fine-tune) fixes both at once. Because
# models/detector_model.py now predicts NUM_BOXES_PER_CELL=3 boxes per cell
# (see below), this step already trains and fine-tunes the current,
# collision-free architecture -- no separate script needed for that part.
python train_detector_mixed.py --synthetic_dataset ./synthetic_dataset \
    --real_dataset ./real_pklot_dataset --epochs 25 --augment_crops 4 \
    --resume_from ./trained_detector/spot_detector.keras --lr 1e-4 --out ./trained_detector_multibox_mixed
python evaluate_detector.py --dataset ./synthetic_dataset --model ./trained_detector_multibox_mixed/spot_detector.keras
python evaluate_detector.py --dataset ./real_pklot_dataset --model ./trained_detector_multibox_mixed/spot_detector.keras

# 9. Start the API + demo (uses trained_detector_multibox_mixed by default)
python api/app.py
# then open http://localhost:5050 in a browser --
# drag and drop any photo, or click a sample lot thumbnail
```

Step 7 needs a raw PKLot COCO-format export (a Roboflow export of PKLot with
`images/` + `_annotations.coco.json`, `space-empty`/`space-occupied`
categories) at `./pklot_raw/<split>/`. This project's own copy came from a
shared Google Drive link, not a public URL worth hardcoding here -- get your
own export from Roboflow Universe (search "PKLot") or the original dataset
page (web.inf.ufpr.br/vri/databases/parking-lot-database), then point the
conversion scripts at wherever you extract it.

**On the multi-box architecture**: `models/detector_model.py` predicts
`NUM_BOXES_PER_CELL=3` boxes per grid cell rather than 1, at the same 16x16
resolution -- this is now the default for every training command above, not
an opt-in flag. It exists because the one-box-per-cell version measurably
capped recall on dense real lots (see Results); if you want to reproduce
the *original* single-box numbers from the report's early sections, set
`NUM_BOXES_PER_CELL = 1` in that file before training (the target
encoding/loss/decoding in `detector_utils.py` adapt automatically).

`curl` smoke test while the server is running:
```bash
curl http://localhost:5050/api/health
curl -F "image=@synthetic_dataset/full_lots/lot_0000.jpg" http://localhost:5050/api/analyze
```

## Streamlit demo (presentation-friendly)

`streamlit_app.py` is the same demo with a nicer UI for presenting live --
tabs, metrics, a results table. It's a pure HTTP client of the Flask API
above (it never imports TensorFlow itself), and **must run in its own,
separate virtual environment**:

```bash
# one-time setup, in a second terminal
python3 -m venv .venv_streamlit
source .venv_streamlit/bin/activate
pip install streamlit requests

# each time (with api/app.py already running in the *other* venv/terminal)
source .venv_streamlit/bin/activate
streamlit run streamlit_app.py
```

Do **not** `pip install streamlit` into the main `.venv` -- in this
environment it upgrades a shared native dependency in a way that breaks
TensorFlow for every process using that venv, not just ones that import
streamlit (we hit this directly: it took down the Flask API too, and
required rebuilding `.venv` from scratch to fix). Two separate venvs
talking over HTTP avoids the conflict entirely.

## Regenerating the report/slides

Both are built from the real numbers above, not hand-typed:
```bash
python report_assets/build_report_docx.py
python report_assets/build_slides_pptx.py
```
Re-run the training/evaluation/ablation/benchmark scripts first if you want
the numbers to reflect a fresh run.

## Known limitations

- The detector does **not** generalize zero-shot to a genuinely unfamiliar
  camera/marking style (tested on CNRPark-EXT — different angle, no painted
  lines). It needs to see data resembling its deployment camera, which it
  can learn to do quickly (25 epochs) once given real examples, per the
  PKLot fine-tuning result above. The multi-box architecture fix does not
  change this -- it addresses density/scale within a familiar domain, not
  transfer to a wholly unfamiliar one.
- **Fine-tuning on real data alone causes catastrophic forgetting**: real
  mAP improved (49.36%) but synthetic mAP collapsed (98.71% → 6.50%). The
  live checkpoint uses rehearsal (mixing synthetic data back into the
  fine-tune) to avoid this — if you retrain further, always re-evaluate on
  the *original* synthetic validation set, not just the new domain, or a
  regression like this can go completely unnoticed.
- **Synthetic mAP costs a little more with 3 boxes/cell than with 1**
  (90.60% vs. 93.31%, both rehearsal-fine-tuned) — a real, small trade-off
  for the +17-point real-data gain. Tuning the synthetic:real:augmented
  mixing ratio further, now that architecture is no longer the bottleneck,
  is the clearest remaining lever.
- The geometric "properly parked" check and its classic-CV car-localization
  step are validated on synthetic data only; real photos (shadows, dense
  packing) remain untested for that specific feature.

## What YOU need to do next

1. **Fill in the team info** — `report_assets/SmartPark_Report.docx` and
   `SmartPark_Slides.pptx` have `[FILL IN]` placeholders for group number,
   member names/roles, and the Project Manager. These can't be filled in
   automatically.
2. **(Optional) Tune the rehearsal mixing ratio** — `train_detector_mixed.py`
   currently uses all available synthetic + real + augmented data at a fixed
   ratio; weighting real/augmented data more heavily might close the
   remaining synthetic-mAP gap without giving back real-world recall.
3. **(Optional) Validate on your actual deployment camera** — if you have a
   specific real lot in mind, a handful of photos from it (with spot
   boundaries marked once) would let you fine-tune both pipelines on that
   camera specifically, the same way this session did for PKLot generally.
