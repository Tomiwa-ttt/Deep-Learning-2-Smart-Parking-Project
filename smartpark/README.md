# SmartPark — Parking Spot Object Detection

A working, tested, end-to-end system: a from-scratch object detector finds
and classifies parking spots (`empty_spot` / `occupied_spot`) directly in
any full parking-lot photo, served live through a REST API and a browser
upload demo. A second pipeline (occupancy classifier + geometric
misparking check) is kept for cameras with known, calibrated spot
boundaries.

The live checkpoint (`trained_detector_mixed/spot_detector.keras`) was
fine-tuned on synthetic **and** real PKLot data together, after two earlier
attempts revealed real failure modes worth knowing about before you trust
any of this on your own photos -- see "The fine-tuning journey" below.

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
| Rehearsal-based mixed fine-tuning (**live default**) | `train_detector_mixed.py` | ✅ 93.31% synthetic mAP + 51.39% real mAP -- best of all variants |
| Occupancy classifier (secondary pipeline) | `models/cnn_model.py`, `train.py` | ✅ 97.5% val accuracy (synthetic), 98.69% (real PKLot) |
| Improper-parking geometry check | `improper_parking.py` | ✅ IoU/overflow logic |
| Full inference pipeline (classifier path) | `inference.py` | ✅ image + video support |
| REST API | `api/app.py` | ✅ serves both pipelines live |
| Upload demo UI | `demo/park_check.html` | ✅ drag-and-drop photo upload, live detection |
| Streamlit demo | `streamlit_app.py` | ✅ same demo, presentation-friendly UI, runs in its own venv |

## Results

**Object detector, live checkpoint** (`trained_detector_mixed`, fine-tuned on
synthetic + real PKLot data together):

```
Synthetic val mAP@0.5:  93.31%   (45 images, 721 spot instances)
Real PKLot val mAP@0.5: 51.39%   (186 images, 10,320 spot instances)
```

Benchmarked against two non-learned baselines on synthetic validation spots:
a majority-class guess (53.3%) and this project's own original classic-CV
heuristic (82.7%) — the trained detector reaches 99.9% classification
accuracy given correct localization on that set.

### The fine-tuning journey (why this specific checkpoint)

Three variants were actually trained and compared -- this wasn't the first
thing tried, and the failures along the way are as informative as the final
number:

| Variant | Synthetic mAP | Real mAP | Notes |
|---|---|---|---|
| Base: synthetic only | 98.71% | -- (never tested on real boxes until fine-tuned) | The original OD deliverable |
| v1: real-only fine-tune | not re-tested | 47.71% | Worked on the standard real val split, but only 3-5 detections on two arbitrary real stock photos (way denser/higher-res than any training data) |
| v2: real + crop augmentation, no rehearsal | 6.50% (collapsed) | 49.36% | Augmentation fixed the stress-test photos dramatically, but **catastrophic forgetting**: fine-tuning on real data alone overwrote synthetic performance |
| **v3: synthetic + real + augmentation (rehearsal) — live default** | **93.31%** | **51.39%** | Recovered synthetic performance *and* improved real performance further -- best of all three |

The zero-shot cross-domain test (115 real CNRPark-EXT photos -- a totally
different camera angle with no painted lines) is a separate, harder result:
the detector carries over a coarse "cars are roughly here" signal but does
not reliably localize per spot or recognize `empty_spot` in that domain,
even with the adopted checkpoint. Domain transfer needs the model to
actually see data resembling the target camera; it does not happen for free.

Real PKLot lots are ~4x denser than the synthetic dataset (~57 vs. ~15
spots/image), which pushes the 16x16 grid's one-box-per-cell collision rate
to 28.2% (vs. 0.27% synthetic) — capping achievable recall near ~72%
regardless of model quality. Classification itself stays nearly as reliable
on real data as on synthetic; the real-data gap is concentrated in
localization recall, not misclassification. See Sections 5.2-5.3 of the
report for the full story, including the two stress-test photos and the
before/after images -- this is the single most useful thing to read before
trusting this on a real deployment.

**Honest residual gap**: the adopted (v3) checkpoint still finds noticeably
fewer cars than v2 on the single most extreme stress-test photo (17 vs. 54
detections) -- rehearsal recovered synthetic performance at some cost to
the very-highest-density case specifically. v3 was still adopted because it
never catastrophically fails on the project's own primary benchmark, but
tuning the synthetic:real:augmented mixing ratio further is a real,
unfinished next step, not a solved problem.

**Occupancy classifier** (secondary pipeline): 97.5% validation accuracy on
synthetic crops; **98.69%** on real PKLot data (freshly retrained on the same
real export used for the detector fine-tuning above).

## Project structure

```
smartpark/
├── data/
│   └── generate_synthetic_data.py   # synthetic detection + classifier dataset
├── models/
│   ├── detector_model.py            # object detector architecture
│   └── cnn_model.py                 # occupancy classifier architecture
├── api/
│   └── app.py                       # Flask REST API, serves both pipelines
├── demo/
│   └── park_check.html              # upload demo UI (served at "/")
├── train_detector.py                # train/fine-tune the object detector (--resume_from, --augment_crops)
├── train_detector_mixed.py          # rehearsal fine-tune: synthetic + real + augmented together (live checkpoint)
├── augmentation.py                  # random crop/zoom augmentation (scale/density generalization)
├── evaluate_detector.py             # precision/recall/F1/AP/mAP
├── hyperparameter_ablation.py       # grid-size + lambda_coord experiments
├── benchmark_baseline.py            # baseline comparison
├── real_photo_test.py               # zero-shot cross-domain test (CNRPark-EXT)
├── streamlit_app.py                 # presentation-friendly demo UI (own venv, see below)
├── detector_utils.py                # target encoding, loss, decode+NMS, drawing
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
│   ├── sample_outputs_real/         # 8 real PKLot annotated examples + summary.json
│   ├── charts/                      # synthetic-data figures used in the report/slides
│   ├── charts_real/                 # real-data figures (fine-tuning results)
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

# 8. (optional, recommended) Fine-tune with rehearsal instead of step 7's
# real-only fine-tune -- this is the version actually served by the API.
# Real-only fine-tuning generalizes poorly to arbitrary real photos; fixing
# that with crop augmentation alone causes catastrophic forgetting of
# synthetic performance (see Results above) -- rehearsal (mixing synthetic
# data back into the fine-tune) fixes both at once.
python train_detector_mixed.py --synthetic_dataset ./synthetic_dataset \
    --real_dataset ./real_pklot_dataset --epochs 25 --augment_crops 4 \
    --resume_from ./trained_detector/spot_detector.keras --lr 1e-4 --out ./trained_detector_mixed
python evaluate_detector.py --dataset ./synthetic_dataset --model ./trained_detector_mixed/spot_detector.keras
python evaluate_detector.py --dataset ./real_pklot_dataset --model ./trained_detector_mixed/spot_detector.keras

# 9. Start the API + demo (uses trained_detector_mixed by default)
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

- **Recall on dense real lots is the main open gap, and it's a specific,
  measured one**: the object detector's one-box-per-cell design drops 28.2%
  of real PKLot ground-truth boxes to grid-cell collisions (real lots
  average ~57 spots/image vs. ~15 for the synthetic dataset), capping
  achievable recall near ~72%. Classification itself is nearly as reliable
  on real data as synthetic (94.5% vs. 99.86%) — this is a localization
  capacity problem, not a "needs more data" problem. A finer grid (32x32+)
  or a multi-box-per-cell/anchor-based head is the concrete next fix.
- The detector does **not** generalize zero-shot to a genuinely unfamiliar
  camera/marking style (tested on CNRPark-EXT — different angle, no painted
  lines). It needs to see data resembling its deployment camera, which it
  can learn to do quickly (25 epochs) once given real examples, per the
  PKLot fine-tuning result above.
- **Fine-tuning on real data alone causes catastrophic forgetting**: real
  mAP improved (49.36%) but synthetic mAP collapsed (98.71% → 6.50%). The
  live checkpoint uses rehearsal (mixing synthetic data back into the
  fine-tune) to avoid this — if you retrain further, always re-evaluate on
  the *original* synthetic validation set, not just the new domain, or a
  regression like this can go completely unnoticed.
- **The adopted (rehearsal) checkpoint still under-performs the
  augmentation-only variant on the single most extreme real photo tested**
  (17 vs. 54 detections on a ~150-car dense lot) — rehearsal traded a little
  of that specific density-robustness back for retaining synthetic
  performance. Tuning the synthetic:real:augmented mixing ratio is a real,
  unfinished next step.
- The geometric "properly parked" check and its classic-CV car-localization
  step are validated on synthetic data only; real photos (shadows, dense
  packing) remain untested for that specific feature.

## What YOU need to do next

1. **Fill in the team info** — `report_assets/SmartPark_Report.docx` and
   `SmartPark_Slides.pptx` have `[FILL IN]` placeholders for group number,
   member names/roles, and the Project Manager. These can't be filled in
   automatically.
2. **(Optional) Try a finer grid or anchor-based head** — the collision-rate
   math above points directly at this as the highest-leverage next change
   for dense real deployments.
2b. **(Optional) Tune the rehearsal mixing ratio** — `train_detector_mixed.py`
   currently uses all available synthetic + real + augmented data at a fixed
   ratio; weighting real/augmented data more heavily might close the
   remaining density gap without giving back synthetic performance.
3. **(Optional) Validate on your actual deployment camera** — if you have a
   specific real lot in mind, a handful of photos from it (with spot
   boundaries marked once) would let you fine-tune both pipelines on that
   camera specifically, the same way this session did for PKLot generally.
