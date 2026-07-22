# SmartPark — Parking Spot Object Detection

A working, tested, end-to-end system: a from-scratch object detector finds
and classifies parking spots (`empty_spot` / `occupied_spot`) directly in
any full parking-lot photo, served live through a REST API and a browser
upload demo. A second pipeline (occupancy classifier + geometric
misparking check) is kept for cameras with known, calibrated spot
boundaries.

For the full writeup (background, dataset, architecture, training,
evaluation, hyperparameter tuning, benchmarking, discussion, and the
real-photo generalization test), see
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
| Real-photo generalization test | `real_photo_test.py` | ✅ qualitative test on real CNRPark-EXT photos |
| Occupancy classifier (secondary pipeline) | `models/cnn_model.py`, `train.py` | ✅ 97.5% val accuracy (synthetic), 97.04% (real PKLot, historical) |
| Improper-parking geometry check | `improper_parking.py` | ✅ IoU/overflow logic |
| Full inference pipeline (classifier path) | `inference.py` | ✅ image + video support |
| REST API | `api/app.py` | ✅ serves both pipelines live |
| Upload demo UI | `demo/park_check.html` | ✅ drag-and-drop photo upload, live detection |

## Results

**Object detector** (held-out synthetic validation, 45 images / 721 spot instances):

```
mAP@0.5:              98.71%
empty_spot     AP:    98.73%   precision 95.5%  recall 94.1%  F1 94.8%
occupied_spot  AP:    98.68%   precision 99.7%  recall 96.1%  F1 97.9%
```

Benchmarked against two non-learned baselines on the same validation spots:
a majority-class guess (53.3%) and this project's own original classic-CV
heuristic (82.7%) — the trained detector reaches 99.9% classification
accuracy given correct localization.

**Real-photo generalization test** (qualitative, 115 real CNRPark-EXT
photos — a different camera angle with no painted spot lines): the
detector carries over a coarse "cars are roughly here" signal but does not
reliably localize per spot or recognize `empty_spot` in this domain — a
measured limitation, not an unexamined one. See Section 5.1 of the report.

**Occupancy classifier** (secondary pipeline): 97.5% validation accuracy on
synthetic crops; 97.04% on real PKLot data in an earlier session (that
checkpoint currently needs re-training in this environment after a
TensorFlow/Keras version upgrade — see `trained_model_real/training_report.txt`
for the original result).

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
├── train_detector.py                # train the object detector
├── evaluate_detector.py             # precision/recall/F1/AP/mAP
├── hyperparameter_ablation.py       # grid-size + lambda_coord experiments
├── benchmark_baseline.py            # baseline comparison
├── real_photo_test.py               # real-photo qualitative test
├── detector_utils.py                # target encoding, loss, decode+NMS, drawing
├── train.py                         # train the occupancy classifier
├── inference.py                     # classifier + geometry inference pipeline
├── improper_parking.py              # IoU/overflow misparking check
├── validate.py                      # classifier pipeline validation
├── convert_coco_to_crops.py         # real PKLot COCO export -> crops
├── generate_report_samples.py       # sample outputs for submission
├── generate_report_charts.py        # training curve, PR curves, confusion matrix
├── generate_architecture_diagram.py # system architecture figure
├── report_assets/
│   ├── SmartPark_Report.docx
│   ├── SmartPark_Slides.pptx
│   ├── sample_outputs/              # 8 annotated examples + summary.json
│   ├── charts/                      # figures used in the report/slides
│   ├── build_report_docx.py         # regenerates the report from these numbers
│   └── build_slides_pptx.py         # regenerates the slide deck
└── requirements.txt
```

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

# 5. (optional) Real-photo qualitative generalization test
python real_photo_test.py --n 6

# 6. Train the occupancy classifier (secondary pipeline)
python train.py --dataset ./synthetic_dataset --epochs 10 --out ./trained_model

# 7. Start the API + demo
python api/app.py
# then open http://localhost:5050 in a browser --
# drag and drop any photo, or click a sample lot thumbnail
```

`curl` smoke test while the server is running:
```bash
curl http://localhost:5050/api/health
curl -F "image=@synthetic_dataset/full_lots/lot_0000.jpg" http://localhost:5050/api/analyze
```

## Regenerating the report/slides

Both are built from the real numbers above, not hand-typed:
```bash
python report_assets/build_report_docx.py
python report_assets/build_slides_pptx.py
```
Re-run the training/evaluation/ablation/benchmark scripts first if you want
the numbers to reflect a fresh run.

## Known limitations

- The object detector is trained and evaluated on synthetic data; the real
  CNRPark-EXT test (Section 5.1 of the report) shows it does not yet
  generalize precisely to an unfamiliar real camera and marking style.
  Fine-tuning on the target deployment camera's own photos is the clear
  next step before production use.
- The occupancy classifier's real-PKLot checkpoint (`trained_model_real/`)
  needs re-training in this environment (Keras version upgrade); the raw
  PKLot source images are no longer available locally to redo it. 97.04%
  is the genuine, previously-recorded result.
- The geometric "properly parked" check and its classic-CV car-localization
  step are validated on synthetic data only; real photos (shadows, dense
  packing) remain untested for that specific feature.

## What YOU need to do next

1. **Fill in the team info** — `report_assets/SmartPark_Report.docx` and
   `SmartPark_Slides.pptx` have `[FILL IN]` placeholders for group number,
   member names/roles, and the Project Manager. These can't be filled in
   automatically.
2. **(Optional) Validate on real data for your actual deployment target** —
   if you have a specific real camera/lot in mind, a handful of photos from
   it (with spot boundaries marked once) would let you fine-tune both
   pipelines on the real thing rather than relying on synthetic data alone.
