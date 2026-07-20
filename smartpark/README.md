# SmartPark AI — Project Package

This package contains a **working, tested, end-to-end pipeline**:
CNN occupancy detection → geometric "properly parked" check → REST API →
mobile app demo UI. Everything has been run and validated on a generated
synthetic dataset so you can see it work today. The next step is swapping
in real training data.

## What's already built and tested

| Component | File | Status |
|---|---|---|
| Synthetic data generator | `data/generate_synthetic_data.py` | ✅ generates labeled lot images + spot crops |
| CNN model | `models/cnn_model.py` | ✅ trains, 96.7% val accuracy on synthetic set |
| Training script | `train.py` | ✅ tested |
| Improper-parking geometry check | `improper_parking.py` | ✅ IoU/overflow logic, unit-tested |
| Full inference pipeline | `inference.py` | ✅ image + video support, 96% occupancy / 97.3% proper-parking accuracy across 60 test lots |
| Validation script | `validate.py` | ✅ scores the whole pipeline against ground truth |
| REST API | `api/app.py` | ✅ tested locally, serves live spot status as JSON |
| Mobile app demo | `demo/campuspark_demo.html` | ✅ interactive, seeded with real model output |

Run `python validate.py` yourself any time to reproduce the accuracy numbers.

## Results on synthetic data (initial sanity check)
```
Occupancy accuracy:        576/600 = 96.0%
Proper-parking accuracy:   321/330 = 97.3%
```
This was a sanity check that the pipeline works end-to-end before touching
real data — synthetic data is much cleaner than real photos.

## Results on REAL data (PKLot, done)
The CNN was retrained on real PKLot images (COCO-format export from
Roboflow: 1,242 real lot photos, 70,684 real labeled parking spaces,
`space-empty` / `space-occupied`), using `convert_coco_to_crops.py` to turn
the per-image bounding-box annotations into the same crop format as the
synthetic run.

```
Final validation accuracy (full held-out set, 14,136 real crops): 97.04%
```
Spot-checked against ground truth on 4 individual real lot photos:
```
2013-03-18_07_05_01: 39/40 correct
2013-01-16_12_00_07: 26/28 correct
2013-03-19_12_05_07: 37/40 correct
2013-04-10_08_30_02: 37/40 correct
   (139/148 = 93.9%)
```
This is a genuinely strong, real result — in the same range as the accuracy
reported in the original PKLot research papers.

### Known limitation: "properly parked" check needs work on real photos
The `improper_parking.py` IoU/geometry logic and the `find_car_bbox_in_spot()`
car-localization step in `inference.py` were tuned and validated against the
synthetic dataset (clean asphalt color, no shadows). On the real PKLot photos,
this classic-CV localization step is not yet reliable — real images have
shadows, texture, and lighting variation the simple color-distance approach
doesn't handle. **The occupancy classifier (97% above) is solid; the
"improperly parked" feature still needs either threshold recalibration
against real images or a real object detector (e.g. a small YOLO model)
in place of `find_car_bbox_in_spot()` before you can trust it on real data.**
This is exactly the "Option B" stretch goal flagged earlier — good to mention
as future work in your report, not a blocker for your core result.

---

## What YOU need to do next

### 1. Get real training data (no photography required)
Download one or both of these free, public datasets:

- **PKLot**: https://web.inf.ufpr.br/vri/databases/parking-lot-database/
  (12,000+ labeled images, 3 lots, multiple weather conditions)
- **CNRPark-EXT**: http://cnrpark.it/
  (~150,000 labeled patches, 9 camera angles, varying occlusion)

Both are organized by folders that are easy to convert into the same format
`train.py` already expects:
```
your_dataset/
  spot_crops/
    empty/      <- put all "Empty" labeled crops here
    occupied/   <- put all "Occupied" labeled crops here
```
(PKLot in particular ships already split this way per weather/lot — you may
just need to merge folders.)

### 2. Retrain on real data
```bash
python train.py --dataset ./pklot_dataset --epochs 15 --out ./trained_model_real
```
This produces `trained_model_real/parking_spot_cnn.keras` — same format,
drop-in replacement for the synthetic model.

### 3. Re-run validation
```bash
python validate.py   # update MODEL_PATH to your new model first
```
Report this real accuracy number — this is the number for your final report.

### 4. (Optional, stronger result) Fine-tune the "properly parked" geometry thresholds
`improper_parking.py` has two tunable constants:
- `iou_threshold` (default 0.55)
- `overflow_threshold` (default 0.30)

Real photos have messier lighting/angles than synthetic data, so you may
need to loosen these slightly. Try a few values against a handful of manually
labeled real "bad parking" examples and pick what separates them best.

### 5. Swap the car-localization step for something more robust (optional, stretch goal)
`inference.py`'s `find_car_bbox_in_spot()` currently uses simple OpenCV
color-distance thresholding — it works well on clean/synthetic images but
will be less reliable on real photos with shadows and varied lighting. If
you have time, replacing it with a small pretrained YOLO model (fine-tuned
on CARPK or your own data) for car detection will make the improper-parking
check meaningfully more robust. This is exactly the "Option B" stretch goal
discussed earlier (oriented bounding boxes via YOLOv8-OBB / DOTA).

### 6. Point the API at your real lot images
`api/app.py` currently reads from `synthetic_dataset/full_lots/`. Point
`LOTS_DIR` and `MANIFEST_PATH` at your real lot's images and a one-time
spot-boundary calibration file (same JSON shape as `manifest.json`: a list
of `{id, bbox}` per spot).

### 7. Regenerate the demo app with real results
Re-run the snippet in `demo/` that extracts `run_inference()` output and
injects it into `campuspark_demo_template.html`, this time using your real
lot's images, so the demo you show your professor reflects real detections.

---

## How to run everything today (synthetic data, already working)

```bash
pip install tensorflow-cpu opencv-python flask --break-system-packages

# 1. Generate synthetic data (or skip -- already included in this package)
python data/generate_synthetic_data.py --out ./synthetic_dataset --n_lots 60

# 2. Train
python train.py --dataset ./synthetic_dataset --epochs 10 --out ./trained_model

# 3. Run inference on one lot image + view annotated output
python inference.py --image ./synthetic_dataset/full_lots/lot_0000.jpg \
    --spots ./synthetic_dataset/manifest.json \
    --model ./trained_model/parking_spot_cnn.keras --lot_name "Lot 0000"

# 4. Validate across the whole dataset
python validate.py

# 5. Start the API
python api/app.py
# then in another terminal:
curl http://localhost:5050/api/lots/lot_0000/status

# 6. Open demo/campuspark_demo.html in any browser to see the mobile app UI
```

---

## What I'd need from you to go further

To turn this from "demo with synthetic data" into "demo with real data,"
I'd need one of:
- **Downloaded PKLot or CNRPark-EXT data** (or a subset) uploaded here, so I
  can retrain the model on it directly, **or**
- **A specific real parking lot** you want this built for (a few photos or
  a short video of it), so I can calibrate spot boundaries and fine-tune on
  it, **or**
- Confirmation you'll download PKLot/CNRPark-EXT yourself and want me to
  just hand off the current code as-is (already the case with this package)

For turning the demo into an actual installable phone app (not just an
in-browser mockup), I'd need to know:
- **iOS, Android, or both?** (affects whether we go React Native / Flutter /
  native)
- Do you want it connecting to a **real live backend** (needs a server to
  host `api/app.py`, e.g. a free tier on Render/Railway), or is an
  **offline demo with periodically refreshed data** good enough for a class
  presentation?

## Project structure
```
smartpark/
├── data/
│   └── generate_synthetic_data.py
├── models/
│   └── cnn_model.py
├── api/
│   └── app.py
├── demo/
│   ├── campuspark_demo_template.html
│   └── campuspark_demo.html          <- open this to see the app
├── train.py
├── inference.py
├── improper_parking.py
├── validate.py
└── README.md                          <- you are here
```
