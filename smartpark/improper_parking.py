"""
improper_parking.py

Geometric "properly parked" vs "improperly parked" check.

This is the Option A approach discussed in planning: rather than training a
second model, we reuse the car's bounding box (found by whatever detector
localizes the car within/near a spot) and compare it against the spot's own
marked boundary using IoU (Intersection over Union) and overflow checks.

No additional labeled dataset is required, this is pure geometry on top of
the outputs you already have from the occupancy detector.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParkingVerdict:
    spot_id: int
    occupied: bool
    properly_parked: bool | None  # None when spot is empty (not applicable)
    iou: float | None
    overflow_ratio: float | None  # fraction of car's area outside the spot
    reason: str


def _intersection_area(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _area(box):
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def evaluate_spot(spot_id, spot_bbox, car_bbox, occupied,
                   iou_threshold=0.55, overflow_threshold=0.30,
                   neighbor_bboxes=None):
    """
    spot_bbox: [x0,y0,x1,y1] of the painted parking spot boundary
    car_bbox: [x0,y0,x1,y1] of the detected car (None if spot is empty)
    occupied: bool, from the CNN classifier
    iou_threshold: below this, the car isn't well-aligned with its own spot
    overflow_threshold: above this fraction of car area outside the spot -> flag
    neighbor_bboxes: optional list of adjacent spot boxes, to report *which*
                      spot got encroached on
    """
    if not occupied or car_bbox is None:
        return ParkingVerdict(spot_id, False, None, None, None, "Spot is empty")

    inter = _intersection_area(spot_bbox, car_bbox)
    car_area = _area(car_bbox)
    spot_area = _area(spot_bbox)
    union = spot_area + car_area - inter
    iou = inter / union if union > 0 else 0.0

    overflow = (car_area - inter) / car_area if car_area > 0 else 0.0

    properly = (iou >= iou_threshold) and (overflow <= overflow_threshold)

    reason = "Within spot boundary" if properly else \
        f"Car extends outside its spot (IoU={iou:.2f}, overflow={overflow*100:.0f}%)"

    # Optionally identify which neighboring spot is being encroached on
    if not properly and neighbor_bboxes:
        for nid, nbox in neighbor_bboxes:
            n_inter = _intersection_area(nbox, car_bbox)
            if n_inter > 0.05 * car_area:
                reason += f" — encroaching into Spot {nid}"
                break

    return ParkingVerdict(spot_id, True, properly, round(iou, 3), round(overflow, 3), reason)


def evaluate_lot(manifest_spots, spot_index_by_id=None):
    """Run evaluate_spot across every spot in a lot, using each spot's own
    marked neighbors (adjacent by index) for the encroachment message."""
    results = []
    for i, s in enumerate(manifest_spots):
        neighbors = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(manifest_spots):
                neighbors.append((manifest_spots[j]["id"], manifest_spots[j]["bbox"]))
        verdict = evaluate_spot(
            spot_id=s["id"],
            spot_bbox=s["bbox"],
            car_bbox=s.get("car_bbox"),
            occupied=s["occupied"],
            neighbor_bboxes=neighbors,
        )
        results.append(verdict)
    return results


if __name__ == "__main__":
    # quick smoke test with a well-parked and a badly-parked example
    spot = [0, 0, 80, 140]
    good_car = [10, 10, 70, 130]
    bad_car = [50, 10, 130, 130]  # shifted right, spills into neighbor lane

    print(evaluate_spot(1, spot, good_car, occupied=True))
    print(evaluate_spot(2, spot, bad_car, occupied=True))
    print(evaluate_spot(3, spot, None, occupied=False))
