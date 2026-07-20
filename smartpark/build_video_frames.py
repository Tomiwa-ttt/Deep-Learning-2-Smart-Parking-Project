"""
build_video_frames.py

Builds every still frame for the "full demo" video: title card, one raw +
one annotated frame per real lot (with caption text burned in), and a
closing stats card. All frames are 480x900 to match the phone-app clip
already recorded, so they can be concatenated into one video.
"""
import json
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 480, 900
BG = (20, 22, 26)         # asphalt-950
PANEL = (36, 40, 47)      # asphalt-800
GREEN = (79, 174, 109)
AMBER = (224, 163, 56)
RED = (224, 82, 58)
MUTED = (139, 144, 156)
WHITE = (244, 243, 239)

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

OUT_DIR = "demo/video_frames"
os.makedirs(OUT_DIR, exist_ok=True)

with open("demo/video_assets/manifest.json") as f:
    manifest = json.load(f)


def font(path, size):
    return ImageFont.truetype(path, size)


def new_canvas():
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def center_text(draw, y, text, fnt, fill, max_width=W - 40):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) / 2, y), text, font=fnt, fill=fill)


def title_card():
    img, d = new_canvas()
    d.ellipse((W / 2 - 6, 300, W / 2 + 6, 312), fill=GREEN)
    center_text(d, 340, "CAMPUSPARK", font(FONT_BOLD, 40), WHITE)
    center_text(d, 395, "REAL-WORLD DEMONSTRATION", font(FONT_MONO, 15), MUTED)
    center_text(d, 460, "CNN trained on real PKLot photos", font(FONT_MONO, 14), AMBER)
    center_text(d, 485, "97.04% validation accuracy", font(FONT_MONO, 14), AMBER)
    img.save(f"{OUT_DIR}/00_title.png")


def section_card(text_lines, accent=AMBER):
    img, d = new_canvas()
    y = 380
    for i, line in enumerate(text_lines):
        fnt = font(FONT_BOLD, 26) if i == 0 else font(FONT_MONO, 15)
        col = WHITE if i == 0 else MUTED
        center_text(d, y, line, fnt, col)
        y += 40 if i == 0 else 26
    d.rectangle((W / 2 - 30, y + 10, W / 2 + 30, y + 13), fill=accent)
    return img


def photo_frame(photo_path, caption_top, caption_bottom_lines, border_color, fname):
    img, d = new_canvas()
    photo = Image.open(photo_path).convert("RGB")
    margin = 24
    target_w = W - margin * 2
    scale = target_w / photo.width
    target_h = int(photo.height * scale)
    photo = photo.resize((target_w, target_h))

    photo_y = 130
    d.rectangle(
        (margin - 4, photo_y - 4, margin + target_w + 4, photo_y + target_h + 4),
        outline=border_color, width=3
    )
    img.paste(photo, (margin, photo_y))

    center_text(d, 55, caption_top, font(FONT_BOLD, 22), WHITE)

    y = photo_y + target_h + 30
    for i, line in enumerate(caption_bottom_lines):
        fnt = font(FONT_MONO, 16) if i == 0 else font(FONT_MONO, 13)
        col = WHITE if i == 0 else MUTED
        center_text(d, y, line, fnt, col)
        y += 26 if i == 0 else 22

    img.save(f"{OUT_DIR}/{fname}")


# ---- Build frames ----
title_card()

for i, m in enumerate(manifest):
    lot_label = f"Lot: {m['lot_name']}"
    acc_pct = 100 * m["correct"] / m["total"]

    photo_frame(
        m["raw_path"],
        f"Real Photo {i+1}/{len(manifest)}",
        [lot_label, "Unprocessed camera input"],
        (60, 65, 74),
        f"{i+1:02d}a_raw.png",
    )

    photo_frame(
        m["annotated_path"],
        "CNN Detection Result",
        [
            f"{m['empty']} empty (green)  ·  {m['occupied']} occupied (orange)",
            f"{m['correct']}/{m['total']} spots correct ({acc_pct:.0f}%) vs. ground truth",
        ],
        AMBER,
        f"{i+1:02d}b_annotated.png",
    )

img = section_card([
    "Real-Data Results",
    "",
    "1,242 real PKLot photos",
    "70,684 labeled real parking spaces",
    "",
    "Final validation accuracy: 97.04%",
    "Spot-check on held-out photos: 93.9%",
])
img.save(f"{OUT_DIR}/90_stats.png")

img = section_card([
    "Now in the App",
    "",
    "Same model, same real detections,",
    "shown live in the CampusPark UI",
], accent=GREEN)
img.save(f"{OUT_DIR}/91_transition.png")

print("Frames built:", sorted(os.listdir(OUT_DIR)))
