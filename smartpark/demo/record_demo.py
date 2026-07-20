import os
from playwright.sync_api import sync_playwright

DEMO_PATH = os.path.abspath("demo/campuspark_demo_real.html")
VIDEO_DIR = "demo/video_raw"
os.makedirs(VIDEO_DIR, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    context = browser.new_context(
        viewport={"width": 480, "height": 900},
        record_video_dir=VIDEO_DIR,
        record_video_size={"width": 480, "height": 900},
    )
    page = context.new_page()
    page.goto("file://" + DEMO_PATH)
    page.wait_for_timeout(1800)  # let the viewer take in the initial state

    # Tap through a few individual stalls to show the detail card updating
    stall_selector = ".stall"
    stalls = page.query_selector_all(stall_selector)

    # tap an empty stall
    for s in stalls:
        cls = s.get_attribute("class")
        if "empty" in cls:
            s.click()
            break
    page.wait_for_timeout(1600)

    # tap an occupied (properly parked) stall
    for s in stalls:
        cls = s.get_attribute("class")
        if "occupied" in cls:
            s.click()
            break
    page.wait_for_timeout(1600)

    # tap another occupied stall for variety
    occupied_found = 0
    for s in stalls:
        cls = s.get_attribute("class")
        if "occupied" in cls:
            occupied_found += 1
            if occupied_found == 2:
                s.click()
                break
    page.wait_for_timeout(1600)

    # hit refresh
    page.click("#refreshBtn")
    page.wait_for_timeout(1400)

    # cycle to next lot
    page.click("#nextLotBtn")
    page.wait_for_timeout(2000)

    # tap a stall on the new lot
    stalls2 = page.query_selector_all(stall_selector)
    for s in stalls2:
        cls = s.get_attribute("class")
        if "occupied" in cls:
            s.click()
            break
    page.wait_for_timeout(1800)

    # cycle to next lot again
    page.click("#nextLotBtn")
    page.wait_for_timeout(2200)

    context.close()
    browser.close()

# find the produced video file
files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".webm")]
print("Recorded:", files)
