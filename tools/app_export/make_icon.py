"""
Draws the IDR app icon (a road through a tunnel, IDR orange for the part
driven without GPS) and writes every iOS / Android launcher size.

Usage (from repo root):
    tools/app_export/.venv/bin/python tools/app_export/make_icon.py
"""

import json
import os

from PIL import Image, ImageDraw

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APP = os.path.join(REPO, "app")
BG, TUNNEL, ROAD, ORANGE = (0x17, 0x18, 0x18), (0x2E, 0x2F, 0x2F), (0xF2, 0xF2, 0xF2), (0xF0, 0x61, 0x31)
S = 4096  # supersampled canvas


def bezier(p0, p1, p2, p3, n=400):
    pts = []
    for i in range(n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3
        pts.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return pts


def stroke(draw, pts, width, color):
    """Round-capped stroke drawn as densely spaced discs (no joint gaps)."""
    r = width / 2
    step = max(r / 6, 1.0)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        k = max(int(seg / step), 1)
        for i in range(k + 1):
            x, y = x0 + (x1 - x0) * i / k, y0 + (y1 - y0) * i / k
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def render():
    img = Image.new("RGB", (S, S), BG)
    d = ImageDraw.Draw(img)
    u = S / 1024
    # Same S-curve as the splash illustration, framed for the icon.
    a = bezier((150 * u, 720 * u), (360 * u, 720 * u), (380 * u, 512 * u), (512 * u, 512 * u))
    b = bezier((512 * u, 512 * u), (644 * u, 512 * u), (664 * u, 304 * u), (874 * u, 304 * u))
    path = a + b[1:]
    n = len(path)
    t0, t1 = int(n * 0.36), int(n * 0.66)
    stroke(d, path[t0:t1], 170 * u, TUNNEL)       # tunnel
    stroke(d, path, 56 * u, ROAD)                  # the road, GPS all the way
    stroke(d, path[t0:t1], 56 * u, ORANGE)         # IDR's estimate in the tunnel
    x, y = path[-1]
    r = 64 * u
    d.ellipse((x - r, y - r, x + r, y + r), fill=ROAD)
    return img.resize((1024, 1024), Image.LANCZOS)


def main():
    icon = render()
    icon.save(os.path.join(APP, "assets", "icon", "icon-1024.png"))

    # iOS: every image listed in the AppIcon set (no alpha channel allowed).
    ios = os.path.join(APP, "ios", "Runner", "Assets.xcassets", "AppIcon.appiconset")
    contents = json.load(open(os.path.join(ios, "Contents.json")))
    for im in contents["images"]:
        if "filename" not in im:
            continue
        pt = float(im["size"].split("x")[0])
        px = round(pt * float(im["scale"].rstrip("x")))
        icon.resize((px, px), Image.LANCZOS).save(os.path.join(ios, im["filename"]))

    # Android legacy launcher icons.
    for folder, px in {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}.items():
        out = os.path.join(APP, "android", "app", "src", "main", "res", f"mipmap-{folder}", "ic_launcher.png")
        icon.resize((px, px), Image.LANCZOS).save(out)
    print("icons written")


if __name__ == "__main__":
    main()
