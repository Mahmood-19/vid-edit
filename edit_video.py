"""Re-style a talking-head clip: new studio background, person cut-out with the
head popping out of a square frame, word-by-word Arabic captions and emoji
graphics that follow what is being said.

usage: python3 edit_video.py WORKDIR OUT_SILENT.mp4
WORKDIR holds src/NNNN.png (30fps frames), mat2/NNNN.png (person matte, see refine step) and
font/package/files/cairo-arabic-900-normal.woff.
"""
import glob
import math
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 720, 1280, 30
WORK, OUT = sys.argv[1], sys.argv[2]

# ---- layout ---------------------------------------------------------------
SCALE = 2.6                    # source -> canvas
FX0, FY0, FX1, FY1 = 90, 470, 630, 1010   # the square frame
SRC_ANCHOR = (233, 480)        # source point (head centre x, collar y)
OX = 360 - SRC_ANCHOR[0] * SCALE
OY = FY0 - SRC_ANCHOR[1] * SCALE

# ---- transcript (start, end, words) ----------------------------------------
SEGMENTS = [
    (0.54, 2.95, "السلام عليكم ورحمة الله وبركاته"),
    (3.64, 6.78, "الجميع اللي منتظرين وصول البضاعة"),
    (7.71, 9.30, "شحنتنا قربت توصل"),
    (9.70, 10.88, "خلكم منتظرين"),
    (11.49, 13.69, "إن شاء الله الجديدة حلوة وجاي في الطريق"),
    (14.77, 15.18, "انتظروا"),
]
WORDS = []  # (t0, t1, text, chunk_idx)
for ci, (a, b, txt) in enumerate(SEGMENTS):
    ws = txt.split()
    wt = [len(w) + 1.5 for w in ws]
    t = a
    for w, k in zip(ws, wt):
        d = (b - a) * k / sum(wt)
        WORDS.append((t, t + d, w, ci))
        t += d


def word_t(text, nth=0):
    return [w for w in WORDS if w[2] == text][nth][0]


# emoji graphics: (emoji, t_in, t_out, slot, motion)
L1, L2, R1, R2 = (105, 245), (105, 405), (615, 245), (615, 405)
EVENTS = [
    ("👋", 0.55, 2.95, L1, "wave"),
    ("🕊️", 1.55, 3.3, R1, "bob"),
    ("👥", word_t("الجميع"), 6.9, L1, "bob"),
    ("⏳", word_t("منتظرين", 0), 6.9, R1, "bob"),
    ("📦", word_t("البضاعة"), 7.4, L2, "bob"),
    ("🚚", 7.71, 9.5, None, "drive_l"),
    ("📍", word_t("قربت"), 9.5, R2, "bob"),
    ("⏰", 9.7, 11.1, L1, "wave"),
    ("🙏", 10.1, 11.1, R1, "bob"),
    ("🤲", word_t("إن"), 12.4, L2, "bob"),
    ("✨", word_t("الجديدة"), 13.9, L1, "bob"),
    ("😍", word_t("حلوة"), 13.9, R1, "bob"),
    ("🛣️", word_t("في"), 13.9, R2, "bob"),
    ("🚚", word_t("وجاي"), 14.6, None, "drive_r"),
    ("⏳", 14.77, 15.5, L1, "wave"),
    ("‼️", 14.85, 15.5, R1, "bob"),
]


def ease_out_back(x):
    c1, c3 = 1.70158, 2.70158
    x = min(max(x, 0), 1)
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


# ---- assets ----------------------------------------------------------------
emoji_font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109)
_emoji_cache = {}


def emoji_img(ch):
    if ch not in _emoji_cache:
        im = Image.new("RGBA", (136, 128), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((0, 0), ch, font=emoji_font, embedded_color=True)
        _emoji_cache[ch] = np.asarray(im.crop(im.getbbox()))
    return _emoji_cache[ch]


FONT_PATH = f"{WORK}/font/package/files/cairo-arabic-900-normal.woff"
CAP_SIZE = 70
cap_font = ImageFont.truetype(FONT_PATH, CAP_SIZE)


def blend(dst, rgba, cx, cy, size, alpha=1.0, angle=0.0):
    """alpha-composite an RGBA uint8 sprite (scaled to `size` px wide) onto dst."""
    h0, w0 = rgba.shape[:2]
    s = size / max(w0, h0)
    nw, nh = max(2, int(w0 * s)), max(2, int(h0 * s))
    sp = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    if angle:
        M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1)
        sp = cv2.warpAffine(sp, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + nw, W), min(y0 + nh, H)
    if xa >= xb or ya >= yb:
        return
    sp = sp[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
    a = sp[..., 3:4] / 255 * alpha
    dst[ya:yb, xa:xb] = dst[ya:yb, xa:xb] * (1 - a) + sp[..., :3] * a


# ---- studio background ------------------------------------------------------
def make_background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = yy / H
    top, mid, bot = np.array([62, 40, 140]), np.array([34, 24, 92]), np.array([18, 14, 52])
    bg = np.where(t[..., None] < 0.6,
                  top + (mid - top) * (t[..., None] / 0.6),
                  mid + (bot - mid) * ((t[..., None] - 0.6) / 0.4))
    # vertical acoustic slats on the back wall
    slat = ((xx // 36) % 2).astype(np.float32)
    shade = 1 - 0.18 * slat + 0.06 * np.sin(xx / 36 * math.pi)
    wall = (yy < 900).astype(np.float32)[..., None]
    bg = bg * (1 - wall) + bg * shade[..., None] * wall
    # big light behind the head so hair/beard separate from the wall
    r = np.hypot(xx - 360, yy - 250)
    glow = np.clip(1 - r / 430, 0, 1) ** 2
    bg += glow[..., None] * np.array([150, 95, 215])
    # neon ring + bars
    img = np.clip(bg, 0, 255).astype(np.uint8)
    neon = np.zeros((H, W, 3), np.uint8)
    cv2.circle(neon, (360, 262), 215, (255, 90, 200), 7, cv2.LINE_AA)
    cv2.circle(neon, (360, 262), 245, (80, 220, 255), 4, cv2.LINE_AA)
    cv2.line(neon, (30, 150), (30, 560), (80, 220, 255), 6, cv2.LINE_AA)
    cv2.line(neon, (690, 150), (690, 560), (255, 90, 200), 6, cv2.LINE_AA)
    blur = cv2.GaussianBlur(neon, (0, 0), 14).astype(np.float32)
    out = img.astype(np.float32) + blur * 1.1 + neon.astype(np.float32) * 0.9
    # floor
    floor = (yy > 1010)[..., None]
    out = np.where(floor, out * 0.55 + np.array([10, 6, 30]), out)
    # bokeh
    rng = np.random.default_rng(7)
    bok = np.zeros((H, W, 3), np.float32)
    for _ in range(34):
        c = rng.choice([0, 1, 2])
        col = [(255, 150, 230), (140, 230, 255), (255, 220, 150)][c]
        cv2.circle(bok, (int(rng.uniform(0, W)), int(rng.uniform(0, 1010))), int(rng.uniform(10, 38)), col, -1, cv2.LINE_AA)
    out += cv2.GaussianBlur(bok, (0, 0), 5) * 0.28
    return np.clip(out, 0, 255).astype(np.float32)


BG = make_background()


def make_frame_panel():
    yy, xx = np.mgrid[FY0:FY1, FX0:FX1].astype(np.float32)
    k = ((xx - FX0) + (yy - FY0)) / ((FX1 - FX0) + (FY1 - FY0))
    c0, c1 = np.array([255, 94, 140]), np.array([255, 178, 70])
    p = c0 + (c1 - c0) * k[..., None]
    stripes = (((xx + yy) // 28) % 2)[..., None] * 14
    return np.clip(p + stripes, 0, 255)


PANEL = make_frame_panel()
RR = 38  # corner radius


def rounded_mask(x0, y0, x1, y1, r, pad=0):
    m = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for cx, cy in [(x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)]:
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m.astype(np.float32) / 255


FRAME_IN = rounded_mask(FX0, FY0, FX1, FY1, RR)
FRAME_OUT = rounded_mask(FX0, FY0, FX1, FY1, RR + 14, pad=14)
BORDER = np.clip(FRAME_OUT - FRAME_IN, 0, 1)
# region where the person may be drawn: inside the frame OR above its top edge
ALLOW = np.maximum(FRAME_IN, (np.arange(H)[:, None] < FY0 + 2).astype(np.float32) * np.ones((1, W), np.float32))


def base_layer(t):
    out = BG.copy()
    # soft pulsing glow on the frame
    pulse = 0.5 + 0.5 * math.sin(t * 4)
    halo = cv2.GaussianBlur(FRAME_OUT, (0, 0), 22)[..., None] * (60 + 40 * pulse) * np.array([1.0, 0.4, 0.8])
    out += halo
    # frame interior
    panel = np.zeros((H, W, 3), np.float32)
    panel[FY0:FY1, FX0:FX1] = PANEL
    out = out * (1 - FRAME_IN[..., None]) + panel * FRAME_IN[..., None]
    # white border (drawn behind the person so the head overlaps it)
    out = out * (1 - BORDER[..., None]) + np.array([255, 255, 255]) * BORDER[..., None]
    return out


# ---- person ------------------------------------------------------------------
def person(i):
    src = cv2.cvtColor(cv2.imread(f"{WORK}/src/{i:04d}.png"), cv2.COLOR_BGR2RGB)
    m = cv2.imread(f"{WORK}/mat2/{i:04d}.png", 0).astype(np.float32) / 255
    m = np.clip((m - 0.4) / 0.4, 0, 1)  # tighten the soft matte edge
    M = np.float32([[SCALE, 0, OX], [0, SCALE, OY]])
    rgb = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    al = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    # kill edge halo of the old background: erode colour bleed a touch
    al = np.clip(al, 0, 1)
    return rgb.astype(np.float32), al


# ---- captions ----------------------------------------------------------------
_word_cache = {}


def word_sprite(text, color):
    key = (text, color)
    if key not in _word_cache:
        pad = 14
        l, t, r, b = cap_font.getbbox(text, direction="rtl", language="ar", stroke_width=8)
        im = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((pad - l, pad - t), text, font=cap_font, fill=color, direction="rtl", language="ar",
                                stroke_width=8, stroke_fill=(30, 12, 70))
        _word_cache[key] = np.asarray(im)
    return _word_cache[key]


def layout_chunk(ci):
    ws = [w for w in WORDS if w[3] == ci]
    sizes = [word_sprite(w[2], (255, 255, 255, 255)).shape[1] - 28 for w in ws]
    gap, maxw = 26, 640
    lines, cur, curw = [], [], 0
    for k, sz in enumerate(sizes):
        add = sz + (gap if cur else 0)
        if cur and curw + add > maxw:
            lines.append(cur); cur, curw = [], 0; add = sz
        cur.append(k); curw += add
    lines.append(cur)
    pos = {}
    lh = 92
    y0 = 1150 - (len(lines) - 1) * lh / 2
    for li, ln in enumerate(lines):
        tot = sum(sizes[k] for k in ln) + gap * (len(ln) - 1)
        x = 360 + tot / 2
        for k in ln:  # RTL: first word on the right
            pos[ws[k][2], ws[k][0]] = (x - sizes[k] / 2, y0 + li * lh)
            x -= sizes[k] + gap
    return ws, pos


LAYOUT = {ci: layout_chunk(ci) for ci in range(len(SEGMENTS))}
ACCENT = [(255, 214, 10, 255), (90, 235, 255, 255), (255, 120, 190, 255)]


def draw_caption(out, t):
    for ci, (a, b, _) in enumerate(SEGMENTS):
        end = b + 0.25 if ci < len(SEGMENTS) - 1 else 99
        if not (a - 0.05 <= t < end):
            continue
        ws, pos = LAYOUT[ci]
        cur = None
        for k, w in enumerate(ws):
            if w[0] - 0.04 <= t:
                cur = k
        for k, w in enumerate(ws):
            if w[0] - 0.04 > t:
                continue
            cx, cy = pos[w[2], w[0]]
            if k == cur:
                age = t - (w[0] - 0.04)
                s = 0.7 + 0.5 * ease_out_back(age / 0.22) - 0.2 * min(age / 0.22, 1) if age < 0.22 else 1.0
                s = max(s, 0.6)
                sp = word_sprite(w[2], ACCENT[ci % 3])
                size = sp.shape[1] * s
            else:
                sp = word_sprite(w[2], (255, 255, 255, 255))
                size = sp.shape[1]
            blend(out, sp, cx, cy, size)
        return


# ---- graphics ------------------------------------------------------------------
def draw_graphics(out, t):
    for ch, a, b, slot, motion in EVENTS:
        if not (a <= t <= b + 0.3):
            continue
        age = t - a
        pop = ease_out_back(age / 0.35)
        fade = 1.0 if t <= b else max(0, 1 - (t - b) / 0.3)
        size = 150 * pop
        if size < 4:
            continue
        sp = emoji_img(ch)
        if motion == "drive_l":
            f = (t - a) / (b - a)
            cx, cy = 760 - f * 800, 330 + 6 * math.sin(t * 25)
            blend(out, sp, cx, cy, 190, fade)
            continue
        if motion == "drive_r":
            f = min((t - a) / max(b - a, 1e-3), 1)
            cx, cy = -40 + f * 800, 335 + 6 * math.sin(t * 25)
            blend(out, sp[:, ::-1].copy(), cx, cy, 190, fade)
            continue
        cx, cy = slot
        ang = 0
        if motion == "bob":
            cy += 9 * math.sin(t * 5 + a * 3)
        elif motion == "wave":
            ang = 18 * math.sin(t * 9)
        # burst ring on entry
        if age < 0.45:
            r = int(40 + age * 330)
            ring = np.zeros((H, W), np.float32)
            cv2.circle(ring, (int(cx), int(cy)), r, 1.0, 5, cv2.LINE_AA)
            al = ring[..., None] * (1 - age / 0.45) * 0.9
            out[:] = out * (1 - al) + np.array([255, 255, 255]) * al
        # soft backing disc
        disc = np.zeros((H, W), np.float32)
        cv2.circle(disc, (int(cx), int(cy)), int(size * 0.62), 1.0, -1, cv2.LINE_AA)
        disc = cv2.GaussianBlur(disc, (0, 0), 3)[..., None] * 0.35 * fade
        out[:] = out * (1 - disc) + np.array([20, 10, 60]) * disc
        blend(out, sp, cx, cy, size, fade, ang)


# ---- main ----------------------------------------------------------------------
def main():
    n = len(glob.glob(f"{WORK}/src/*.png"))
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(1, n + 1):
        t = (i - 1) / FPS
        out = base_layer(t)
        rgb, al = person(i)
        al = al * ALLOW
        # drop shadow of the person for depth
        sh = cv2.GaussianBlur(al, (0, 0), 14)
        sh = np.roll(sh, 14, axis=0)[..., None] * 0.45
        out = out * (1 - sh)
        out = out * (1 - al[..., None]) + rgb * al[..., None]
        draw_graphics(out, t)
        draw_caption(out, t)
        vw.write(cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()


if __name__ == "__main__":
    main()
