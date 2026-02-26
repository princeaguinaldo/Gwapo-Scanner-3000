"""
╔══════════════════════════════════════════════════════════╗
║           GWAPO SCANNER 3000  —  Python Edition          ║
║     Uses your webcam + OpenCV for face detection         ║
╚══════════════════════════════════════════════════════════╝

REQUIREMENTS:
    pip install opencv-python numpy

HOW TO RUN:
    python gwapo_scanner.py

CONTROLS:
    SPACE  — Scan the current face
    Q      — Quit
"""

import cv2
import numpy as np
import random
import time
import os

# ── Colors (BGR) ─────────────────────────────────────────
CYAN       = (212, 245,   0)
PINK       = (107,  45, 255)
YELLOW     = ( 53, 225, 255)
WHITE      = (255, 255, 255)
BLACK      = (  0,   0,   0)
DIM        = (100, 100, 120)
BG_COLOR   = ( 15,  10,  10)

# ── Haar Cascade face detector (ships with OpenCV) ────────
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def draw_corner_brackets(frame, x, y, w, h, color, thickness=2, size=22):
    """Draw stylish corner brackets around a face box."""
    pts = [
        [(x, y),             (x + size, y),       (x, y + size)],
        [(x+w, y),           (x+w - size, y),     (x+w, y + size)],
        [(x, y+h),           (x + size, y+h),     (x, y+h - size)],
        [(x+w, y+h),         (x+w - size, y+h),   (x+w, y+h - size)],
    ]
    for corner in pts:
        cv2.line(frame, corner[0], corner[1], color, thickness)
        cv2.line(frame, corner[0], corner[2], color, thickness)


def draw_dashed_rect(frame, x, y, w, h, color, dash=12, gap=6, thickness=1):
    """Draw a dashed rectangle."""
    perimeter = 2 * (w + h)
    step = dash + gap
    dist = 0
    # top, right, bottom, left edges
    edges = [
        ((x, y), (x + w, y), True),
        ((x + w, y), (x + w, y + h), False),
        ((x + w, y + h), (x, y + h), True),
        ((x, y + h), (x, y), False),
    ]
    for (x1, y1), (x2, y2), horizontal in edges:
        length = abs(x2 - x1) if horizontal else abs(y2 - y1)
        pos = 0
        while pos < length:
            end = min(pos + dash, length)
            if horizontal:
                dx = 1 if x2 > x1 else -1
                cv2.line(frame, (x1 + dx*pos, y1), (x1 + dx*end, y1), color, thickness)
            else:
                dy = 1 if y2 > y1 else -1
                cv2.line(frame, (x1, y1 + dy*pos), (x1, y1 + dy*end), color, thickness)
            pos += step


def put_text_center(frame, text, cy, font_scale=0.6, color=WHITE, thickness=1):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (frame.shape[1] - tw) // 2
    cv2.putText(frame, text, (x, cy), font, font_scale, color, thickness, cv2.LINE_AA)


def put_text(frame, text, x, y, font_scale=0.5, color=WHITE, thickness=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_progress_bar(frame, x, y, w, h, progress, color_start, color_end):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 50), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), DIM, 1)
    fill = int(w * progress)
    if fill > 0:
        # Gradient-ish by blending colors
        bar_color = tuple(int(color_start[i] + (color_end[i] - color_start[i]) * progress)
                          for i in range(3))
        cv2.rectangle(frame, (x, y), (x + fill, y + h), bar_color, -1)


def run_scanner():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam. Check your camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WIN = "GWAPO SCANNER 3000  |  SPACE = Scan  |  Q = Quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 700, 620)

    # State machine
    STATE_IDLE    = "idle"
    STATE_SCAN    = "scanning"
    STATE_RESULT  = "result"

    state          = STATE_IDLE
    scan_start     = 0
    scan_duration  = 2.2   # seconds
    result         = None
    result_time    = 0
    scan_line_y    = 0
    frame_count    = 0
    flash_frames   = 0

    PHASES = [
        "DETECTING FACIAL LANDMARKS...",
        "MEASURING GOLDEN RATIO...",
        "CALCULATING SYMMETRY INDEX...",
        "CROSS-REFERENCING GWAPO DB...",
        "GENERATING VERDICT...",
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        frame_count += 1

        # ── Background darkening panel ────────────────────
        overlay_bg = frame.copy()
        cv2.rectangle(overlay_bg, (0, fh - 160), (fw, fh), (8, 8, 14), -1)
        frame = cv2.addWeighted(frame, 0.85, overlay_bg, 0.15, 0)

        # ── Face detection ────────────────────────────────
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces      = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        face_found = len(faces) > 0

        # Pick largest face
        main_face = None
        if face_found:
            main_face = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw2, fh2 = main_face

            if state == STATE_IDLE:
                # Dashed box + brackets
                draw_dashed_rect(frame, fx, fy, fw2, fh2, CYAN, thickness=1)
                draw_corner_brackets(frame, fx, fy, fw2, fh2, CYAN, thickness=2)

                # Crosshair
                cx, cy_face = fx + fw2 // 2, fy + fh2 // 2
                cv2.line(frame, (cx - 12, cy_face), (cx + 12, cy_face), CYAN, 1)
                cv2.line(frame, (cx, cy_face - 12), (cx, cy_face + 12), CYAN, 1)

            elif state == STATE_SCAN:
                draw_corner_brackets(frame, fx, fy, fw2, fh2, YELLOW, thickness=2)

                # Animated scan line within face box
                scan_line_y = (scan_line_y + 4) % fh2
                sy = fy + scan_line_y
                cv2.line(frame, (fx, sy), (fx + fw2, sy), YELLOW, 1)

                # Glow overlay
                glow = frame.copy()
                cv2.rectangle(glow, (fx, fy), (fx + fw2, fy + fh2), YELLOW, -1)
                frame = cv2.addWeighted(frame, 0.95, glow, 0.05, 0)

            elif state == STATE_RESULT and result is not None:
                col = CYAN if result["gwapo"] else PINK
                draw_corner_brackets(frame, fx, fy, fw2, fh2, col, thickness=2)
                cv2.rectangle(frame, (fx, fy), (fx + fw2, fy + fh2), col, 1)

        # ── Flash effect ──────────────────────────────────
        if flash_frames > 0:
            white = np.ones_like(frame) * 255
            alpha = flash_frames / 6.0
            frame = cv2.addWeighted(frame, 1 - alpha, white, alpha, 0)
            flash_frames -= 1

        # ── Bottom HUD panel ──────────────────────────────
        panel_y = fh - 155
        panel = frame[panel_y:, :].copy()
        dark  = np.zeros_like(panel)
        frame[panel_y:] = cv2.addWeighted(panel, 0.3, dark, 0.7, 0)

        # Divider line
        cv2.line(frame, (0, panel_y), (fw, panel_y), CYAN, 1)

        # Title
        put_text(frame, "GWAPO SCANNER 3000", 10, panel_y + 25,
                 font_scale=0.65, color=CYAN, thickness=1)

        # Status dot
        dot_color = CYAN if face_found else DIM
        if state == STATE_SCAN:
            dot_color = YELLOW if (frame_count % 10 < 5) else (100, 150, 50)
        cv2.circle(frame, (fw - 20, panel_y + 15), 6, dot_color, -1)

        # ── IDLE state HUD ────────────────────────────────
        if state == STATE_IDLE:
            if face_found:
                status = "FACE DETECTED  —  PRESS SPACE TO SCAN"
                put_text(frame, status, 10, panel_y + 50, 0.45, CYAN)
            else:
                status = "NO FACE DETECTED  —  POSITION YOURSELF IN FRAME"
                put_text(frame, status, 10, panel_y + 50, 0.42, DIM)

            put_text(frame, "AWAITING SCAN...", 10, panel_y + 75, 0.4, DIM)

            # Empty progress bar
            draw_progress_bar(frame, 10, panel_y + 90, fw - 20, 10, 0.0, CYAN, YELLOW)

            put_text(frame, "[ SPACE ] SCAN    [ Q ] QUIT",
                     10, panel_y + 140, 0.38, DIM)

        # ── SCANNING state HUD ────────────────────────────
        elif state == STATE_SCAN:
            elapsed  = time.time() - scan_start
            progress = min(elapsed / scan_duration, 1.0)
            phase_i  = min(int(progress * len(PHASES)), len(PHASES) - 1)

            put_text(frame, PHASES[phase_i], 10, panel_y + 50, 0.42, YELLOW)
            put_text(frame, f"ANALYZING...  {int(progress*100)}%", 10, panel_y + 75, 0.42, WHITE)

            draw_progress_bar(frame, 10, panel_y + 90, fw - 20, 12,
                              progress, YELLOW, CYAN)

            if progress >= 1.0:
                state  = STATE_RESULT
                result = generate_result()
                result_time = time.time()
                flash_frames = 6

        # ── RESULT state HUD ─────────────────────────────
        elif state == STATE_RESULT and result is not None:
            col     = CYAN if result["gwapo"] else PINK
            verdict = result["label"]
            score   = result["score"]

            # Big verdict text
            (tw, _), _ = cv2.getTextSize(verdict, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
            vx = (fw - tw) // 2
            cv2.putText(frame, verdict, (vx, panel_y + 55),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, col, 2, cv2.LINE_AA)

            # Score bar
            draw_progress_bar(frame, 10, panel_y + 70, fw - 20, 10,
                              score / 100, col, WHITE)

            score_txt = f"GWAPO SCORE: {score}/100   SYM:{result['sym']}%  FEAT:{result['feat']}%  EXPR:{result['expr']}"
            put_text(frame, score_txt, 10, panel_y + 100, 0.35, WHITE)

            # Time remaining to next scan
            elapsed = time.time() - result_time
            put_text(frame, "[ SPACE ] SCAN AGAIN    [ Q ] QUIT",
                     10, panel_y + 140, 0.38, DIM)

            # Auto-reset after 6 seconds
            if elapsed > 6:
                state  = STATE_IDLE
                result = None

        # ── Scanlines overlay ─────────────────────────────
        for yy in range(0, fh, 4):
            cv2.line(frame, (0, yy), (fw, yy), (0, 0, 0), 1)
        scanline_mask = np.zeros_like(frame)
        for yy in range(0, fh, 4):
            scanline_mask[yy] = [10, 10, 10]
        frame = cv2.addWeighted(frame, 1.0, scanline_mask, 0.15, 0)

        cv2.imshow(WIN, frame)

        # ── Key handling ──────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        if key == ord(' '):
            if state == STATE_IDLE:
                state      = STATE_SCAN
                scan_start = time.time()
                scan_line_y = 0
            elif state == STATE_RESULT:
                state  = STATE_SCAN
                result = None
                scan_start = time.time()
                scan_line_y = 0

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ GWAPO SCANNER 3000 closed. Salamat!")


def generate_result():
    """Generate a randomized (entertainment-only) result."""
    is_gwapo = random.random() > 0.3  # slight bias toward gwapo — be kind!

    gwapo_labels   = ["CERTIFIED GWAPO!", "SOBRANG GWAPO!", "GWAPO LEVEL: MAX", "GWAPO DETECTED"]
    not_gwapo_labels = ["NOT GWAPO... YET", "GWAPO: PENDING", "NEEDS IMPROVEMENT", "ALMOST GWAPO"]

    label = random.choice(gwapo_labels if is_gwapo else not_gwapo_labels)
    score = random.randint(80, 99) if is_gwapo else random.randint(35, 65)
    sym   = round(random.uniform(70, 95) if is_gwapo else random.uniform(40, 65), 1)
    feat  = round(random.uniform(75, 96) if is_gwapo else random.uniform(38, 62), 1)
    expr  = random.choice(["NEUTRAL", "HAPPY", "SERIOUS", "MYSTERIOUS"])

    return {
        "gwapo": is_gwapo,
        "label": label,
        "score": score,
        "sym":   sym,
        "feat":  feat,
        "expr":  expr,
    }


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║           GWAPO SCANNER 3000  —  Python Edition          ║
╠══════════════════════════════════════════════════════════╣
║  CONTROLS:  SPACE = Scan   |   Q = Quit                  ║
║  NOTE: For entertainment purposes only 🇵🇭               ║
╚══════════════════════════════════════════════════════════╝
    """)

    # Check dependencies
    try:
        import cv2
        import numpy
    except ImportError:
        print("❌ Missing dependencies! Run this first:\n")
        print("   pip install opencv-python numpy\n")
        exit(1)

    run_scanner()
