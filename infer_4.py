"""
Quarkball Live QC (Basler + PyTorch MobileNetV2) - 3-class robust live inference

Assumes classifier trained on 3 classes: good, bad, no_product (or similar).

Behavior:
- Runs classifier on ROI every frame.
- Uses p(no_product) with hysteresis + streaks to detect:
    IDLE (no product) -> TRACKING (product present) -> IDLE (product gone)
- While TRACKING, collects a burst of ROI frames + probs + blur metric.
- On departure, selects best frame and finalizes ONE count (good/bad/uncertain).
- UI:
    - Big label shows GOOD/BAD briefly after a product is finalized, then returns to NO PRODUCT.
    - Small line always shows Last finalized decision.

Extra:
- On-screen STOP button (touchscreen/mouse click) so you don't need a keyboard.
"""

import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from pypylon import pylon
import argparse
import json


# -----------------------------
# Config (tune these)
# -----------------------------
MODEL_PATH = Path("models/quarkball_mobilenet_v2_best.pth")
WINDOW_NAME = "Quarkball Live QC"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Whole-frame thick border colors (BGR)
BORDER_YELLOW = (0, 255, 255)
BORDER_GREEN  = (0, 255, 0)
BORDER_RED    = (0, 0, 255)
BORDER_THICKNESS = 18

# ROI where the quarkball passes (fractions of frame W/H)
ROI_REL = (0.20, 0.20, 0.60, 0.60)  # (x,y,w,h)

# --- Presence detection from classifier (hysteresis)
ENTER_NO_THR = 0.35
EXIT_NO_THR  = 0.75
ENTER_FRAMES = 3
EXIT_FRAMES  = 6

# --- Burst capture
MAX_BURST_FRAMES = 25
MIN_BURST_FRAMES = 5

# --- Final decision thresholds
FINAL_CONF_THR = 0.60

# --- UI behavior
LAST_DECISION_HOLD_SEC = 0.8  # show GOOD/BAD big for this long after item finalizes

# Optional: save chosen frame per product for boss/demo evidence
SAVE_BEST_FRAMES = True
SAVE_DIR = Path("captures_best")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Preprocess size (match training)
RESIZE_TO = (256, 256)  # then center crop 224


# -----------------------------
# STOP BUTTON
# -----------------------------
STOP_REQUESTED = False
STOP_RECT = (0, 0, 0, 0)  # (x1,y1,x2,y2)


def _mouse_cb(event, x, y, flags, param):
    """Touchscreen tap/mouse click handler for STOP button."""
    global STOP_REQUESTED, STOP_RECT
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = STOP_RECT
        if x1 <= x <= x2 and y1 <= y <= y2:
            STOP_REQUESTED = True


def draw_stop_button(frame):
    """
    Draw a red STOP button at top-right and update STOP_RECT.
    MODIFIED: bigger button + bigger text.
    """
    global STOP_RECT
    h, w = frame.shape[:2]

    # Bigger, more touch-friendly
    bw, bh = 300, 110
    m = 20

    x2 = w - m
    x1 = x2 - bw
    y1 = m
    y2 = y1 + bh
    STOP_RECT = (x1, y1, x2, y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    text = "STOP"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)
    tx = x1 + (bw - tw) // 2
    ty = y1 + (bh + th) // 2

    cv2.putText(
        frame,
        text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        (255, 255, 255),
        5,
        cv2.LINE_AA,
    )


# -----------------------------
# Model utilities
# -----------------------------
def build_mobilenet_v2(num_classes: int) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    classes = ckpt.get("classes", None)
    if not classes:
        raise ValueError("Checkpoint missing 'classes'. Expected {'model_state_dict':..., 'classes':...}")

    model = build_mobilenet_v2(len(classes))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(DEVICE).eval()
    return model, classes


def make_preprocess():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(RESIZE_TO),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def find_class_index(classes, want: str):
    want = want.lower().replace(" ", "").replace("-", "").replace("_", "")
    norm = []
    for c in classes:
        cc = str(c).lower().replace(" ", "").replace("-", "").replace("_", "")
        norm.append(cc)

    if want in norm:
        return norm.index(want)

    for i, cc in enumerate(norm):
        if want in cc:
            return i

    raise ValueError(f"Could not find class '{want}' in classes: {classes}")


@torch.no_grad()
def predict_probs(model, preprocess, roi_bgr):
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    x = preprocess(rgb).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    return probs, top_idx, top_conf


# -----------------------------
# Basler camera
# -----------------------------
def open_basler_latest_only():
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if len(devices) == 0:
        raise RuntimeError("No Basler camera found. Check connection/drivers.")

    camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    camera.Open()

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return camera, converter


def grab_frame(camera, converter, timeout_ms=500):
    grab = camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
    if not grab or not grab.GrabSucceeded():
        if grab:
            grab.Release()
        return None
    img = converter.Convert(grab)
    frame = img.GetArray()
    grab.Release()
    return frame


# -----------------------------
# Drawing / helpers
# -----------------------------
def roi_from_rel(frame_shape, roi_rel):
    h, w = frame_shape[:2]
    rx, ry, rw, rh = roi_rel
    x = int(rx * w)
    y = int(ry * h)
    ww = int(rw * w)
    hh = int(rh * h)
    return x, y, ww, hh


def draw_full_border(img, color, thickness):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thickness)


def draw_panel(img, lines, origin=(20, 50), line_h=44, pad=18):
    """
    MODIFIED: bigger default origin, line height, and padding so text is larger and cleaner.
    """
    x, y = origin
    max_w = 0
    total_h = line_h * len(lines)

    for (text, scale, thick) in lines:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        max_w = max(max_w, tw)

    x2 = x + max_w + pad * 2
    y2 = y + total_h + pad
    cv2.rectangle(img, (x - pad, y - pad), (x2, y2), (0, 0, 0), -1)

    for i, (text, scale, thick) in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thick, cv2.LINE_AA)


def blur_metric(roi_bgr):
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def score_frame(p_no, p_good, p_bad, blur_var):
    prodness = 1.0 - p_no
    class_conf = max(p_good, p_bad)
    blur_score = clamp01(blur_var / 200.0)
    return prodness * class_conf * blur_score

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result-json", default="", help="Write final counts to this JSON file on exit")
    return p.parse_args()



# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    result_json_path = args.result_json
    global STOP_REQUESTED
    model, classes = load_model(MODEL_PATH)
    preprocess = make_preprocess()

    print("Loaded classes:", classes)

    good_idx = find_class_index(classes, "good")
    bad_idx  = find_class_index(classes, "bad")
    no_idx   = find_class_index(classes, "no_product")

    camera, converter = open_basler_latest_only()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Force fullscreen
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # enable clicking/tapping STOP
    cv2.setMouseCallback(WINDOW_NAME, _mouse_cb)

    # Counters
    seen_products = 0
    good_count = 0
    bad_count = 0
    uncertain_count = 0

    # State machine
    state = "IDLE"  # IDLE or TRACKING
    enter_streak = 0
    exit_streak = 0

    # Burst storage while TRACKING
    burst = []

    # Last finalized decision (for logging)
    last_decision = "NONE"
    last_conf = 0.0

    # Display decision (big label) + timer
    display_decision = "NO PRODUCT"
    display_conf = 0.0
    last_final_time = 0.0

    # FPS meter
    t0 = time.time()
    frames = 0
    fps = 0.0

    def finalize_one_item():
        nonlocal seen_products, good_count, bad_count, uncertain_count, burst

        if len(burst) == 0:
            return None

        best = max(burst, key=lambda d: d["score"])

        p_good = float(best["probs"][good_idx])
        p_bad  = float(best["probs"][bad_idx])
        p_no   = float(best["probs"][no_idx])

        seen_products += 1

        decision = "UNCERTAIN"
        if max(p_good, p_bad) >= FINAL_CONF_THR:
            if p_good >= p_bad:
                decision = "GOOD"
                good_count += 1
            else:
                decision = "BAD"
                bad_count += 1
        else:
            uncertain_count += 1

        # save proof image
        if SAVE_BEST_FRAMES:
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = (
                f"{ts}_{seen_products:05d}_{decision}"
                f"_pg{p_good:.2f}_pb{p_bad:.2f}_pno{p_no:.2f}_blur{best['blur']:.0f}.jpg"
            )
            cv2.imwrite(str(SAVE_DIR / fname), best["roi"])

        burst = []
        return decision, max(p_good, p_bad), p_good, p_bad, p_no

    try:
        while True:
            frame = grab_frame(camera, converter)
            if frame is None:
                key = cv2.waitKey(1) & 0xFF
                if STOP_REQUESTED:
                    break
                if key in (27, ord("q")):
                    break
                continue

            # FPS update
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            # ROI
            x, y, ww, hh = roi_from_rel(frame.shape, ROI_REL)
            roi = frame[y:y+hh, x:x+ww]
            cv2.rectangle(frame, (x, y), (x+ww, y+hh), (255, 255, 255), 2)

            # Predict probabilities
            probs, top_idx, top_conf = predict_probs(model, preprocess, roi)
            p_no = float(probs[no_idx])
            p_good = float(probs[good_idx])
            p_bad  = float(probs[bad_idx])

            # --- Hysteresis presence logic driven by p_no
            product_like = (p_no < ENTER_NO_THR)
            empty_like   = (p_no > EXIT_NO_THR)

            if state == "IDLE":
                if product_like:
                    enter_streak += 1
                else:
                    enter_streak = 0

                if enter_streak >= ENTER_FRAMES:
                    state = "TRACKING"
                    enter_streak = 0
                    exit_streak = 0
                    burst = []

            elif state == "TRACKING":
                if empty_like:
                    exit_streak += 1
                else:
                    exit_streak = 0

                # collect burst frame
                b = blur_metric(roi)
                s = score_frame(p_no, p_good, p_bad, b)
                burst.append({
                    "roi": roi.copy(),
                    "probs": probs.copy(),
                    "blur": b,
                    "score": s,
                    "t": time.time(),
                })
                if len(burst) > MAX_BURST_FRAMES:
                    burst.pop(0)

                # finalize on exit
                if exit_streak >= EXIT_FRAMES:
                    result = finalize_one_item()
                    if result is not None:
                        decision, best_conf, pg, pb, pn = result
                        last_decision = decision
                        last_conf = best_conf

                        # show big decision briefly, then revert to NO PRODUCT
                        display_decision = decision
                        display_conf = best_conf
                        last_final_time = time.time()

                    state = "IDLE"
                    exit_streak = 0

            # --- Update big display label
            now = time.time()
            if state == "IDLE":
                if (now - last_final_time) > LAST_DECISION_HOLD_SEC:
                    display_decision = "NO PRODUCT"
                    display_conf = 0.0

            # --- Border and panel
            if display_decision == "GOOD":
                last_color = BORDER_GREEN
            elif display_decision == "BAD":
                last_color = BORDER_RED
            else:
                last_color = BORDER_YELLOW

            if state == "TRACKING":
                border = BORDER_YELLOW
                status = "TRACKING PRODUCT..."
            else:
                border = last_color
                status = "NO PRODUCT" if display_decision == "NO PRODUCT" else "FINALIZED"

            draw_full_border(frame, border, BORDER_THICKNESS)

            pred_line = f"p_no:{p_no:.2f}  p_good:{p_good:.2f}  p_bad:{p_bad:.2f}"
            if state == "TRACKING":
                pred_line += f"  burst:{len(burst)}"

            # MODIFIED: bigger text scales + thickness for the live panel
            lines = [
                (f"STATE: {state}   STATUS: {status}", 1.2, 3),
                (pred_line, 1.0, 3),
                (f"DISPLAY: {display_decision}  (conf {display_conf:.2f})", 1.1, 3),
                (f"Last finalized: {last_decision}  (conf {last_conf:.2f})", 0.9, 2),
                (f"Products: {seen_products}   Good: {good_count}   Bad: {bad_count}   Uncertain: {uncertain_count}", 1.0, 3),
                (f"ENTER(no<{ENTER_NO_THR:.2f})x{ENTER_FRAMES}   EXIT(no>{EXIT_NO_THR:.2f})x{EXIT_FRAMES}", 0.85, 2),
                (f"FPS: {fps:.1f}", 0.85, 2),
                ("Tap STOP to quit", 0.9, 2),
            ]
            draw_panel(frame, lines)

            # MODIFIED: make the BIG decision label bigger
            cv2.rectangle(frame, (20, frame.shape[0]-140), (360, frame.shape[0]-20), last_color, -1)
            cv2.putText(
                frame,
                display_decision,
                (40, frame.shape[0]-50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )

            # draw STOP button
            draw_stop_button(frame)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if STOP_REQUESTED:
                break
            if key in (27, ord("q")):
                break

    finally:
        print("\n================ FINAL RESULTS ================")
        print(f"Total products seen : {seen_products}")
        print(f"GOOD products       : {good_count}")
        print(f"BAD products        : {bad_count}")
        print(f"UNCERTAIN products  : {uncertain_count}")
        if SAVE_BEST_FRAMES:
            print(f"Saved best frames to: {SAVE_DIR.resolve()}")
        print("================================================\n")

        # ---- Write final results for Qt launcher ----
        if result_json_path:
            try:
                payload = {
                    "total": int(seen_products),
                    "good": int(good_count),
                    "bad": int(bad_count),
                    "uncertain": int(uncertain_count),
                }
                with open(result_json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] Could not write result JSON: {e}")


        try:
            if camera and camera.IsGrabbing():
                camera.StopGrabbing()
            if camera:
                camera.Close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

