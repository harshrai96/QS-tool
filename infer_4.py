# """
# Quarkball Live QC (Basler + PyTorch MobileNetV2) - 3-class robust live inference
#
# Assumes classifier trained on 3 classes: good, bad, no_product (or similar).
#
# Behavior:
# - Runs classifier on ROI every frame.
# - Uses p(no_product) with hysteresis + streaks to detect:
#     IDLE (no product) -> TRACKING (product present) -> IDLE (product gone)
# - While TRACKING, collects a burst of ROI frames + probs + blur metric.
# - On departure, selects best frame and finalizes ONE count (good/bad/uncertain).
# - Border: YELLOW=idle/uncertain, GREEN=good, RED=bad
# """
#
# import time
# from pathlib import Path
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from pypylon import pylon
#
#
# # -----------------------------
# # Config (tune these)
# # -----------------------------
# MODEL_PATH = Path("models/quarkball_mobilenet_v2_best.pth")
# WINDOW_NAME = "Quarkball Live QC"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# # Whole-frame thick border colors (BGR)
# BORDER_YELLOW = (0, 255, 255)
# BORDER_GREEN  = (0, 255, 0)
# BORDER_RED    = (0, 0, 255)
# BORDER_THICKNESS = 18
#
# # ROI where the quarkball passes (fractions of frame W/H)
# ROI_REL = (0.20, 0.20, 0.60, 0.60)  # (x,y,w,h)
#
# # --- Presence detection from classifier (hysteresis)
# # Enter TRACKING when p_no_product goes below ENTER_NO_THR for ENTER_FRAMES frames
# # Exit TRACKING when p_no_product goes above EXIT_NO_THR for EXIT_FRAMES frames
# ENTER_NO_THR = 0.35
# EXIT_NO_THR  = 0.75
# ENTER_FRAMES = 3
# EXIT_FRAMES  = 6
#
# # --- Burst capture
# MAX_BURST_FRAMES = 25          # max frames to store while product is present
# MIN_BURST_FRAMES = 5           # if less, still finalize but may be weaker
#
# # --- Final decision thresholds
# FINAL_CONF_THR = 0.60          # if max(p_good,p_bad) below this => "UNCERTAIN"
#
# # Optional: save chosen frame per product for boss/demo evidence
# SAVE_BEST_FRAMES = True
# SAVE_DIR = Path("captures_best")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)
#
# # Preprocess size (match training)
# RESIZE_TO = (256, 256)         # then center crop 224
#
#
# # -----------------------------
# # Model utilities
# # -----------------------------
# def build_mobilenet_v2(num_classes: int) -> nn.Module:
#     weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
#     model = models.mobilenet_v2(weights=weights)
#     in_features = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(in_features, num_classes)
#     return model
#
# def load_model(model_path: Path):
#     if not model_path.exists():
#         raise FileNotFoundError(f"Model not found: {model_path}")
#
#     ckpt = torch.load(model_path, map_location="cpu")
#     classes = ckpt.get("classes", None)
#     if not classes:
#         raise ValueError("Checkpoint missing 'classes'. Expected {'model_state_dict':..., 'classes':...}")
#
#     model = build_mobilenet_v2(len(classes))
#     model.load_state_dict(ckpt["model_state_dict"], strict=True)
#     model.to(DEVICE).eval()
#     return model, classes
#
# def make_preprocess():
#     mean = [0.485, 0.456, 0.406]
#     std  = [0.229, 0.224, 0.225]
#     return transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(RESIZE_TO),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])
#
# def find_class_index(classes, want: str):
#     """Find class index robustly by keywords."""
#     want = want.lower().replace(" ", "").replace("-", "").replace("_", "")
#     norm = []
#     for c in classes:
#         cc = str(c).lower().replace(" ", "").replace("-", "").replace("_", "")
#         norm.append(cc)
#
#     # Exact match
#     if want in norm:
#         return norm.index(want)
#
#     # Fuzzy contains
#     for i, cc in enumerate(norm):
#         if want in cc:
#             return i
#
#     raise ValueError(f"Could not find class '{want}' in classes: {classes}")
#
# @torch.no_grad()
# def predict_probs(model, preprocess, roi_bgr):
#     """Return probs array (num_classes,) and top1."""
#     rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
#     x = preprocess(rgb).unsqueeze(0).to(DEVICE)
#     logits = model(x)
#     probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
#     top_idx = int(np.argmax(probs))
#     top_conf = float(probs[top_idx])
#     return probs, top_idx, top_conf
#
#
# # -----------------------------
# # Basler camera
# # -----------------------------
# def open_basler_latest_only():
#     factory = pylon.TlFactory.GetInstance()
#     devices = factory.EnumerateDevices()
#     if len(devices) == 0:
#         raise RuntimeError("No Basler camera found. Check connection/drivers.")
#
#     camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
#     camera.Open()
#
#     converter = pylon.ImageFormatConverter()
#     converter.OutputPixelFormat = pylon.PixelType_BGR8packed
#     converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
#
#     camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
#     return camera, converter
#
# def grab_frame(camera, converter, timeout_ms=500):
#     grab = camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
#     if not grab or not grab.GrabSucceeded():
#         if grab:
#             grab.Release()
#         return None
#     img = converter.Convert(grab)
#     frame = img.GetArray()
#     grab.Release()
#     return frame
#
#
# # -----------------------------
# # Drawing / helpers
# # -----------------------------
# def roi_from_rel(frame_shape, roi_rel):
#     h, w = frame_shape[:2]
#     rx, ry, rw, rh = roi_rel
#     x = int(rx * w)
#     y = int(ry * h)
#     ww = int(rw * w)
#     hh = int(rh * h)
#     return x, y, ww, hh
#
# def draw_full_border(img, color, thickness):
#     h, w = img.shape[:2]
#     cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thickness)
#
# def draw_panel(img, lines, origin=(20, 40), line_h=30, pad=12):
#     x, y = origin
#     max_w = 0
#     total_h = line_h * len(lines)
#
#     for (text, scale, thick) in lines:
#         (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
#         max_w = max(max_w, tw)
#
#     x2 = x + max_w + pad * 2
#     y2 = y + total_h + pad
#     cv2.rectangle(img, (x - pad, y - pad), (x2, y2), (0, 0, 0), -1)
#
#     for i, (text, scale, thick) in enumerate(lines):
#         yy = y + i * line_h
#         cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
#                     scale, (255, 255, 255), thick, cv2.LINE_AA)
#
# def blur_metric(roi_bgr):
#     g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
#     return float(cv2.Laplacian(g, cv2.CV_64F).var())
#
# def clamp01(x: float) -> float:
#     return float(max(0.0, min(1.0, x)))
#
#
# # -----------------------------
# # Best-frame scoring
# # -----------------------------
# def score_frame(p_no, p_good, p_bad, blur_var):
#     """
#     Score frame to pick the best one:
#     - want low p_no (product clearly present)
#     - want high max(p_good,p_bad) (confident classification)
#     - want sharp image (blur metric high)
#     """
#     prodness = 1.0 - p_no
#     class_conf = max(p_good, p_bad)
#
#     # normalize blur loosely (tune divisor if needed)
#     blur_score = clamp01(blur_var / 200.0)
#
#     return prodness * class_conf * blur_score
#
#
# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     model, classes = load_model(MODEL_PATH)
#     preprocess = make_preprocess()
#
#     print("Loaded classes:", classes)
#
#     good_idx = find_class_index(classes, "good")
#     bad_idx  = find_class_index(classes, "bad")
#     # accept "no_product", "noprod", "empty", etc.
#     try:
#         no_idx = find_class_index(classes, "no_product")
#     except ValueError:
#         # common alternatives
#         for alt in ["noprod", "no", "empty", "background", "belt"]:
#             try:
#                 no_idx = find_class_index(classes, alt)
#                 break
#             except ValueError:
#                 no_idx = None
#         if no_idx is None:
#             raise ValueError(f"Could not find no-product class in {classes}. Rename to include 'no_product' or 'empty'.")
#
#     camera, converter = open_basler_latest_only()
#     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#
#     # Counters
#     seen_products = 0
#     good_count = 0
#     bad_count = 0
#     uncertain_count = 0
#
#     # State machine
#     state = "IDLE"  # IDLE or TRACKING
#     enter_streak = 0
#     exit_streak = 0
#
#     # Burst storage while TRACKING
#     burst = []  # list of dicts: roi, probs, blur, score, timestamp
#
#     # FPS meter
#     t0 = time.time()
#     frames = 0
#     fps = 0.0
#
#     def finalize_one_item():
#         nonlocal seen_products, good_count, bad_count, uncertain_count, burst
#
#         if len(burst) == 0:
#             return None
#
#         # pick best scored frame
#         best = max(burst, key=lambda d: d["score"])
#
#         p_good = float(best["probs"][good_idx])
#         p_bad  = float(best["probs"][bad_idx])
#         p_no   = float(best["probs"][no_idx])
#         best_idx = int(np.argmax(best["probs"]))
#         best_conf = float(best["probs"][best_idx])
#
#         seen_products += 1
#
#         decision = "UNCERTAIN"
#         if max(p_good, p_bad) >= FINAL_CONF_THR:
#             if p_good >= p_bad:
#                 decision = "GOOD"
#                 good_count += 1
#             else:
#                 decision = "BAD"
#                 bad_count += 1
#         else:
#             uncertain_count += 1
#
#         # save proof image
#         if SAVE_BEST_FRAMES:
#             ts = time.strftime("%Y%m%d_%H%M%S")
#             fname = f"{ts}_{seen_products:05d}_{decision}_pg{p_good:.2f}_pb{p_bad:.2f}_pno{p_no:.2f}_blur{best['blur']:.0f}.jpg"
#             cv2.imwrite(str(SAVE_DIR / fname), best["roi"])
#
#         # clear burst
#         burst = []
#
#         return decision, best_conf, p_good, p_bad, p_no
#
#     last_decision = "NO PRODUCT"
#     last_conf = 0.0
#     last_p_no = 1.0
#     last_p_good = 0.0
#     last_p_bad = 0.0
#
#     try:
#         while True:
#             frame = grab_frame(camera, converter)
#             if frame is None:
#                 key = cv2.waitKey(1) & 0xFF
#                 if key in (27, ord("q")):
#                     break
#                 continue
#
#             # FPS update
#             frames += 1
#             dt = time.time() - t0
#             if dt >= 1.0:
#                 fps = frames / dt
#                 frames = 0
#                 t0 = time.time()
#
#             # ROI
#             x, y, ww, hh = roi_from_rel(frame.shape, ROI_REL)
#             roi = frame[y:y+hh, x:x+ww]
#             cv2.rectangle(frame, (x, y), (x+ww, y+hh), (255, 255, 255), 2)
#
#             # Predict probabilities
#             probs, top_idx, top_conf = predict_probs(model, preprocess, roi)
#             p_no = float(probs[no_idx])
#             p_good = float(probs[good_idx])
#             p_bad  = float(probs[bad_idx])
#
#             last_p_no = p_no
#             last_p_good = p_good
#             last_p_bad = p_bad
#
#             # --- Hysteresis presence logic driven by p_no
#             product_like = (p_no < ENTER_NO_THR)
#             empty_like   = (p_no > EXIT_NO_THR)
#
#             if state == "IDLE":
#                 if product_like:
#                     enter_streak += 1
#                 else:
#                     enter_streak = 0
#
#                 if enter_streak >= ENTER_FRAMES:
#                     state = "TRACKING"
#                     enter_streak = 0
#                     exit_streak = 0
#                     burst = []  # start new burst
#
#             elif state == "TRACKING":
#                 if empty_like:
#                     exit_streak += 1
#                 else:
#                     exit_streak = 0
#
#                 # collect burst frame
#                 b = blur_metric(roi)
#                 s = score_frame(p_no, p_good, p_bad, b)
#                 burst.append({
#                     "roi": roi.copy(),
#                     "probs": probs.copy(),
#                     "blur": b,
#                     "score": s,
#                     "t": time.time(),
#                 })
#                 if len(burst) > MAX_BURST_FRAMES:
#                     # keep most recent; alternatively keep top-N by score
#                     burst.pop(0)
#
#                 # finalize on exit
#                 if exit_streak >= EXIT_FRAMES:
#                     result = finalize_one_item()
#                     if result is not None:
#                         decision, best_conf, pg, pb, pn = result
#                         last_decision = decision
#                         last_conf = best_conf
#                     state = "IDLE"
#                     exit_streak = 0
#
#             # --- UI / border based on current state / last decision
#             if state == "IDLE":
#                 # show "NO PRODUCT" unless we just finalized something
#                 status = "NO PRODUCT"
#                 border = BORDER_YELLOW
#                 pred_line = f"p_no:{p_no:.2f}  p_good:{p_good:.2f}  p_bad:{p_bad:.2f}"
#             else:
#                 # tracking
#                 status = "TRACKING PRODUCT..."
#                 border = BORDER_YELLOW
#                 pred_line = f"TRACK p_no:{p_no:.2f}  p_good:{p_good:.2f}  p_bad:{p_bad:.2f}  burst:{len(burst)}"
#
#             # show last finalized decision prominently
#             if last_decision == "GOOD":
#                 last_color = BORDER_GREEN
#             elif last_decision == "BAD":
#                 last_color = BORDER_RED
#             else:
#                 last_color = BORDER_YELLOW
#
#             draw_full_border(frame, border, BORDER_THICKNESS)
#
#             lines = [
#                 (f"STATE: {state}   STATUS: {status}", 0.9, 2),
#                 (pred_line, 0.75, 2),
#                 (f"LAST FINAL: {last_decision}  (conf {last_conf:.2f})", 0.85, 2),
#                 (f"Products: {seen_products}   Good: {good_count}   Bad: {bad_count}   Uncertain: {uncertain_count}", 0.75, 2),
#                 (f"ENTER thr(no<{ENTER_NO_THR:.2f})x{ENTER_FRAMES}   EXIT thr(no>{EXIT_NO_THR:.2f})x{EXIT_FRAMES}", 0.65, 2),
#                 (f"FPS: {fps:.1f}", 0.65, 2),
#                 ("Press 'q' or ESC to quit", 0.65, 1),
#             ]
#             draw_panel(frame, lines)
#
#             # small indicator box for last decision
#             cv2.rectangle(frame, (20, frame.shape[0]-80), (220, frame.shape[0]-20), last_color, -1)
#             cv2.putText(frame, last_decision, (30, frame.shape[0]-35),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3, cv2.LINE_AA)
#
#             cv2.imshow(WINDOW_NAME, frame)
#             key = cv2.waitKey(1) & 0xFF
#             if key in (27, ord("q")):
#                 break
#
#     finally:
#         print("\n================ FINAL RESULTS ================")
#         print(f"Total products seen : {seen_products}")
#         print(f"GOOD products       : {good_count}")
#         print(f"BAD products        : {bad_count}")
#         print(f"UNCERTAIN products  : {uncertain_count}")
#         if SAVE_BEST_FRAMES:
#             print(f"Saved best frames to: {SAVE_DIR.resolve()}")
#         print("================================================\n")
#
#         try:
#             if camera and camera.IsGrabbing():
#                 camera.StopGrabbing()
#             if camera:
#                 camera.Close()
#         except Exception:
#             pass
#         cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()

# with roi small, white box small, next ocde has more roi
# """
# Quarkball Live QC (Basler + PyTorch MobileNetV2) - 3-class robust live inference
#
# Assumes classifier trained on 3 classes: good, bad, no_product (or similar).
#
# Behavior:
# - Runs classifier on ROI every frame.
# - Uses p(no_product) with hysteresis + streaks to detect:
#     IDLE (no product) -> TRACKING (product present) -> IDLE (product gone)
# - While TRACKING, collects a burst of ROI frames + probs + blur metric.
# - On departure, selects best frame and finalizes ONE count (good/bad/uncertain).
# - UI:
#     - Big label shows GOOD/BAD briefly after a product is finalized, then returns to NO PRODUCT.
#     - Small line always shows Last finalized decision.
# """
#
# import time
# from pathlib import Path
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from pypylon import pylon
#
#
# # -----------------------------
# # Config (tune these)
# # -----------------------------
# MODEL_PATH = Path("models/quarkball_mobilenet_v2_best.pth")
# WINDOW_NAME = "Quarkball Live QC"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# # Whole-frame thick border colors (BGR)
# BORDER_YELLOW = (0, 255, 255)
# BORDER_GREEN  = (0, 255, 0)
# BORDER_RED    = (0, 0, 255)
# BORDER_THICKNESS = 18
#
# # ROI where the quarkball passes (fractions of frame W/H)
# ROI_REL = (0.20, 0.20, 0.60, 0.60)  # (x,y,w,h)
#
# # --- Presence detection from classifier (hysteresis)
# ENTER_NO_THR = 0.35
# EXIT_NO_THR  = 0.75
# ENTER_FRAMES = 3
# EXIT_FRAMES  = 6
#
# # --- Burst capture
# MAX_BURST_FRAMES = 25
# MIN_BURST_FRAMES = 5
#
# # --- Final decision thresholds
# FINAL_CONF_THR = 0.60
#
# # --- UI behavior
# LAST_DECISION_HOLD_SEC = 0.8  # show GOOD/BAD big for this long after item finalizes
#
# # Optional: save chosen frame per product for boss/demo evidence
# SAVE_BEST_FRAMES = True
# SAVE_DIR = Path("captures_best")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)
#
# # Preprocess size (match training)
# RESIZE_TO = (256, 256)  # then center crop 224
#
#
# # -----------------------------
# # Model utilities
# # -----------------------------
# def build_mobilenet_v2(num_classes: int) -> nn.Module:
#     weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
#     model = models.mobilenet_v2(weights=weights)
#     in_features = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(in_features, num_classes)
#     return model
#
# def load_model(model_path: Path):
#     if not model_path.exists():
#         raise FileNotFoundError(f"Model not found: {model_path}")
#
#     ckpt = torch.load(model_path, map_location="cpu")
#     classes = ckpt.get("classes", None)
#     if not classes:
#         raise ValueError("Checkpoint missing 'classes'. Expected {'model_state_dict':..., 'classes':...}")
#
#     model = build_mobilenet_v2(len(classes))
#     model.load_state_dict(ckpt["model_state_dict"], strict=True)
#     model.to(DEVICE).eval()
#     return model, classes
#
# def make_preprocess():
#     mean = [0.485, 0.456, 0.406]
#     std  = [0.229, 0.224, 0.225]
#     return transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(RESIZE_TO),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])
#
# def find_class_index(classes, want: str):
#     want = want.lower().replace(" ", "").replace("-", "").replace("_", "")
#     norm = []
#     for c in classes:
#         cc = str(c).lower().replace(" ", "").replace("-", "").replace("_", "")
#         norm.append(cc)
#
#     if want in norm:
#         return norm.index(want)
#
#     for i, cc in enumerate(norm):
#         if want in cc:
#             return i
#
#     raise ValueError(f"Could not find class '{want}' in classes: {classes}")
#
# @torch.no_grad()
# def predict_probs(model, preprocess, roi_bgr):
#     rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
#     x = preprocess(rgb).unsqueeze(0).to(DEVICE)
#     logits = model(x)
#     probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
#     top_idx = int(np.argmax(probs))
#     top_conf = float(probs[top_idx])
#     return probs, top_idx, top_conf
#
#
# # -----------------------------
# # Basler camera
# # -----------------------------
# def open_basler_latest_only():
#     factory = pylon.TlFactory.GetInstance()
#     devices = factory.EnumerateDevices()
#     if len(devices) == 0:
#         raise RuntimeError("No Basler camera found. Check connection/drivers.")
#
#     camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
#     camera.Open()
#
#     converter = pylon.ImageFormatConverter()
#     converter.OutputPixelFormat = pylon.PixelType_BGR8packed
#     converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
#
#     camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
#     return camera, converter
#
# def grab_frame(camera, converter, timeout_ms=500):
#     grab = camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
#     if not grab or not grab.GrabSucceeded():
#         if grab:
#             grab.Release()
#         return None
#     img = converter.Convert(grab)
#     frame = img.GetArray()
#     grab.Release()
#     return frame
#
#
# # -----------------------------
# # Drawing / helpers
# # -----------------------------
# def roi_from_rel(frame_shape, roi_rel):
#     h, w = frame_shape[:2]
#     rx, ry, rw, rh = roi_rel
#     x = int(rx * w)
#     y = int(ry * h)
#     ww = int(rw * w)
#     hh = int(rh * h)
#     return x, y, ww, hh
#
# def draw_full_border(img, color, thickness):
#     h, w = img.shape[:2]
#     cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thickness)
#
# def draw_panel(img, lines, origin=(20, 40), line_h=30, pad=12):
#     x, y = origin
#     max_w = 0
#     total_h = line_h * len(lines)
#
#     for (text, scale, thick) in lines:
#         (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
#         max_w = max(max_w, tw)
#
#     x2 = x + max_w + pad * 2
#     y2 = y + total_h + pad
#     cv2.rectangle(img, (x - pad, y - pad), (x2, y2), (0, 0, 0), -1)
#
#     for i, (text, scale, thick) in enumerate(lines):
#         yy = y + i * line_h
#         cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
#                     scale, (255, 255, 255), thick, cv2.LINE_AA)
#
# def blur_metric(roi_bgr):
#     g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
#     return float(cv2.Laplacian(g, cv2.CV_64F).var())
#
# def clamp01(x: float) -> float:
#     return float(max(0.0, min(1.0, x)))
#
# def score_frame(p_no, p_good, p_bad, blur_var):
#     prodness = 1.0 - p_no
#     class_conf = max(p_good, p_bad)
#     blur_score = clamp01(blur_var / 200.0)
#     return prodness * class_conf * blur_score
#
#
# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     model, classes = load_model(MODEL_PATH)
#     preprocess = make_preprocess()
#
#     print("Loaded classes:", classes)
#
#     good_idx = find_class_index(classes, "good")
#     bad_idx  = find_class_index(classes, "bad")
#     no_idx   = find_class_index(classes, "no_product")
#
#     camera, converter = open_basler_latest_only()
#     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#
#     # Counters
#     seen_products = 0
#     good_count = 0
#     bad_count = 0
#     uncertain_count = 0
#
#     # State machine
#     state = "IDLE"  # IDLE or TRACKING
#     enter_streak = 0
#     exit_streak = 0
#
#     # Burst storage while TRACKING
#     burst = []
#
#     # Last finalized decision (for logging)
#     last_decision = "NONE"
#     last_conf = 0.0
#
#     # Display decision (big label) + timer
#     display_decision = "NO PRODUCT"
#     display_conf = 0.0
#     last_final_time = 0.0
#
#     # FPS meter
#     t0 = time.time()
#     frames = 0
#     fps = 0.0
#
#     def finalize_one_item():
#         nonlocal seen_products, good_count, bad_count, uncertain_count, burst
#
#         if len(burst) == 0:
#             return None
#
#         best = max(burst, key=lambda d: d["score"])
#
#         p_good = float(best["probs"][good_idx])
#         p_bad  = float(best["probs"][bad_idx])
#         p_no   = float(best["probs"][no_idx])
#
#         seen_products += 1
#
#         decision = "UNCERTAIN"
#         if max(p_good, p_bad) >= FINAL_CONF_THR:
#             if p_good >= p_bad:
#                 decision = "GOOD"
#                 good_count += 1
#             else:
#                 decision = "BAD"
#                 bad_count += 1
#         else:
#             uncertain_count += 1
#
#         # save proof image
#         if SAVE_BEST_FRAMES:
#             ts = time.strftime("%Y%m%d_%H%M%S")
#             fname = (
#                 f"{ts}_{seen_products:05d}_{decision}"
#                 f"_pg{p_good:.2f}_pb{p_bad:.2f}_pno{p_no:.2f}_blur{best['blur']:.0f}.jpg"
#             )
#             cv2.imwrite(str(SAVE_DIR / fname), best["roi"])
#
#         burst = []
#         return decision, max(p_good, p_bad), p_good, p_bad, p_no
#
#     try:
#         while True:
#             frame = grab_frame(camera, converter)
#             if frame is None:
#                 key = cv2.waitKey(1) & 0xFF
#                 if key in (27, ord("q")):
#                     break
#                 continue
#
#             # FPS update
#             frames += 1
#             dt = time.time() - t0
#             if dt >= 1.0:
#                 fps = frames / dt
#                 frames = 0
#                 t0 = time.time()
#
#             # ROI
#             x, y, ww, hh = roi_from_rel(frame.shape, ROI_REL)
#             roi = frame[y:y+hh, x:x+ww]
#             cv2.rectangle(frame, (x, y), (x+ww, y+hh), (255, 255, 255), 2)
#
#             # Predict probabilities
#             probs, top_idx, top_conf = predict_probs(model, preprocess, roi)
#             p_no = float(probs[no_idx])
#             p_good = float(probs[good_idx])
#             p_bad  = float(probs[bad_idx])
#
#             # --- Hysteresis presence logic driven by p_no
#             product_like = (p_no < ENTER_NO_THR)
#             empty_like   = (p_no > EXIT_NO_THR)
#
#             if state == "IDLE":
#                 if product_like:
#                     enter_streak += 1
#                 else:
#                     enter_streak = 0
#
#                 if enter_streak >= ENTER_FRAMES:
#                     state = "TRACKING"
#                     enter_streak = 0
#                     exit_streak = 0
#                     burst = []
#
#             elif state == "TRACKING":
#                 if empty_like:
#                     exit_streak += 1
#                 else:
#                     exit_streak = 0
#
#                 # collect burst frame
#                 b = blur_metric(roi)
#                 s = score_frame(p_no, p_good, p_bad, b)
#                 burst.append({
#                     "roi": roi.copy(),
#                     "probs": probs.copy(),
#                     "blur": b,
#                     "score": s,
#                     "t": time.time(),
#                 })
#                 if len(burst) > MAX_BURST_FRAMES:
#                     burst.pop(0)
#
#                 # finalize on exit
#                 if exit_streak >= EXIT_FRAMES:
#                     result = finalize_one_item()
#                     if result is not None:
#                         decision, best_conf, pg, pb, pn = result
#                         last_decision = decision
#                         last_conf = best_conf
#
#                         # NEW: show big decision briefly, then revert to NO PRODUCT
#                         display_decision = decision
#                         display_conf = best_conf
#                         last_final_time = time.time()
#
#                     state = "IDLE"
#                     exit_streak = 0
#
#             # --- Update big display label
#             now = time.time()
#             if state == "IDLE":
#                 if (now - last_final_time) > LAST_DECISION_HOLD_SEC:
#                     display_decision = "NO PRODUCT"
#                     display_conf = 0.0
#
#             # --- Border and panel
#             if display_decision == "GOOD":
#                 last_color = BORDER_GREEN
#             elif display_decision == "BAD":
#                 last_color = BORDER_RED
#             else:
#                 last_color = BORDER_YELLOW
#
#             # frame border: show tracking as yellow, show final decision color briefly
#             # If you prefer border always reflect big label, set border = last_color always.
#             if state == "TRACKING":
#                 border = BORDER_YELLOW
#                 status = "TRACKING PRODUCT..."
#             else:
#                 border = last_color
#                 status = "NO PRODUCT" if display_decision == "NO PRODUCT" else "FINALIZED"
#
#             draw_full_border(frame, border, BORDER_THICKNESS)
#
#             pred_line = f"p_no:{p_no:.2f}  p_good:{p_good:.2f}  p_bad:{p_bad:.2f}"
#             if state == "TRACKING":
#                 pred_line += f"  burst:{len(burst)}"
#
#             lines = [
#                 (f"STATE: {state}   STATUS: {status}", 0.9, 2),
#                 (pred_line, 0.75, 2),
#                 (f"DISPLAY: {display_decision}  (conf {display_conf:.2f})", 0.85, 2),
#                 (f"Last finalized: {last_decision}  (conf {last_conf:.2f})", 0.65, 2),
#                 (f"Products: {seen_products}   Good: {good_count}   Bad: {bad_count}   Uncertain: {uncertain_count}", 0.75, 2),
#                 (f"ENTER(no<{ENTER_NO_THR:.2f})x{ENTER_FRAMES}   EXIT(no>{EXIT_NO_THR:.2f})x{EXIT_FRAMES}", 0.65, 2),
#                 (f"FPS: {fps:.1f}", 0.65, 2),
#                 ("Press 'q' or ESC to quit", 0.65, 1),
#             ]
#             draw_panel(frame, lines)
#
#             # Big indicator box
#             cv2.rectangle(frame, (20, frame.shape[0]-80), (260, frame.shape[0]-20), last_color, -1)
#             cv2.putText(frame, display_decision, (30, frame.shape[0]-35),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
#
#             cv2.imshow(WINDOW_NAME, frame)
#             key = cv2.waitKey(1) & 0xFF
#             if key in (27, ord("q")):
#                 break
#
#     finally:
#         print("\n================ FINAL RESULTS ================")
#         print(f"Total products seen : {seen_products}")
#         print(f"GOOD products       : {good_count}")
#         print(f"BAD products        : {bad_count}")
#         print(f"UNCERTAIN products  : {uncertain_count}")
#         if SAVE_BEST_FRAMES:
#             print(f"Saved best frames to: {SAVE_DIR.resolve()}")
#         print("================================================\n")
#
#         try:
#             if camera and camera.IsGrabbing():
#                 camera.StopGrabbing()
#             if camera:
#                 camera.Close()
#         except Exception:
#             pass
#         cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()
#
#
#


# """
# Quarkball Live QC (Basler + PyTorch MobileNetV2) - 3-class robust live inference
#
# Assumes classifier trained on 3 classes: good, bad, no_product (or similar).
#
# Behavior:
# - Runs classifier on ROI every frame.
# - Uses p(no_product) with hysteresis + streaks to detect:
#     IDLE (no product) -> TRACKING (product present) -> IDLE (product gone)
# - While TRACKING, collects a burst of best-scoring frames.
# - When product exits, finalize decision using burst aggregator.
# - Optional save best frame per product.
#
# Keys:
# - Tap STOP (touchscreen/mouse) OR press 'q' or ESC to quit.
# """
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
# STOP BUTTON (only addition)
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
    """Draw a red STOP button at top-right and update STOP_RECT."""
    global STOP_RECT
    h, w = frame.shape[:2]
    bw, bh = 200, 70
    m = 20
    x2 = w - m
    x1 = x2 - bw
    y1 = m
    y2 = y1 + bh
    STOP_RECT = (x1, y1, x2, y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    text = "STOP"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
    tx = x1 + (bw - tw) // 2
    ty = y1 + (bh + th) // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (255, 255, 255), 3, cv2.LINE_AA)


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


def draw_panel(img, lines, origin=(20, 40), line_h=30, pad=12):
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


# -----------------------------
# Main
# -----------------------------
def main():
    global STOP_REQUESTED
    model, classes = load_model(MODEL_PATH)
    preprocess = make_preprocess()

    print("Loaded classes:", classes)

    good_idx = find_class_index(classes, "good")
    bad_idx  = find_class_index(classes, "bad")
    no_idx   = find_class_index(classes, "no_product")

    camera, converter = open_basler_latest_only()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # NEW: enable clicking/tapping STOP
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

            lines = [
                (f"STATE: {state}   STATUS: {status}", 0.9, 2),
                (pred_line, 0.75, 2),
                (f"DISPLAY: {display_decision}  (conf {display_conf:.2f})", 0.85, 2),
                (f"Last finalized: {last_decision}  (conf {last_conf:.2f})", 0.65, 2),
                (f"Products: {seen_products}   Good: {good_count}   Bad: {bad_count}   Uncertain: {uncertain_count}", 0.75, 2),
                (f"ENTER(no<{ENTER_NO_THR:.2f})x{ENTER_FRAMES}   EXIT(no>{EXIT_NO_THR:.2f})x{EXIT_FRAMES}", 0.65, 2),
                (f"FPS: {fps:.1f}", 0.65, 2),
                ("Tap STOP or press 'q' or ESC to quit", 0.65, 1),
            ]
            draw_panel(frame, lines)

            # Big indicator box
            cv2.rectangle(frame, (20, frame.shape[0]-80), (260, frame.shape[0]-20), last_color, -1)
            cv2.putText(frame, display_decision, (30, frame.shape[0]-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)

            # NEW: draw STOP button (does not affect ROI/inference)
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
