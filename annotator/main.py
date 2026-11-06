import os
import sys
import copy
import math
from datetime import datetime
import json
from collections import deque

import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from image_utils import to_8bit, apply_clahe

DEFAULT_ROI_UM = 300.0
LASTDIR_PATH = os.path.join(os.path.expanduser("~"), ".floodseg_lastdir_v2")
ANNOT_DOT_RADIUS_PX = 6
DELETE_THRESHOLD_PX = 12

def save_last_dir(path):
    try:
        with open(LASTDIR_PATH, "w", encoding="utf8") as f:
            json.dump({"last_dir": path}, f)
    except:
        pass

def load_last_dir():
    try:
        if os.path.exists(LASTDIR_PATH):
            with open(LASTDIR_PATH, "r", encoding="utf8") as f:
                data = json.load(f)
                return data.get("last_dir", None)
    except:
        pass
    return None

def choose_file_via_dialog(initialdir=".", title="Select a TIFF image"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(initialdir=initialdir, title=title,
                                               filetypes=[("TIFF files", "*.tif;*.tiff"), ("NumPy files", "*.npy;*.npz"), ("All files", "*.*")])
        root.destroy()
        return file_path if file_path else None
    except:
        return None

def draw_annotations_overlay(base_rgb, annotations):
    overlay = base_rgb.copy().astype(np.uint8)
    for a in annotations:
        x, y = int(round(a["x_px"])), int(round(a["y_px"]))
        color = a.get("color", (255, 0, 0))
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.circle(overlay, (x, y), ANNOT_DOT_RADIUS_PX, bgr, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x, y), ANNOT_DOT_RADIUS_PX, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        label = str(a.get("id", ""))
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(label, font, scale, thickness)
        tx = x + ANNOT_DOT_RADIUS_PX + 4
        ty = y + int(text_size[1] / 2)
        cv2.putText(overlay, label, (tx, ty), font, scale, (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(overlay, label, (tx, ty), font, scale, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return overlay

def compute_summary_metrics(records, pixel_size_um, img_w, img_h):
    n = len(records)
    if n == 0:
        return {"total_cells": 0, "cell_density_per_mm2": 0.0, "mean_nn_distance_um": 0.0, "mean_pairwise_distance_um": 0.0,
                "roi_width_um": img_w * pixel_size_um, "roi_height_um": img_h * pixel_size_um}
    cents = np.array([[r["centroid_x_um"], r["centroid_y_um"]] for r in records], dtype=float)
    diff = cents[:, None, :] - cents[None, :, :]
    dists = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(dists, np.inf)
    nn = np.min(dists, axis=1)
    mean_nn = float(np.mean(nn))
    iu = np.triu_indices(n, k=1)
    mean_pair = float(np.mean(dists[iu])) if len(iu[0]) > 0 else 0.0
    roi_w_mm = (img_w * pixel_size_um) / 1000.0
    roi_h_mm = (img_h * pixel_size_um) / 1000.0
    roi_area_mm2 = roi_w_mm * roi_h_mm
    density = n / roi_area_mm2 if roi_area_mm2 > 0 else 0
    return {"total_cells": int(n), "cell_density_per_mm2": float(density), "mean_nn_distance_um": float(mean_nn),
            "mean_pairwise_distance_um": float(mean_pair), "roi_width_um": img_w * pixel_size_um, "roi_height_um": img_h * pixel_size_um}

def write_csv_with_summary(records, csvpath, pixel_size_um, img_w, img_h):
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=[
        "timestamp", "image", "seed_x_px", "seed_y_px", "centroid_x_px", "centroid_y_px", "centroid_x_um", "centroid_y_um",
        "area_px", "area_um2", "equiv_diam_px", "equiv_diam_um", "perimeter_px", "perimeter_um", "pixel_size_um", "cell_id"])
    summary = compute_summary_metrics(records, pixel_size_um, img_w, img_h)
    summary_row = {"timestamp": datetime.utcnow().isoformat(), "image": "SUMMARY", "seed_x_px": "", "seed_y_px": "",
                   "centroid_x_px": "", "centroid_y_px": "", "centroid_x_um": "", "centroid_y_um": "", "area_px": "",
                   "area_um2": "", "equiv_diam_px": "", "equiv_diam_um", "perimeter_px": "", "perimeter_um": "",
                   "pixel_size_um": pixel_size_um, "total_cells": summary["total_cells"], "cell_density_per_mm2": summary["cell_density_per_mm2"],
                   "mean_nn_distance_um": summary["mean_nn_distance_um"], "mean_pairwise_distance_um": summary["mean_pairwise_distance_um"],
                   "roi_width_um": summary["roi_width_um"], "roi_height_um": summary["roi_height_um"]}
    df_summary = pd.DataFrame([summary_row])
    try:
        df.to_csv(csvpath, index=False)
        df_summary.to_csv(csvpath, mode="a", index=False, header=False)
    except Exception as e:
        print("Failed to write CSV:", e)

def save_overlay_png(overlay_rgb, outpath):
    try:
        bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(outpath, bgr)
    except Exception as e:
        print("Failed to save overlay PNG:", e)

class InteractiveSegmenter:
    def __init__(self, image_path):
        self.image_path = image_path

        try:
            lp = image_path.lower()
            if lp.endswith((".npy", ".npz")):
                data = np.load(image_path, allow_pickle=False)
                if isinstance(data, np.lib.npyio.NpzFile):
                    keys = list(data.keys())
                    arr = data[keys[0]] if keys else None
                else:
                    arr = data
                self.img = np.asarray(arr)
            else:
                self.img = tifffile.imread(image_path)
        except Exception as e:
            try:
                self.img = tifffile.imread(image_path)
            except:
                raise RuntimeError(f"Failed to load image '{image_path}': {e}")

        if self.img.ndim > 2:
            self.img = self.img[..., 0]
        self.img8 = to_8bit(self.img)
        self.h, self.w = self.img8.shape
        self.pixel_size_um = DEFAULT_ROI_UM / max(self.h, self.w)
        self.img_clahe = apply_clahe(self.img8, clipLimit=2.0)
        self.base_rgb = cv2.cvtColor(self.img_clahe, cv2.COLOR_GRAY2RGB)

        self.annotations = []
        self.records = []
        self.undo_history = []
        self.redo_history = []
        self.shift_down = False
        self.ctrl_down = False
        self.mouse_pressed = False
        self.exit_requested = False

        self.dir = os.path.dirname(os.path.abspath(self.image_path))
        self.basename = os.path.splitext(os.path.basename(image_path))[0]
        self.csv_path = os.path.join(self.dir, f"{self.basename}_measurements.csv")
        self.overlay_path = os.path.join(self.dir, f"{self.basename}_overlay.png")

        self._init_figure()

    def _init_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)
        self.update_overlay()
        self.ax.set_title(f"Annotation mode — left-click to add numbered dot, right-click to delete. Image: {os.path.basename(self.image_path)}")
        self.ax.axis("off")

        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_key_press = self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.cid_key_release = self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.cid_close = self.fig.canvas.mpl_connect("close_event", self.on_close)

    def push_history(self):
        ann_copy = [dict(a) for a in self.annotations]
        rec_copy = copy.deepcopy(self.records)
        self.undo_history.append((ann_copy, rec_copy))
        self.redo_history.clear()

    def pop_history(self):
        if not self.undo_history:
            return False
        current_state = ([dict(a) for a in self.annotations], copy.deepcopy(self.records))
        last_ann, last_rec = self.undo_history.pop()
        self.redo_history.append(current_state)
        self.annotations = [dict(a) for a in last_ann]
        self.records = copy.deepcopy(last_rec)
        self.update_overlay()
        write_csv_with_summary(self.records, self.csv_path, self.pixel_size_um, self.w, self.h)
        save_overlay_png(draw_annotations_overlay(self.base_rgb, self.annotations), self.overlay_path)
        return True

    def on_key_press(self, event):
        key = event.key
        if key is None:
            return
        if "shift" in key:
            self.shift_down = True
        if "control" in key or "ctrl" in key:
            self.ctrl_down = True
        if key.lower() in ("ctrl+z", "control+z", "cmd+z"):
            if self.pop_history():
                print("Undo: reverted to previous annotation state.")
            else:
                print("Nothing to undo.")
        if key.lower() in ("ctrl+y", "control+y", "cmd+y"):
            if self.redo_history:
                cur = ([dict(a) for a in self.annotations], copy.deepcopy(self.records))
                self.undo_history.append(cur)
                ann, rec = self.redo_history.pop()
                self.annotations = [dict(a) for a in ann]
                self.records = copy.deepcopy(rec)
                self.update_overlay()
                write_csv_with_summary(self.records, self.csv_path, self.pixel_size_um, self.w, self.h)
                save_overlay_png(draw_annotations_overlay(self.base_rgb, self.annotations), self.overlay_path)
                print("Redo: reapplied annotation state.")
            else:
                print("Nothing to redo.")
        if key.lower() in ("q", "escape"):
            self.exit_requested = True
            plt.close(self.fig)

    def on_key_release(self, event):
        key = event.key
        if key is None:
            return
        if "shift" in key:
            self.shift_down = False
        if "control" in key or "ctrl" in key:
            self.ctrl_down = False

    def on_button_press(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        self.mouse_pressed = True
        if event.button == 1:
            self.push_history()
            self.add_annotation_at(x, y)
        if event.button == 3:
            if not self.annotations:
                return
            dists = [math.hypot(a['x_px'] - x, a['y_px'] - y) for a in self.annotations]
            idx = int(np.argmin(dists))
            if dists[idx] <= DELETE_THRESHOLD_PX:
                self.push_history()
                removed = self.annotations.pop(idx)
                for j, r in enumerate(self.records[::-1]):
                    if int(r.get("centroid_x_px", -1)) == int(removed['x_px']) and int(r.get("centroid_y_px", -1)) == int(removed['y_px']):
                        idx_to_remove = len(self.records) - 1 - j
                        self.records.pop(idx_to_remove)
                        break
                write_csv_with_summary(self.records, self.csv_path, self.pixel_size_um, self.w, self.h)
                save_overlay_png(draw_annotations_overlay(self.base_rgb, self.annotations), self.overlay_path)
                self.update_overlay()
                print(f"Deleted annotation #{removed.get('id')} at x={removed.get('x_px')}, y={removed.get('y_px')}")

    def on_motion(self, event):
        return

    def on_button_release(self, event):
        self.mouse_pressed = False

    def on_close(self, event):
        self.save_all_outputs()

    def add_annotation_at(self, x_px, y_px):
        next_id = (max([a.get("id", 0) for a in self.annotations]) + 1) if self.annotations else 1
        color = tuple(int(c) for c in np.random.choice(range(80, 256), size=3))
        x_um = float(x_px) * self.pixel_size_um
        y_um = float(y_px) * self.pixel_size_um
        timestamp = datetime.utcnow().isoformat()
        ann = {"id": int(next_id), "x_px": int(x_px), "y_px": int(y_px), "x_um": float(x_um), "y_um": float(y_um),
               "color": color, "timestamp": timestamp}
        self.annotations.append(ann)
        rec = {"timestamp": timestamp, "image": os.path.basename(self.image_path), "seed_x_px": int(x_px), "seed_y_px": int(y_px),
               "centroid_x_px": float(x_px), "centroid_y_px": float(y_px), "centroid_x_um": float(x_um), "centroid_y_um": float(y_um),
               "area_px": 0, "area_um2": 0.0, "equiv_diam_px": 0.0, "equiv_diam_um": 0.0, "perimeter_px": 0.0, "perimeter_um": 0.0,
               "pixel_size_um": self.pixel_size_um, "cell_id": int(next_id)}
        self.records.append(rec)
        write_csv_with_summary(self.records, self.csv_path, self.pixel_size_um, self.w, self.h)
        save_overlay_png(draw_annotations_overlay(self.base_rgb, self.annotations), self.overlay_path)
        self.update_overlay()
        print(f"Added annotation #{next_id} at x={x_px}, y={y_px}.")

    def update_overlay(self):
        overlay_rgb = draw_annotations_overlay(self.base_rgb, self.annotations)
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(overlay_rgb)
        else:
            self.imshow_obj.set_data(overlay_rgb)
        self.fig.canvas.draw_idle()

    def save_all_outputs(self):
        write_csv_with_summary(self.records, self.csv_path, self.pixel_size_um, self.w, self.h)
        save_overlay_png(draw_annotations_overlay(self.base_rgb, self.annotations), self.overlay_path)
        print(f"Saved outputs for {self.basename}: CSV={self.csv_path} overlay={self.overlay_path}")

def main_loop():
    parser = argparse.ArgumentParser(description="Interactive annotation (point-based).")
    parser.add_argument("image", nargs="?", help="Optional path to TIFF/NumPy image. If omitted a file dialog will open.")
    args = parser.parse_args()

    exit_all = False
    while not exit_all:
        image_path = args.image
        if not image_path:
            lastdir = load_last_dir() or "."
            image_path = choose_file_via_dialog(initialdir=lastdir)
            if not image_path:
                print("No file selected, exiting.")
                return
            save_last_dir(os.path.dirname(os.path.abspath(image_path)))
        if not os.path.exists(image_path):
            print("Image not found:", image_path)
            return
        seg = InteractiveSegmenter(image_path)
        plt.show()
        if seg.exit_requested:
            exit_all = True
        else:
            args.image = None

if __name__ == "__main__":
    main_loop()