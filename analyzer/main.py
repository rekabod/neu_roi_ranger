import os
import sys
import csv
import argparse
from datetime import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except:
    TIFFFILE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except:
    PIL_AVAILABLE = False

try:
    import tkinter as _tk
    from tkinter import filedialog as _filedialog
    TK_AVAILABLE = True
except:
    TK_AVAILABLE = False

ROI_EXPECTED_SIZE = (500, 500)

def sanitize_path(p):
    return (p or "").strip().strip('"').strip("'").replace("\\", "/").strip()

def prompt_for_directory():
    if not TK_AVAILABLE:
        return ""
    try:
        root = _tk.Tk()
        root.withdraw()
        path = _filedialog.askdirectory(title="Select ROI directory")
        root.destroy()
        return sanitize_path(path)
    except:
        return ""

def is_npy_file(path):
    lower_path = path.lower()
    return lower_path.endswith('.npy') or lower_path.endswith('.tif.npy') or lower_path.endswith('.tiff.npy')

def warn_if_not_500x500(img, path):
    try:
        h, w = img.shape[:2]
        if (w, h) != ROI_EXPECTED_SIZE:
            print(f"WARNING: ROI is {w}x{h}, expected {ROI_EXPECTED_SIZE[0]}x{ROI_EXPECTED_SIZE[1]}: {os.path.basename(path)}")
    except:
        pass

def _normalize_to_u8(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img[..., :3].mean(axis=-1)
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx > mn:
        img = (img - mn) / (mx - mn) * 255.0
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if is_npy_file(path):
        try:
            img = np.load(path, allow_pickle=True)
            if img.dtype == object:
                img = img.item() if img.shape == () else img[0]
            img = _normalize_to_u8(img)
            print(f"Loaded .npy file: {os.path.basename(path)} | shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
            return img
        except Exception as e:
            print(f"Error loading .npy file {path}: {e}")
            import traceback
            traceback.print_exc()
            warnings.warn(f"Failed to load .npy file: {e}")
            raise
    if TIFFFILE_AVAILABLE and path.lower().endswith(('.tif', '.tiff')):
        try:
            img = tifffile.imread(path)
            img = _normalize_to_u8(img)
            return img
        except Exception as e:
            warnings.warn(f"tifffile failed: {e}")
    if PIL_AVAILABLE:
        try:
            Image.MAX_IMAGE_PIXELS = None
            pil_img = Image.open(path)
            if pil_img.mode != 'L':
                pil_img = pil_img.convert('L')
            img = np.array(pil_img, dtype=np.uint8)
            return img
        except Exception as e:
            warnings.warn(f"Pillow failed: {e}")
    if CV2_AVAILABLE:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError("cv2.imread returned None")
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = _normalize_to_u8(img)
            return img
        except Exception as e:
            warnings.warn(f"OpenCV failed: {e}")
    raise RuntimeError("Failed to load image with any available backend")

def find_image_files(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            lower_f = f.lower()
            if (lower_f.endswith(('.tif', '.tiff', '.npy', '.jpg', '.jpeg')) or lower_f.endswith('.tif.npy') or lower_f.endswith('.tiff.npy')):
                image_files.append(os.path.join(root, f))
    return sorted(image_files)

def get_file_format(filepath):
    basename = os.path.basename(filepath).lower()
    if basename.endswith('.tif.npy'):
        return '.tif.npy'
    elif basename.endswith('.tiff.npy'):
        return '.tiff.npy'
    else:
        return os.path.splitext(filepath)[1]

def apply_brightness_contrast(img, brightness=0, exposure=1.0):
    img_float = img.astype(np.float32)
    img_float = img_float * exposure
    img_float = img_float + brightness
    img_float = np.clip(img_float, 0, 255)
    return img_float

def apply_threshold_inverted(img, threshold):
    return img <= threshold

def calculate_coverage(mask):
    if mask.size == 0:
        return 0.0
    return 100.0 * np.sum(mask) / mask.size

def polygon_to_mask(polygon_points, shape):
    if len(polygon_points) < 3:
        return np.zeros(shape, dtype=bool)
    height, width = shape
    y, x = np.mgrid[:height, :width]
    points = np.vstack((x.ravel(), y.ravel())).T
    path = Path(polygon_points)
    mask = path.contains_points(points)
    return mask.reshape((height, width))

class ROIAnalyzer:
    def __init__(self, roi_files, output_dir):
        self.roi_files = roi_files
        self.output_dir = output_dir
        self.current_idx = 0
        self.roi_data = {}
        self.auto_saved = False
        self.mode = 'threshold'
        self.cell_positions = {}
        self.cell_artists = []
        self.drawn_regions = {}
        self.erased_regions = {}
        self.drawing_active = False
        self.erasing_active = False
        self.current_drawing = []
        self.drawing_line = None
        self.region_artists = {}
        self.region_history = {}
        self.brightness = 0
        self.exposure = 1.0
        self.threshold = 128
        self.last_saved_state = {}

        self.original_img = load_image(self.roi_files[self.current_idx])
        warn_if_not_500x500(self.original_img, self.roi_files[self.current_idx])
        self.current_img = None
        self.current_mask = None
        self._build_figure()
        self._update_display()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 9))
        main_gs = GridSpec(1, 2, figure=self.fig, left=0.02, right=0.98, top=0.96, bottom=0.08, wspace=0.05, width_ratios=[2.5, 1])
        image_gs = main_gs[0].subgridspec(1, 2, wspace=0.10)
        self.ax_adjusted = self.fig.add_subplot(image_gs[0])
        self.ax_overlay = self.fig.add_subplot(image_gs[1])
        for ax in [self.ax_adjusted, self.ax_overlay]:
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
        self.ax_adjusted.set_title("Adjusted", fontsize=13, fontweight='bold', pad=10)
        self.ax_overlay.set_title("Analysis", fontsize=13, fontweight='bold', pad=10)

        control_left = 0.74
        control_top_start = 0.88
        slider_width = 0.22
        slider_height = 0.025
        slider_vspace = 0.055

        ax_mode = self.fig.add_axes([control_left, control_top_start, 0.18, 0.10])
        self.radio_mode = RadioButtons(ax_mode, ('Threshold', 'Count'), active=0)
        self.radio_mode.on_clicked(self._on_mode_change)

        current_y = control_top_start - 0.15
        ax_brightness = self.fig.add_axes([control_left, current_y, slider_width, slider_height])
        current_y -= slider_vspace
        ax_exposure = self.fig.add_axes([control_left, current_y, slider_width, slider_height])
        current_y -= slider_vspace
        ax_threshold = self.fig.add_axes([control_left, current_y, slider_width, slider_height])
        current_y -= slider_vspace
        ax_roi_slider = self.fig.add_axes([control_left, current_y, slider_width, slider_height])

        self.slider_brightness = Slider(ax_brightness, 'Bright', -100, 100, valinit=self.brightness, valstep=1)
        self.slider_exposure = Slider(ax_exposure, 'Exposure', 0.1, 2.0, valinit=self.exposure, valstep=0.05)
        self.slider_threshold = Slider(ax_threshold, 'Thresh', 0, 255, valinit=self.threshold, valstep=1)
        self.slider_roi = Slider(ax_roi_slider, 'ROI #', 1, len(self.roi_files), valinit=1, valstep=1, valfmt='%d')

        self.slider_brightness.on_changed(self._on_brightness_changed)
        self.slider_exposure.on_changed(self._on_exposure_changed)
        self.slider_threshold.on_changed(self._on_threshold_changed)
        self.slider_roi.on_changed(self._on_roi_changed)

        button_width = 0.10
        button_height = 0.04
        current_y -= 0.08
        ax_prev = self.fig.add_axes([control_left, current_y, button_width, button_height])
        ax_next = self.fig.add_axes([control_left + 0.12, current_y, button_width, button_height])
        self.btn_prev = Button(ax_prev, '◄ Prev')
        self.btn_next = Button(ax_next, 'Next ►')
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)

        current_y -= 0.06
        ax_clear = self.fig.add_axes([control_left, current_y, button_width * 2 + 0.12, button_height])
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self._on_clear)

        current_y -= 0.05
        self.txt_info = self.fig.text(control_left, current_y, '', fontsize=9, family='monospace', verticalalignment='top')

        instructions = ("Navigation: ←/→ or A/D  |  Threshold: Drag=add, Shift+Drag=erase  |  Count: Click cells  |  Ctrl+Z: Undo  |  Q/ESC: Close & save")
        self.fig.text(0.5, 0.02, instructions, fontsize=9, ha='center', style='italic', color='#555')

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        plt.show(block=False)

    def _on_mode_change(self, label):
        old_mode = self.mode
        self.mode = 'threshold' if label == 'Threshold' else 'count'
        if old_mode != self.mode:
            self._save_current_mode_data()
            if self.mode == 'count':
                self.btn_clear.label.set_text('Clear Counts')
            else:
                self.btn_clear.label.set_text('Clear Regions')
            self._update_display()
            print(f"Switched to {self.mode.upper()} mode")

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax_overlay:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.mode == 'threshold':
            if event.button == 1:
                if event.key == 'shift':
                    self.erasing_active = True
                    self.drawing_active = False
                else:
                    self.drawing_active = True
                    self.erasing_active = False
                self.current_drawing = [(event.xdata, event.ydata)]
        elif self.mode == 'count':
            if event.button == 3:
                if self.current_idx not in self.cell_positions:
                    return
                positions = self.cell_positions[self.current_idx]
                if not positions:
                    return
                x, y = event.xdata, event.ydata
                distances = [np.sqrt((px - x) ** 2 + (py - y) ** 2) for px, py in positions]
                min_idx = int(np.argmin(distances))
                if distances[min_idx] < 20:
                    positions.pop(min_idx)
                    self._save_current_mode_data()
                    self._update_display()
                return
            if event.button == 1:
                self.cell_positions.setdefault(self.current_idx, []).append((event.xdata, event.ydata))
                self._save_current_mode_data()
                self._update_display()

    def _on_mouse_release(self, event):
        if self.mode == 'threshold' and (self.drawing_active or self.erasing_active) and event.button == 1:
            is_erasing = self.erasing_active
            self.drawing_active = False
            self.erasing_active = False
            if len(self.current_drawing) >= 3:
                self.current_drawing.append(self.current_drawing[0])
                self.region_history.setdefault(self.current_idx, [])
                current_state = {'drawn': [poly[:] for poly in self.drawn_regions.get(self.current_idx, [])],
                                 'erased': [poly[:] for poly in self.erased_regions.get(self.current_idx, [])]}
                self.region_history[self.current_idx].append(current_state)
                if is_erasing:
                    self.erased_regions.setdefault(self.current_idx, []).append(self.current_drawing[:])
                    print(f"Added ERASE region with {len(self.current_drawing)} points")
                else:
                    self.drawn_regions.setdefault(self.current_idx, []).append(self.current_drawing[:])
                    print(f"Added ADD region with {len(self.current_drawing)} points")
                self._save_current_mode_data()
                self._update_display()
            self.current_drawing = []
            if self.drawing_line:
                self.drawing_line.remove()
                self.drawing_line = None
                self.fig.canvas.draw_idle()

    def _on_mouse_move(self, event):
        if self.mode == 'threshold' and (self.drawing_active or self.erasing_active):
            if event.inaxes != self.ax_overlay:
                return
            if event.xdata is None or event.ydata is None:
                return
            self.current_drawing.append((event.xdata, event.ydata))
            if self.drawing_line:
                self.drawing_line.remove()
            xs = [p[0] for p in self.current_drawing]
            ys = [p[1] for p in self.current_drawing]
            color = 'red' if self.erasing_active else 'yellow'
            self.drawing_line, = self.ax_overlay.plot(xs, ys, color=color, linewidth=2, alpha=0.8)
            self.fig.canvas.draw_idle()

    def _on_clear(self, event):
        if self.mode == 'count':
            if self.current_idx in self.cell_positions:
                self.cell_positions[self.current_idx] = []
                self._save_current_mode_data()
                self._update_display()
                print(f"Cleared counts for ROI {self.current_idx + 1}")
        else:
            needs_save = False
            self.region_history.setdefault(self.current_idx, [])
            current_state = {'drawn': [poly[:] for poly in self.drawn_regions.get(self.current_idx, [])],
                             'erased': [poly[:] for poly in self.erased_regions.get(self.current_idx, [])]}
            if self.drawn_regions.get(self.current_idx):
                self.region_history[self.current_idx].append(current_state)
                self.drawn_regions[self.current_idx] = []
                needs_save = True
            if self.erased_regions.get(self.current_idx):
                if not needs_save:
                    self.region_history[self.current_idx].append(current_state)
                self.erased_regions[self.current_idx] = []
                needs_save = True
            if needs_save:
                self._save_current_mode_data()
                self._update_display()
                print(f"Cleared all regions for ROI {self.current_idx + 1}")

    def _undo_last_region(self):
        if self.mode != 'threshold':
            return
        if self.current_idx not in self.region_history or not self.region_history[self.current_idx]:
            print("Nothing to undo")
            return
        previous_state = self.region_history[self.current_idx].pop()
        self.drawn_regions[self.current_idx] = [poly[:] for poly in previous_state['drawn']]
        self.erased_regions[self.current_idx] = [poly[:] for poly in previous_state['erased']]
        print("Undid last region change")
        self._save_current_mode_data()
        self._update_display()

    def _update_display(self):
        self.current_img = apply_brightness_contrast(self.original_img, self.brightness, self.exposure)
        self.current_mask = apply_threshold_inverted(self.current_img, self.threshold)
        if self.mode == 'threshold':
            for polygon in self.drawn_regions.get(self.current_idx, []):
                if len(polygon) >= 3:
                    self.current_mask = np.logical_or(self.current_mask, polygon_to_mask(polygon, self.current_mask.shape))
            for polygon in self.erased_regions.get(self.current_idx, []):
                if len(polygon) >= 3:
                    self.current_mask = np.logical_and(self.current_mask, ~polygon_to_mask(polygon, self.current_mask.shape))

        if self.mode == 'threshold':
            coverage = calculate_coverage(self.current_mask)
            count_value = None
        else:
            coverage = None
            count_value = len(self.cell_positions.get(self.current_idx, []))

        self.ax_adjusted.clear()
        self.ax_overlay.clear()
        self.ax_adjusted.imshow(self.current_img, cmap='gray', vmin=0, vmax=255)
        self.ax_adjusted.set_title("Adjusted", fontsize=13, fontweight='bold', pad=10)
        self.ax_adjusted.axis('off')

        if self.mode == 'threshold':
            overlay = np.stack([self.current_img / 255] * 3, axis=-1)
            overlay[self.current_mask, 0] *= 0.3
            overlay[self.current_mask, 1] = 1.0
            overlay[self.current_mask, 2] = 1.0
            self.ax_overlay.imshow(overlay)
            for polygon in self.drawn_regions.get(self.current_idx, []):
                if len(polygon) >= 3:
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    self.ax_overlay.plot(xs, ys, 'yellow', linewidth=2, alpha=0.9)
            for polygon in self.erased_regions.get(self.current_idx, []):
                if len(polygon) >= 3:
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    self.ax_overlay.plot(xs, ys, 'red', linewidth=2, alpha=0.9, linestyle='--')
            num_drawn = len(self.drawn_regions.get(self.current_idx, []))
            num_erased = len(self.erased_regions.get(self.current_idx, []))
            self.ax_overlay.set_title(f"Coverage: {coverage:.2f}% | Add: {num_drawn} | Erase: {num_erased}", fontsize=13, fontweight='bold', pad=10)
        else:
            self.ax_overlay.imshow(self.current_img, cmap='gray', vmin=0, vmax=255)
            self.cell_artists = []
            for x, y in self.cell_positions.get(self.current_idx, []):
                circle = plt.Circle((x, y), 8, color='red', fill=False, linewidth=2, alpha=0.8)
                self.ax_overlay.add_patch(circle)
                self.cell_artists.append(circle)
                self.ax_overlay.plot(x, y, 'r+', markersize=10, markeredgewidth=2)
            self.ax_overlay.set_title(f"Count: {count_value}", fontsize=13, fontweight='bold', pad=10)
        self.ax_overlay.axis('off')

        roi_name = os.path.basename(self.roi_files[self.current_idx])
        file_ext = get_file_format(self.roi_files[self.current_idx]).upper()
        mode_str = self.mode.upper()
        if self.mode == 'threshold':
            num_drawn = len(self.drawn_regions.get(self.current_idx, []))
            num_erased = len(self.erased_regions.get(self.current_idx, []))
            metric_str = (f"Coverage: {coverage:.2f}%\nSignal px: {int(np.sum(self.current_mask)):,}\nTotal px: {int(self.current_mask.size):,}\nAdd regions: {num_drawn}\nErase regions: {num_erased}")
        else:
            metric_str = (f"Cell count: {count_value}\n(Left-click: add)\n(Right-click: remove)")
        info_text = (f"ROI: {self.current_idx + 1}/{len(self.roi_files)}\nFormat: {file_ext}\nMode: {mode_str}\n\n{roi_name[:26]}\n\nSize: {self.original_img.shape[1]}×{self.original_img.shape[0]}\n\n--- Adjust ---\nBright: {self.brightness:+.0f}\nExp: {self.exposure:.2f}×\n")
        if self.mode == 'threshold':
            info_text += f"\n--- Threshold ---\nValue: {self.threshold:.0f}\n(Dark ≤ thresh)\n(Drag: add)\n(Shift+Drag: erase)\n"
        info_text += f"\n--- Results ---\n{metric_str}\n"
        self.txt_info.set_text(info_text)
        self.fig.canvas.draw_idle()

    def _on_brightness_changed(self, val):
        self.brightness = int(val)
        self._update_display()
        self._save_current_mode_data()

    def _on_exposure_changed(self, val):
        self.exposure = float(val)
        self._update_display()
        self._save_current_mode_data()

    def _on_threshold_changed(self, val):
        self.threshold = int(val)
        if self.mode == 'threshold':
            self._update_display()
            self._save_current_mode_data()

    def _load_current_idx_image(self):
        self.original_img = load_image(self.roi_files[self.current_idx])
        warn_if_not_500x500(self.original_img, self.roi_files[self.current_idx])

    def _on_roi_changed(self, val):
        new_idx = int(val) - 1
        if new_idx != self.current_idx and 0 <= new_idx < len(self.roi_files):
            self._save_current_mode_data()
            self.current_idx = new_idx
            self._load_current_idx_image()
            self._update_display()

    def _on_prev(self, event):
        if self.current_idx > 0:
            self._save_current_mode_data()
            self.current_idx -= 1
            self.slider_roi.set_val(self.current_idx + 1)
            self._load_current_idx_image()
            self._update_display()

    def _on_next(self, event):
        if self.current_idx < len(self.roi_files) - 1:
            self._save_current_mode_data()
            self.current_idx += 1
            self.slider_roi.set_val(self.current_idx + 1)
            self._load_current_idx_image()
            self._update_display()

    def _get_current_state_key(self):
        if self.mode == 'threshold':
            drawn = tuple(tuple(tuple(p) for p in poly) for poly in self.drawn_regions.get(self.current_idx, []))
            erased = tuple(tuple(tuple(p) for p in poly) for poly in self.erased_regions.get(self.current_idx, []))
            return (self.current_idx, self.mode, self.brightness, self.exposure, self.threshold, drawn, erased)
        else:
            coords = tuple(self.cell_positions.get(self.current_idx, []))
            return (self.current_idx, self.mode, self.brightness, self.exposure, coords)

    def _save_current_mode_data(self):
        current_state = self._get_current_state_key()
        if self.current_idx not in self.roi_data:
            self.roi_data[self.current_idx] = {'roi_file': os.path.basename(self.roi_files[self.current_idx]),
                                               'roi_path': self.roi_files[self.current_idx], 'roi_index': self.current_idx,
                                               'width': self.original_img.shape[1], 'height': self.original_img.shape[0],
                                               'file_format': get_file_format(self.roi_files[self.current_idx]), 'adjustments': {}}
        self.roi_data[self.current_idx]['adjustments'] = {'brightness': self.brightness, 'exposure': self.exposure}
        if self.mode == 'threshold':
            coverage = calculate_coverage(self.current_mask)
            drawn_regions = self.drawn_regions.get(self.current_idx, [])
            erased_regions = self.erased_regions.get(self.current_idx, [])
            self.roi_data[self.current_idx]['threshold'] = {'threshold_value': self.threshold, 'coverage_percent': coverage,
                                                            'positive_pixels': int(np.sum(self.current_mask)), 'total_pixels': int(self.current_mask.size),
                                                            'num_drawn_regions': len(drawn_regions), 'num_erased_regions': len(erased_regions),
                                                            'drawn_regions': [poly[:] for poly in drawn_regions], 'erased_regions': [poly[:] for poly in erased_regions]}
        else:
            cell_coords = self.cell_positions.get(self.current_idx, [])
            self.roi_data[self.current_idx]['count'] = {'cell_count': len(cell_coords), 'cell_coordinates': cell_coords.copy() if cell_coords else []}
        self.last_saved_state[(self.current_idx, self.mode)] = current_state

    def _export_csv(self):
        if not self.roi_data:
            print("No results to export.")
            return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(self.output_dir, f"roi_analysis_{timestamp}.csv")
        fieldnames = ['roi_file', 'roi_path', 'roi_index', 'file_format', 'width', 'height', 'brightness', 'exposure',
                      'threshold_value', 'coverage_percent', 'positive_pixels', 'total_pixels', 'num_drawn_regions',
                      'num_erased_regions', 'cell_count', 'has_coordinates']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for roi_idx in sorted(self.roi_data.keys()):
                data = self.roi_data[roi_idx]
                adj = data.get('adjustments', {})
                thresh = data.get('threshold', {})
                count = data.get('count', {})
                row = {'roi_file': data.get('roi_file', ''), 'roi_path': data.get('roi_path', ''), 'roi_index': data.get('roi_index', roi_idx),
                       'file_format': data.get('file_format', ''), 'width': data.get('width', ''), 'height': data.get('height', ''),
                       'brightness': adj.get('brightness', ''), 'exposure': adj.get('exposure', ''), 'threshold_value': thresh.get('threshold_value', ''),
                       'coverage_percent': thresh.get('coverage_percent', ''), 'positive_pixels': thresh.get('positive_pixels', ''),
                       'total_pixels': thresh.get('total_pixels', ''), 'num_drawn_regions': thresh.get('num_drawn_regions', ''),
                       'num_erased_regions': thresh.get('num_erased_regions', ''), 'cell_count': count.get('cell_count', ''),
                       'has_coordinates': 'yes' if (count and count.get('cell_coordinates')) else 'no'}
                writer.writerow(row)
        coords_data = []
        for roi_idx in sorted(self.roi_data.keys()):
            data = self.roi_data[roi_idx]
            count = data.get('count', {})
            coords = count.get('cell_coordinates', []) if count else []
            if coords:
                roi_file = data.get('roi_file', '')
                for i, (x, y) in enumerate(coords, 1):
                    coords_data.append({'roi_index': roi_idx, 'roi_file': roi_file, 'cell_number': i, 'x_coordinate': x, 'y_coordinate': y})
        if coords_data:
            coords_csv_path = os.path.join(self.output_dir, f"roi_cell_coordinates_{timestamp}.csv")
            with open(coords_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['roi_index', 'roi_file', 'cell_number', 'x_coordinate', 'y_coordinate'])
                writer.writeheader()
                writer.writerows(coords_data)
            print(f"✓ Exported cell coordinates to: {coords_csv_path}")
        regions_data = []
        for roi_idx in sorted(self.roi_data.keys()):
            data = self.roi_data[roi_idx]
            thresh = data.get('threshold', {})
            drawn = thresh.get('drawn_regions', []) if thresh else []
            erased = thresh.get('erased_regions', []) if thresh else []
            roi_file = data.get('roi_file', '')
            for region_num, polygon in enumerate(drawn, 1):
                for point_num, (x, y) in enumerate(polygon, 1):
                    regions_data.append({'roi_index': roi_idx, 'roi_file': roi_file, 'region_type': 'add', 'region_number': region_num,
                                         'point_number': point_num, 'x_coordinate': x, 'y_coordinate': y})
            for region_num, polygon in enumerate(erased, 1):
                for point_num, (x, y) in enumerate(polygon, 1):
                    regions_data.append({'roi_index': roi_idx, 'roi_file': roi_file, 'region_type': 'erase', 'region_number': region_num,
                                         'point_number': point_num, 'x_coordinate': x, 'y_coordinate': y})
        if regions_data:
            regions_csv_path = os.path.join(self.output_dir, f"roi_drawn_regions_{timestamp}.csv")
            with open(regions_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['roi_index', 'roi_file', 'region_type', 'region_number', 'point_number', 'x_coordinate', 'y_coordinate'])
                writer.writeheader()
                writer.writerows(regions_data)
            print(f"✓ Exported drawn regions (add + erase) to: {regions_csv_path}")
        print(f"\n{'=' * 70}")
        print(f"✓ Exported {len(self.roi_data)} ROI analyses to:")
        print(f"  {csv_path}")
        format_counts = {}
        for data in self.roi_data.values():
            fmt = data.get('file_format', 'unknown')
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        print(f"  File formats: {', '.join(f'{fmt}({count})' for fmt, count in sorted(format_counts.items()))}")
        print(f"  Total ROIs with threshold data: {sum(1 for d in self.roi_data.values() if d.get('threshold'))}")
        print(f"  Total ROIs with count data: {sum(1 for d in self.roi_data.values() if d.get('count'))}")
        print(f"{'=' * 70}\n")

    def _on_close(self, event):
        if not self.auto_saved:
            print("\nWindow closing - auto-saving all ROIs...")
            self._save_current_mode_data()
            self._export_csv()
            self.auto_saved = True

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key in ('left', 'a'):
            self._on_prev(None)
        elif key in ('right', 'd'):
            self._on_next(None)
        elif key.lower() in ('q', 'escape'):
            plt.close(self.fig)
        elif key == 'c':
            self._on_clear(None)
        elif 'ctrl+z' in key or 'control+z' in key or (key == 'z' and event.key and 'ctrl' in event.key.lower()):
            self._undo_last_region()

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Interactive ROI analyzer with threshold (coverage + drawing + erasing) and count modes - supports TIFF, NPY, and JPEG ROI crops")
    parser.add_argument('directory', nargs='?', default='', help='Directory containing ROI files (TIFF/NPY/JPEG). If omitted, opens file browser.')
    parser.add_argument('--output', '-o', default='', help='Output directory for CSV results (default: same as ROI directory)')
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    roi_dir = sanitize_path(args.directory)
    if not roi_dir:
        print("Opening file browser to select ROI directory...")
        roi_dir = prompt_for_directory()
    if not roi_dir:
        print("No directory selected. Exiting.")
        return 1
    if not os.path.isdir(roi_dir):
        print(f"Directory not found: {roi_dir}")
        return 2
    image_files = find_image_files(roi_dir)
    if not image_files:
        print(f"No ROI files found in: {roi_dir}")
        return 3
    tiff_count = sum(1 for f in image_files if f.lower().endswith(('.tif', '.tiff')))
    npy_count = sum(1 for f in image_files if f.lower().endswith('.npy'))
    tif_npy_count = sum(1 for f in image_files if f.lower().endswith(('.tif.npy', '.tiff.npy')))
    jpg_count = sum(1 for f in image_files if f.lower().endswith(('.jpg', '.jpeg')))
    print(f"\n{'=' * 70}")
    print(f"Found {len(image_files)} ROI files in:")
    print(f"  {roi_dir}")
    print(f"  TIFF: {tiff_count}, NPY: {npy_count}, TIF.NPY: {tif_npy_count}, JPEG: {jpg_count}")
    print(f"  NOTE: Soft check expects ROI size {ROI_EXPECTED_SIZE[0]}x{ROI_EXPECTED_SIZE[1]} for crops.")
    print(f"{'=' * 70}\n")
    output_dir = sanitize_path(args.output) if args.output else roi_dir
    os.makedirs(output_dir, exist_ok=True)
    analyzer = ROIAnalyzer(image_files, output_dir)
    print("ROI Analyzer - Dual Mode with Drawing + Erasing + JPEG ROI support:")
    print("  • File formats: .jpg, .jpeg, .tif, .tiff, .npy, .tif.npy, .tiff.npy")
    print("  • Soft check: warns if an ROI is not 500x500")
    print("  • THRESHOLD mode:")
    print("    - Automatic coverage (dark ≤ threshold)")
    print("    - DRAG on overlay to draw additional regions (ADD - yellow)")
    print("    - SHIFT+DRAG on overlay to erase regions (ERASE - red dashed)")
    print("    - Ctrl+Z to undo last drawn/erased region")
    print("  • COUNT mode:")
    print("    - Left-click to add cell marker")
    print("    - Right-click to remove nearest marker")
    print("\n  • Navigation: Left/Right arrows or A/D")
    print("  • Clear: C key or button (clears regions or counts depending on mode)")
    print("  • Close: Q/Escape (exports all data to CSV)\n")
    try:
        while plt.fignum_exists(analyzer.fig.number):
            plt.pause(0.1)
    except KeyboardInterrupt:
        pass
    return 0

if __name__ == "__main__":
    sys.exit(main())