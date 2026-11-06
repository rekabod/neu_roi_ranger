import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # For image_utils
from image_utils import to_8bit, apply_clahe

ROI_SIZE = 500

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("JPEG ROI Cropper (500x500)")

        self.img_path = None
        self.pil_img = None
        self.img_np = None
        self.rois = []
        self.roi_patches = []

        # UI
        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X)
        tk.Button(top, text="Open JPEG...", command=self.open_image).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(top, text="Save ROIs...", command=self.save_rois).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(top, text="Clear ROIs", command=self.clear_rois).pack(side=tk.LEFT, padx=5, pady=5)

        self.status = tk.StringVar(value="Open a JPEG to start. Left-click add ROI, right-click remove ROI.")
        tk.Label(root, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("JPEG files", "*.jpg *.jpeg"), ("All files", "*.*")])
        if not path:
            return
        try:
            pil = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}")
            return

        self.img_path = path
        self.pil_img = pil
        self.img_np = np.array(pil)
        self.clear_rois(redraw=False)
        self.redraw_image()
        w, h = pil.size
        self.status.set(f"Loaded: {os.path.basename(path)} ({w}x{h}). ROIs: 0")

    def clamp_roi_centered(self, cx, cy, w, h, size):
        half = size // 2
        x0 = int(cx - half)
        y0 = int(cy - half)
        x0 = max(0, min(x0, w - size))
        y0 = max(0, min(y0, h - size))
        return (x0, y0, x0 + size, y0 + size)

    def find_roi_at(self, x, y):
        for i in range(len(self.rois) - 1, -1, -1):
            x0, y0, x1, y1 = self.rois[i]
            if x0 <= x < x1 and y0 <= y < y1:
                return i
        return None

    def on_click(self, event):
        if self.pil_img is None or event.inaxes != self.ax:
            return
        x, y = float(event.xdata), float(event.ydata)
        w, h = self.pil_img.size

        if event.button == 1:  # add
            roi = self.clamp_roi_centered(x, y, w, h, ROI_SIZE)
            self.rois.append(roi)
            self.add_patch(roi)
            self.canvas.draw_idle()
            self.status.set(f"ROIs: {len(self.rois)} (added at x={int(x)}, y={int(y)})")

        elif event.button == 3:  # remove
            idx = self.find_roi_at(x, y)
            if idx is None:
                self.status.set(f"ROIs: {len(self.rois)} (no ROI under cursor)")
                return
            patch = self.roi_patches.pop(idx)
            patch.remove()
            self.rois.pop(idx)
            self.canvas.draw_idle()
            self.status.set(f"ROIs: {len(self.rois)} (removed ROI #{idx + 1})")

    def redraw_image(self):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.imshow(self.img_np, origin="upper")
        self.roi_patches = []
        for roi in self.rois:
            self.add_patch(roi, draw=False)
        self.canvas.draw_idle()

    def add_patch(self, roi, draw=True):
        x0, y0, x1, y1 = roi
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="red", facecolor="none")
        self.ax.add_patch(rect)
        self.roi_patches.append(rect)
        if draw:
            self.canvas.draw_idle()

    def clear_rois(self, redraw=True):
        self.rois = []
        for p in getattr(self, "roi_patches", []):
            try:
                p.remove()
            except:
                pass
        self.roi_patches = []
        if redraw and self.pil_img is not None:
            self.redraw_image()
        if self.pil_img is not None:
            self.status.set("ROIs: 0 (cleared)")

    def save_rois(self):
        if self.pil_img is None:
            messagebox.showinfo("Save ROIs", "Open an image first.")
            return
        if not self.rois:
            messagebox.showinfo("Save ROIs", "No ROIs to save.")
            return

        out_dir = filedialog.askdirectory(title="Choose output folder for ROI JPEGs")
        if not out_dir:
            return

        base = os.path.splitext(os.path.basename(self.img_path))[0]
        saved = 0
        for i, (x0, y0, x1, y1) in enumerate(self.rois, start=1):
            crop = self.pil_img.crop((x0, y0, x1, y1))
            if crop.mode not in ("L", "RGB"):
                crop = crop.convert("L")
            out_path = os.path.join(out_dir, f"{base}_roi_{i:03d}_{x0}_{y0}.jpg")
            crop.save(out_path, format="JPEG", quality=95)
            saved += 1

        messagebox.showinfo("Save ROIs", f"Saved {saved} ROI(s) to:\n{out_dir}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()