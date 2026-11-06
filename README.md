# NeuROIranger Toolbox

A modular Python toolkit for microscopy image processing, designed for neuroscience workflows. Includes tools for cropping ROIs, interactive annotation, and ROI analysis with support for TIFF, NPY, and JPEG formats.

## Features

- **ROI Cropper**: Select and save 500x500 ROIs from JPEG images interactively.
- **Interactive Annotator**: Add/remove point annotations on images (TIFF/NPY), with undo/redo and CSV/overlay export.
- **ROI Analyzer**: Analyze ROIs in threshold (coverage/drawing) or count modes, with CSV export for metrics.
- **Shared Utilities**: Common image loading, CLAHE enhancement, and normalization across tools.

## Architecture

```
neuro_image_toolbox/
├── image_utils.py          # Shared image processing functions
├── roi_cropper/            # JPEG ROI cropping tool
├── annotator/              # Point annotation tool
├── analyzer/               # ROI analysis tool
├── README.md               # This file
├── requirements.txt        # Dependencies
├── .gitignore              # Git ignore rules
└── LICENSE                 # MIT License
```

Each tool is self-contained with a `main.py` entry point.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rekabod/neuro_image_toolbox.git
   cd neuro_image_toolbox
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run each tool separately:

- **ROI Cropper**:
  ```
  python roi_cropper/main.py
  ```
  Open a JPEG, click to add/remove ROIs, save crops.

- **Interactive Annotator**:
  ```
  python annotator/main.py path/to/image.tif
  ```
  Left-click to annotate, right-click to remove, Ctrl+Z/Y for undo/redo, Q to save and exit.

- **ROI Analyzer**:
  ```
  python analyzer/main.py /path/to/roi/directory
  ```
  Switch between threshold (drag to add/erase regions) and count (click cells) modes, export CSVs.

For details, see each tool's `main.py` docstrings.

## Requirements

- Python 3.8+
- Libraries: See `requirements.txt`
- Optional: Tkinter (for file dialogs, usually included with Python)

## Development

- **Adding Tools**: Create a new folder (e.g., `new_tool/`), add `main.py` and `utils.py`, import from `image_utils`.
- **Testing**: Add unit tests in a `tests/` folder (future enhancement).
- **Contributing**: Fork, make changes, submit PRs.

## License

MIT License - see LICENSE file.

## Contributing

Contributions welcome! Open issues or PRs for bugs/features.
