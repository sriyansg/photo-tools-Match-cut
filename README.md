# Match-cut Circle Aligner

A desktop Python application that uses a reference image's target circle to automatically and interactively align all subsequent images. This is designed for creating perfectly matched cut transitions between images sharing a common subject (e.g. circles, faces) using bounded transformations (translation, rotation, and uniform zoom). 

## Setup
1. Ensure Python 3.9+ is installed.
2. Install requirements:
   `pip install -r requirements.txt`
3. Place initial images in the `input/` directory. If there's an explicit reference image, place it there and set that image path in `config.json` parameter `reference_image_path`. Otherwise, the first image retrieved in alphabetical order will be used as the reference image.
a
## Usage
1. Run `python align_matchcut_circles.py`.
2. **First Step:** The reference image will pop up. Click the center of the target subject/circle and drag outwards to set its radius. Press `Enter` to confirm.
3. **Review loop:**
   - The tool will compute the transformation to bring each image's target subject/circle to the exact location and scale as the reference image's subject.
   - You must review every image before saving. Use the keyboard shorts to accept, skip, modify, or recalculate:
     - `Enter` or `A`: Accept and save aligned version.
     - `S`: Skip the image.
     - `Q`: Quit the program.
   - **Manual Adjustments:**
     - `Up/Down/Left/Right` or `W/A/S/D`: Nudge small translations.
     - `I/J/K/L`: Nudge larger translations.
     - `+` or `=`: Zoom in uniformly.
     - `-` or `_`: Zoom out uniformly.
     - `[` or `,`: Rotate left.
     - `]` or `.`: Rotate right.
     - `R`: Reset manual adjustments to automatically detected alignments.
     - `D`: Toggle debug overlays on the output view.
     - `N`: Cycle through possible circle detection candidates if the primary one is incorrect.
     - `F`: Expand the Region of Interest (ROI) when the system can't see the circle.
     - `G`: Do a full-frame search for the target.

## Allowed Transformations
To prevent distortions (stretch, shear, non-uniform scaling, perspective warping), the application strictly restricts operations to geometric similarities: uniform scale, translation, and rotation.

## Output
Review outcomes direct images seamlessly to respective sections in the `output/` folder (`aligned`, `debug`, `skipped`). Metadata concerning transformations performed are cataloged sequentially inside `output/logs/run_log.txt`.
