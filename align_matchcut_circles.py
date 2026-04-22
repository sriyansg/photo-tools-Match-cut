import cv2
import numpy as np
import json
import os
import glob
import math
from datetime import datetime

class MatchCutAligner:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_directories()
        
        self.ref_center = None
        self.ref_radius = None

    def load_config(self):
        if not os.path.exists(self.config_path):
            print(f"Error: Could not find config file {self.config_path}.")
            exit(1)
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def setup_directories(self):
        dirs = [
            self.config.get('input_folder', 'input'),
            os.path.join(self.config.get('output_folder', 'output'), 'aligned'),
            os.path.join(self.config.get('output_folder', 'output'), 'debug'),
            os.path.join(self.config.get('output_folder', 'output'), 'skipped'),
            os.path.join(self.config.get('output_folder', 'output'), 'logs')
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def get_log_file(self):
        log_dir = os.path.join(self.config.get('output_folder', 'output'), 'logs')
        return os.path.join(log_dir, 'run_log.txt')

    def log_entry(self, entry_dict):
        with open(self.get_log_file(), 'a') as f:
            entry_dict['timestamp'] = datetime.now().isoformat()
            f.write(json.dumps(entry_dict) + "\n")

    def run(self):
        input_folder = self.config.get('input_folder', 'input')
        image_exts = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
        image_paths = []
        for ext in image_exts:
            image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        
        if not image_paths:
            print(f"No images found in {input_folder}")
            return
            
        # Deduplicate and sort (glob over multiple exts on Windows captures duplicates)
        image_paths = sorted(list(set(image_paths)))
            
        ref_path = self.config.get('reference_image_path', '')
        if not ref_path or not os.path.exists(ref_path):
            print("Please specify a valid reference_image_path in config (or place one and set it). Using first image as reference.")
            ref_path = image_paths.pop(0)
        else:
            if ref_path in image_paths:
                image_paths.remove(ref_path)

        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            print(f"Failed to load reference image: {ref_path}")
            return
            
        print(f"Using reference image: {ref_path}")
        self.mark_reference_circle(ref_img)
        print(f"Reference circle set: Center={self.ref_center}, Radius={self.ref_radius}")
        
        for img_path in image_paths:
            self.process_and_review(img_path)

    def mark_reference_circle(self, img):
        self.ref_center = None
        self.ref_radius = None
        drawing = False
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                self.ref_center = (x, y)
                self.ref_radius = 0
                drawing = True
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    self.ref_radius = int(math.hypot(x - self.ref_center[0], y - self.ref_center[1]))
            elif event == cv2.EVENT_LBUTTONUP:
                if drawing:
                    self.ref_radius = int(math.hypot(x - self.ref_center[0], y - self.ref_center[1]))
                    drawing = False
                    
        cv2.namedWindow('Reference Image - Mark Target Circle', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Reference Image - Mark Target Circle', mouse_callback)
        
        while True:
            display_img = img.copy()
            if self.ref_center and self.ref_radius is not None and self.ref_radius > 0:
                cv2.circle(display_img, self.ref_center, self.ref_radius, (0, 255, 0), 2)
                cv2.drawMarker(display_img, self.ref_center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            
            cv2.putText(display_img, "1. Click to set Center. 2. Drag to Set Radius. 3. Press Enter to Confirm.", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            
            cv2.imshow('Reference Image - Mark Target Circle', display_img)
            key = cv2.waitKey(20) & 0xFF
            if key == 13 or key == ord('a') or key == ord('A'): 
                if self.ref_center and self.ref_radius > 0:
                    break
            elif key == 27 or key == ord('q'): 
                print("Quitting tool.")
                exit(0)
        cv2.destroyWindow('Reference Image - Mark Target Circle')

    def get_roi_bounds(self, img_shape, x, y, r, margin_mult):
        h, w = img_shape[:2]
        roi_r = int(r * margin_mult)
        x1 = max(0, int(x - roi_r))
        y1 = max(0, int(y - roi_r))
        x2 = min(w, int(x + roi_r))
        y2 = min(h, int(y + roi_r))
        return x1, y1, x2, y2

    def find_candidates(self, img, x, y, r, margin_mult):
        x1, y1, x2, y2 = self.get_roi_bounds(img.shape, x, y, r, margin_mult)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        candidates = []
        
        min_r_ratio = self.config.get('min_radius_ratio', 0.8)
        max_r_ratio = self.config.get('max_radius_ratio', 1.2)
        expected_min_r = max(1, int(r * min_r_ratio))
        expected_max_r = int(r * max_r_ratio)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=self.config.get('hough_dp', 1.2),
            minDist=self.config.get('hough_minDist', 50),
            param1=self.config.get('hough_param1', 50),
            param2=self.config.get('hough_param2', 30),
            minRadius=expected_min_r,
            maxRadius=expected_max_r
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, radius = i[0], i[1], i[2]
                candidates.append({
                    'center': (int(cx + x1), int(cy + y1)),
                    'radius': int(radius),
                    'method': 'hough'
                })
                
        if not candidates:
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 5:
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    if expected_min_r <= radius <= expected_max_r:
                        candidates.append({
                            'center': (int(cx + x1), int(cy + y1)),
                            'radius': int(radius),
                            'method': 'contour'
                        })
        
        candidates.sort(key=lambda c: abs(c['radius'] - self.ref_radius))
        return candidates

    def process_and_review(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading {img_path}")
            return
            
        print(f"\nProcessing {img_path}...")

        margin = self.config.get('search_margin_multiplier', 2.0)
        expand_factor = self.config.get('roi_expand_factor', 1.5)
        
        step = 0
        current_margin = margin
        candidates = self.find_candidates(img, self.ref_center[0], self.ref_center[1], self.ref_radius, current_margin)
        candidate_idx = 0
        
        manual_tx, manual_ty = 0.0, 0.0
        manual_scale = 1.0
        manual_rot = 0.0
        show_debug = True

        cv2.namedWindow('Review', cv2.WINDOW_AUTOSIZE)

        while True:
            current_candidate = candidates[candidate_idx] if candidates and candidate_idx < len(candidates) else None
            
            if current_candidate:
                cx, cy = current_candidate['center']
                cr = current_candidate['radius']
            else:
                cx, cy = self.ref_center  
                cr = self.ref_radius if self.ref_radius > 0 else 1
                
            base_s = self.ref_radius / float(cr) if self.config.get('allow_scale') else 1.0
            final_s = base_s * manual_scale if self.config.get('allow_scale') else 1.0
            final_rot = manual_rot if self.config.get('allow_rotation') else 0.0
            
            M = cv2.getRotationMatrix2D((cx, cy), final_rot, final_s)
            
            tx = (self.ref_center[0] - cx) if self.config.get('allow_translation') else 0
            ty = (self.ref_center[1] - cy) if self.config.get('allow_translation') else 0
            
            final_tx = tx + manual_tx
            final_ty = ty + manual_ty
            
            M[0, 2] += final_tx
            M[1, 2] += final_ty
            
            h, w = img.shape[:2]
            border_str = self.config.get('border_mode', 'constant').lower()
            border_mode = cv2.BORDER_REPLICATE if border_str == 'replicate' else cv2.BORDER_CONSTANT
            
            res = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0,0,0))
            
            display_res = res.copy()
            orig_display = img.copy()
            
            if show_debug:
                cv2.circle(display_res, self.ref_center, self.ref_radius, (0, 255, 0), 2)
                cv2.drawMarker(display_res, self.ref_center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                
                if current_candidate:
                    cv2.circle(orig_display, (int(cx), int(cy)), int(cr), (0, 0, 255), 2)
                    cv2.drawMarker(orig_display, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
                
                x1, y1, x2, y2 = self.get_roi_bounds(img.shape, self.ref_center[0], self.ref_center[1], self.ref_radius, current_margin)
                cv2.rectangle(orig_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
            # Draw original as a Picture-in-Picture in the top right
            pip_w, pip_h = int(w * 0.25), int(h * 0.25)
            if pip_w > 0 and pip_h > 0:
                pip = cv2.resize(orig_display, (pip_w, pip_h))
                display_res[0:pip_h, w-pip_w:w] = pip
                cv2.rectangle(display_res, (w-pip_w, 0), (w, pip_h), (255, 255, 255), 2)

            import ctypes
            try:
                user32 = ctypes.windll.user32
                screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            except:
                screen_w, screen_h = 1920, 1080
                
            # Leave 5% gap from borders -> size is 90%
            max_w = int(screen_w * 0.90)
            max_h = int(screen_h * 0.90)
            
            sc = min(max_w / float(w), max_h / float(h))
            fit_w = int(max(1, w * sc))
            fit_h = int(max(1, h * sc))
            
            combined = cv2.resize(display_res, (fit_w, fit_h))

            # Increase font scale and draw UI relative to final window space
            font_scale = 1.0
            font_thick = 2
            
            text_lines_res = [
                f"TRANSFORMED PREVIEW",
                f"Candidate: {'YES' if current_candidate else 'NO'} ({candidate_idx+1}/{len(candidates)})",
                f"Zoom: {final_s:.3f}x (manual: {manual_scale:.3f})",
                f"Rot: {final_rot:.2f} deg",
                f"Trans: x={int(final_tx)}, y={int(final_ty)}"
            ]
            for i, line in enumerate(text_lines_res):
                cv2.putText(combined, line, (20, 40 + i*40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick + 2)
                cv2.putText(combined, line, (20, 40 + i*40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thick)
                
            text_controls = "Controls: Enter/A(Acc) S(Skip) Q(Quit) Arrows(Mov) IJKL(Mov-L) +/-(Zm) [/](Rot) R(Reset) D(Debug) N(Cycle) F(Expand) G(Full)"
            cv2.putText(combined, text_controls, (20, fit_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(combined, text_controls, (20, fit_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Review', combined)
            
            key = cv2.waitKeyEx(0)
            char = key & 0xFF
            
            n_sm = self.config.get('nudge_pixels_small', 1)
            n_lg = self.config.get('nudge_pixels_large', 10)
            r_deg = self.config.get('rotation_step_degrees', 0.5)
            s_step = self.config.get('scale_step', 0.01)

            if char == 13 or char == ord('a') or char == ord('A'):
                self.save_result(img_path, res, cx, cy, cr, final_tx, final_ty, final_s, final_rot, orig_display, display_res)
                break
            elif char == ord('s') or char == ord('S'):
                self.log_skip(img_path)
                break
            elif char == 27 or char == ord('q') or char == ord('Q'):
                print("Quitting tool.")
                exit(0)
            elif char == ord('r') or char == ord('R'):
                manual_tx, manual_ty, manual_scale, manual_rot = 0, 0, 1.0, 0.0
            elif char == ord('d') or char == ord('D'):
                show_debug = not show_debug
            elif char == ord('n') or char == ord('N'):
                if candidates:
                    candidate_idx = (candidate_idx + 1) % len(candidates)
            elif char == ord('f') or char == ord('F'):
                step += 1
                current_margin = margin * (expand_factor ** step)
                candidates = self.find_candidates(img, self.ref_center[0], self.ref_center[1], self.ref_radius, current_margin)
                candidate_idx = 0
            elif char == ord('g') or char == ord('G'):
                current_margin = 100.0
                candidates = self.find_candidates(img, self.ref_center[0], self.ref_center[1], self.ref_radius, current_margin)
                candidate_idx = 0
            elif char == ord('=') or char == ord('+'):
                manual_scale += s_step
            elif char == ord('-') or char == ord('_'):
                manual_scale -= s_step
            elif char == ord('[') or char == ord(','):
                manual_rot -= r_deg
            elif char == ord(']') or char == ord('.'):
                manual_rot += r_deg
            
            # Windows arrow keys via waitKeyEx
            elif key == 2490368 or char == ord('w'): # Up
                manual_ty -= n_sm
            elif key == 2621440 or char == ord('s'): # Down
                manual_ty += n_sm
            elif key == 2424832 or char == ord('a'): # Left
                manual_tx -= n_sm
            elif key == 2555904 or char == ord('d'): # Right
                manual_tx += n_sm
                
            # Large nudges
            elif char == ord('i') or char == ord('I'): # Up Large
                manual_ty -= n_lg
            elif char == ord('k') or char == ord('K'): # Down Large
                manual_ty += n_lg
            elif char == ord('j') or char == ord('J'): # Left Large
                manual_tx -= n_lg
            elif char == ord('l') or char == ord('L'): # Right Large
                manual_tx += n_lg

        cv2.destroyWindow('Review')

    def save_result(self, img_path, res_img, cx, cy, cr, tx, ty, scale, rot, orig_display, display_res):
        out_aligned = os.path.join(self.config.get('output_folder', 'output'), 'aligned')
        out_debug = os.path.join(self.config.get('output_folder', 'output'), 'debug')
        base_name = os.path.basename(img_path)
        
        cv2.imwrite(os.path.join(out_aligned, base_name), res_img)
        
        if self.config.get('save_debug_preview', True):
            comb = np.hstack((orig_display, display_res))
            cv2.imwrite(os.path.join(out_debug, base_name), comb)
            
        self.log_entry({
            "action": "saved",
            "file": img_path,
            "ref_center": self.ref_center,
            "ref_radius": self.ref_radius,
            "detected_center": (int(cx), int(cy)),
            "detected_radius": int(cr),
            "final_tx": float(tx),
            "final_ty": float(ty),
            "final_scale": float(scale),
            "final_rot": float(rot)
        })
        print(f"Saved: {base_name}")

    def log_skip(self, img_path):
        out_skip = os.path.join(self.config.get('output_folder', 'output'), 'skipped')
        base_name = os.path.basename(img_path)
        
        # Optionally move original or just touch a file. The user didn't ask to move, just logically skipped.
        # We'll just drop a dummy file or simply log it. Let's just create an empty .txt to mark skip
        with open(os.path.join(out_skip, base_name + ".skipped.txt"), 'w') as f:
            f.write("Skipped by user")
            
        self.log_entry({
            "action": "skipped",
            "file": img_path
        })
        print(f"Skipped: {base_name}")

if __name__ == "__main__":
    aligner = MatchCutAligner()
    aligner.run()
