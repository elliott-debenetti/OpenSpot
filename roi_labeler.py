import cv2
import numpy as np
import json
import os
import glob
from pathlib import Path


class TemplateBasedLabeler:
    """
    Fast occupancy labeling using a pre-defined ROI template.
    Only labels occupied/empty - ROI shapes are fixed from template.
    """
    def __init__(self, image_dir, template_file, output_file='annotations.json'):
        self.image_dir = Path(image_dir)
        self.template_file = template_file
        self.output_file = output_file
        
        # Load template
        self.load_template()
        
        # Load all images
        self.image_files = sorted(glob.glob(str(self.image_dir / "*.jpg")) + 
                                  glob.glob(str(self.image_dir / "*.JPG")) +
                                  glob.glob(str(self.image_dir / "*.png")))
        
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images")
        print(f"Template has {len(self.template_rois)} ROIs")
        
        # Current state
        self.current_idx = 0
        self.img = None
        self.display_img = None
        self.img_height = 0
        self.img_width = 0
        
        # Annotations for all images
        self.annotations = {
            'file_names': [],
            'rois_list': [],
            'occupancy_list': []
        }
        
        # Current image annotations (copy of template for each image)
        self.current_occupancy = []
        
        # UI state
        self.selected_roi = None
        self.hover_roi = None
        self.show_help = False
        self.show_labels = True
        
        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.pan_mode = False
        self.is_panning = False
        self.pan_start = None
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        
        self.window_name = 'Template-Based Labeler (Fast Mode)'
        
        # Load existing annotations if they exist
        self.load_annotations()
        
        # Load first image
        self.load_image(0)
    
    def load_template(self):
        """Load ROI template from file."""
        with open(self.template_file, 'r') as f:
            template_data = json.load(f)
        
        self.template_rois = template_data['rois']
        print(f"✓ Loaded template: {len(self.template_rois)} ROIs")
    
    def load_annotations(self):
        """Load existing annotations from file."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                
                # Support both old split format and new flat format
                if 'file_names' in data:
                    # New flat format
                    self.annotations = data
                    print(f"Loaded {len(self.annotations['file_names'])} existing annotations")
                elif 'train' in data or 'valid' in data or 'test' in data:
                    # Old split format - merge all splits
                    print("⚠️  Detected old split-based format. Converting to flat format...")
                    merged_annotations = {
                        'file_names': [],
                        'rois_list': [],
                        'occupancy_list': []
                    }
                    for split_name in ['train', 'valid', 'test']:
                        if split_name in data:
                            split_data = data[split_name]
                            merged_annotations['file_names'].extend(split_data.get('file_names', []))
                            merged_annotations['rois_list'].extend(split_data.get('rois_list', []))
                            merged_annotations['occupancy_list'].extend(split_data.get('occupancy_list', []))
                    self.annotations = merged_annotations
                    print(f"Loaded and merged {len(self.annotations['file_names'])} annotations from old format")
            except Exception as e:
                print(f"Could not load annotations: {e}")
    
    
    def save_annotations(self):
        """Save annotations to file."""
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print(f"✓ Saved: {len(self.annotations['file_names'])} images")
    
    
    def load_image(self, idx):
        """Load image at given index."""
        if 0 <= idx < len(self.image_files):
            self.current_idx = idx
            self.img = cv2.imread(self.image_files[idx])
            self.img_height, self.img_width = self.img.shape[:2]
            
            # Load or initialize occupancy for this image
            fname = os.path.basename(self.image_files[idx])
            if fname in self.annotations['file_names']:
                file_idx = self.annotations['file_names'].index(fname)
                saved_occupancy = self.annotations['occupancy_list'][file_idx].copy()
                
                # Handle mismatch between saved ROIs and current template
                if len(saved_occupancy) != len(self.template_rois):
                    print(f"⚠️  Warning: Saved annotation has {len(saved_occupancy)} ROIs but template has {len(self.template_rois)} ROIs")
                    
                    # Pad with False if template has more ROIs now
                    if len(saved_occupancy) < len(self.template_rois):
                        saved_occupancy.extend([False] * (len(self.template_rois) - len(saved_occupancy)))
                        print(f"   Added {len(self.template_rois) - len(saved_occupancy)} empty ROIs")
                    # Truncate if template has fewer ROIs now
                    else:
                        saved_occupancy = saved_occupancy[:len(self.template_rois)]
                        print(f"   Removed extra ROIs (kept first {len(self.template_rois)})")
                
                self.current_occupancy = saved_occupancy
            else:
                # Initialize all as empty
                self.current_occupancy = [False] * len(self.template_rois)
            
            self.selected_roi = None
            self.draw()
    
    def normalized_to_image(self, point):
        """Convert normalized coordinates (0-1) to image coordinates with zoom/pan."""
        x = int(point[0] * self.img_width)
        y = int(point[1] * self.img_height)
        return self.apply_zoom_pan([x, y])
    
    def apply_zoom_pan(self, point):
        """Apply zoom and pan transformation to a point."""
        x, y = point
        # Apply zoom
        x = int(x * self.zoom_level)
        y = int(y * self.zoom_level)
        # Apply pan
        x += self.pan_offset[0]
        y += self.pan_offset[1]
        return [x, y]
    
    def screen_to_image(self, x, y):
        """Convert screen coordinates back to image coordinates (inverse of zoom/pan)."""
        # Remove pan offset
        img_x = x - self.pan_offset[0]
        img_y = y - self.pan_offset[1]
        # Remove zoom
        img_x = int(img_x / self.zoom_level)
        img_y = int(img_y / self.zoom_level)
        return img_x, img_y
    
    def draw(self):
        """Draw the current state with zoom and pan support."""
        # Create zoomed image
        zoomed_width = int(self.img_width * self.zoom_level)
        zoomed_height = int(self.img_height * self.zoom_level)
        zoomed_img = cv2.resize(self.img, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)
        
        # Calculate canvas size based on window size (use original image size as viewport)
        canvas_width = self.img_width
        canvas_height = self.img_height
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Calculate where to place zoomed image on canvas
        # pan_offset can be positive or negative
        x_offset = self.pan_offset[0]
        y_offset = self.pan_offset[1]
        
        # Determine the region of the zoomed image that's visible on the canvas
        # Canvas destination coordinates
        dst_x_start = max(0, x_offset)
        dst_y_start = max(0, y_offset)
        dst_x_end = min(canvas_width, x_offset + zoomed_width)
        dst_y_end = min(canvas_height, y_offset + zoomed_height)
        
        # Source region in zoomed image
        src_x_start = max(0, -x_offset)
        src_y_start = max(0, -y_offset)
        src_x_end = src_x_start + (dst_x_end - dst_x_start)
        src_y_end = src_y_start + (dst_y_end - dst_y_start)
        
        # Only copy if there's a valid region to display
        if dst_x_end > dst_x_start and dst_y_end > dst_y_start:
            canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                zoomed_img[src_y_start:src_y_end, src_x_start:src_x_end]
        
        self.display_img = canvas.copy()
        
        # Draw all ROIs from template
        for i, roi in enumerate(self.template_rois):
            # Convert to image coordinates with zoom/pan
            pts = [self.normalized_to_image(p) for p in roi]
            pts = np.array(pts, dtype=np.int32)
            
            # Color based on occupancy and selection
            if i == self.selected_roi:
                color = (255, 255, 0)  # Yellow for selected
                thickness = max(2, int(4 * self.zoom_level))
                alpha = 0.1
            elif i == self.hover_roi:
                color = (255, 165, 0)  # Orange for hover
                thickness = max(1, int(1 * self.zoom_level))
                alpha = 0.3
            elif self.current_occupancy[i]:
                color = (0, 0, 255)  # Red for occupied
                thickness = max(1, int(1 * self.zoom_level))
                alpha = 0.15
            else:
                color = (0, 255, 0)  # Green for empty
                thickness = max(1, int(1 * self.zoom_level))
                alpha = 0.10
            
            # Draw filled polygon
            overlay = self.display_img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, alpha, self.display_img, 1-alpha, 0, self.display_img)
            
            # Draw outline
            cv2.polylines(self.display_img, [pts], True, color, thickness)
            
            # Draw label
            if self.show_labels:
                center = np.mean(pts, axis=0).astype(int)
                label = "OCC" if self.current_occupancy[i] else "EMP"
                
                # Scale text with zoom
                font_scale = 0.6 * min(2.0, self.zoom_level)
                text_thickness = max(1, int(2 * min(2.0, self.zoom_level)))
                
                # Background for text
                text_size = cv2.getTextSize(f"{i+1}:{label}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
                padding = max(3, int(3 * self.zoom_level))
                cv2.rectangle(self.display_img, 
                            (center[0] - text_size[0]//2 - padding, center[1] - text_size[1]//2 - padding),
                            (center[0] + text_size[0]//2 + padding, center[1] + text_size[1]//2 + padding),
                            (0, 0, 0), -1)
                
                cv2.putText(self.display_img, f"{i+1}:{label}", 
                           (center[0] - text_size[0]//2, center[1] + text_size[1]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
        
        # Crop to window size
        self.display_img = self.display_img[:self.img_height, :self.img_width]
        
        # Draw instructions
        self.draw_instructions()
        
        cv2.imshow(self.window_name, self.display_img)
    
    def draw_instructions(self):
        """Draw instruction text overlay."""
        if self.show_help:
            self.draw_help_overlay()
            return
        
        fname = os.path.basename(self.image_files[self.current_idx])
        occupied_count = sum(self.current_occupancy)
        empty_count = len(self.current_occupancy) - occupied_count
        
        pan_status = " [PAN MODE]" if self.pan_mode else ""
        
        instructions = [
            f"Image: {self.current_idx+1}/{len(self.image_files)} - {fname}",
            f"Spots: {len(self.current_occupancy)} | Occupied: {occupied_count} | Empty: {empty_count} | Zoom: {self.zoom_level:.1f}x{pan_status}",
            "",
            "FAST MODE: Click spot → 1 (occupied) or 2 (empty) | O: All occupied | E: All empty | C: Copy previous",
            "Zoom: +/- or Scroll | P: Pan mode | L: Toggle text",
            "Arrow Keys: Navigate | S: Save | Q: Quit | H: Help"
        ]
        
        y = 25
        for line in instructions:
            if line == "":
                y += 10
                continue
            
            # Background
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(self.display_img, (5, y-18), (w+15, y+5), (0, 0, 0), -1)
            
            # Highlight fast mode line
            if "FAST MODE" in line:
                cv2.putText(self.display_img, line, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            else:
                cv2.putText(self.display_img, line, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 25
    
    def draw_help_overlay(self):
        """Draw detailed help overlay."""
        overlay = self.display_img.copy()
        height, width = overlay.shape[:2]
        
        cv2.rectangle(overlay, (50, 50), (width-50, height-50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.display_img, 0.3, 0, self.display_img)
        
        help_text = [
            "TEMPLATE-BASED LABELER - HELP",
            "",
            "FAST OCCUPANCY LABELING:",
            "  ROIs are pre-defined from template",
            "  You only label occupied vs empty",
            "",
            "LABELING:",
            "  • Click on parking spot to select it",
            "  • 1: Mark selected spot as Occupied (red)",
            "  • 2: Mark selected spot as Empty (green)",
            "  • 1-9: Quick select spot by number",
            "",
            "BULK OPERATIONS:",
            "  • O: Mark ALL spots as Occupied",
            "  • E: Mark ALL spots as Empty",
            "  • C: Copy occupancy from previous image",
            "",
            "ZOOM & PAN:",
            "  • +/= : Zoom in",
            "  • - : Zoom out",
            "  • 0 : Reset zoom to 1.0x",
            "  • Mouse wheel: Zoom in/out",
            "  • P : Toggle pan mode (then click + drag to pan)",
            "",
            "QUICK WORKFLOW:",
            "  1. Click spot (or press number key)",
            "  2. Press 1 (occupied) or 2 (empty)",
            "  3. Press Right arrow to next image",
            "  4. Repeat!",
            "",
            "OR for similar images:",
            "  1. Press C to copy from previous image",
            "  2. Adjust any differences",
            "  3. Press Right arrow to next image",
            "",
            "NAVIGATION:",
            "  • Left/Right Arrow: Previous/Next image",
            "  • Home/End: First/Last image",
            "  • S: Save all annotations",
            "  • L: Toggle ROI text labels (hide/show)",
            "  • Q: Quit (auto-saves)",
            "",
            "TIPS:",
            "  • Keep one hand on 1/2 keys, other on arrows",
            "  • Use C to copy when parking lot is similar",
            "  • Use O/E for empty or full lots",
            "  • Auto-saves when navigating between images",
            "  • Use zoom for precise work on small ROIs",
            "  • Enable pan mode (P) when zoomed in to navigate",
            "",
            "Press H again to close help..."
        ]
        
        y = 65
        for line in help_text:
            if "HELP" in line or line.endswith(":"):
                color = (0, 255, 255)
                thickness = 2
                font_scale = 0.6
            else:
                color = (255, 255, 255)
                thickness = 1
                font_scale = 0.5
            
            cv2.putText(self.display_img, line, (70, y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            y += 30 if "HELP" in line else 22
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.pan_mode:
                self.is_panning = True
                self.pan_start = (x, y)
            else:
                self.on_left_click(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.pan_mode:
                self.is_panning = False
                self.pan_start = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning:
                self.on_pan_move(x, y)
            else:
                self.on_mouse_move(x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.on_mouse_wheel(x, y, flags)
    
    def on_mouse_wheel(self, x, y, flags):
        """Handle mouse wheel for zooming."""
        if flags > 0:  # Scroll up - zoom in
            self.zoom_in()
        else:  # Scroll down - zoom out
            self.zoom_out()
    
    def on_pan_move(self, x, y):
        """Handle panning with middle mouse drag."""
        if self.pan_start is not None:
            dx = x - self.pan_start[0]
            dy = y - self.pan_start[1]
            self.pan_offset[0] += dx
            self.pan_offset[1] += dy
            self.pan_start = (x, y)
            self.draw()
    
    def zoom_in(self, factor=1.2):
        """Zoom in by factor."""
        new_zoom = self.zoom_level * factor
        if new_zoom <= self.max_zoom:
            self.zoom_level = new_zoom
            self.draw()
    
    def zoom_out(self, factor=1.2):
        """Zoom out by factor."""
        new_zoom = self.zoom_level / factor
        if new_zoom >= self.min_zoom:
            self.zoom_level = new_zoom
            self.draw()
    
    def reset_zoom(self):
        """Reset zoom to 1.0x and center pan."""
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.draw()
    
    def on_left_click(self, x, y):
        """Handle left mouse click - select ROI."""
        # Convert screen coordinates to image coordinates
        img_x, img_y = self.screen_to_image(x, y)
        
        # Check if clicking inside an ROI
        for i, roi in enumerate(self.template_rois):
            pts = [[int(p[0] * self.img_width), int(p[1] * self.img_height)] for p in roi]
            pts = np.array(pts, dtype=np.int32)
            if cv2.pointPolygonTest(pts, (img_x, img_y), False) >= 0:
                self.selected_roi = i
                self.draw()
                return
        
        # Click outside all ROIs - deselect
        self.selected_roi = None
        self.draw()
    
    def on_mouse_move(self, x, y):
        """Handle mouse movement - show hover."""
        old_hover = self.hover_roi
        self.hover_roi = None
        
        # Convert screen coordinates to image coordinates
        img_x, img_y = self.screen_to_image(x, y)
        
        for i, roi in enumerate(self.template_rois):
            pts = [[int(p[0] * self.img_width), int(p[1] * self.img_height)] for p in roi]
            pts = np.array(pts, dtype=np.int32)
            if cv2.pointPolygonTest(pts, (img_x, img_y), False) >= 0:
                self.hover_roi = i
                break
        
        if old_hover != self.hover_roi:
            self.draw()
    
    def save_current_image(self):
        """Save annotations for current image."""
        fname = os.path.basename(self.image_files[self.current_idx])
        
        # Update or add annotations (always use template ROIs)
        if fname in self.annotations['file_names']:
            idx = self.annotations['file_names'].index(fname)
            self.annotations['rois_list'][idx] = self.template_rois.copy()
            self.annotations['occupancy_list'][idx] = self.current_occupancy.copy()
        else:
            self.annotations['file_names'].append(fname)
            self.annotations['rois_list'].append(self.template_rois.copy())
            self.annotations['occupancy_list'].append(self.current_occupancy.copy())
    
    def run(self):
        """Main loop for the labeler."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.draw()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                self.save_current_image()
                self.save_annotations()
                break
            
            elif key == ord('s'):  # Save
                self.save_current_image()
                self.save_annotations()
            
            elif key == ord('h'):  # Help
                self.show_help = not self.show_help
                self.draw()
            
            elif key == ord('l'):  # Toggle labels
                self.show_labels = not self.show_labels
                self.draw()
            
            elif key == ord('p'):  # Toggle pan mode
                self.pan_mode = not self.pan_mode
                self.is_panning = False
                self.pan_start = None
                self.draw()
            
            elif key == ord('+') or key == ord('='):  # Zoom in
                self.zoom_in()
            
            elif key == ord('-') or key == ord('_'):  # Zoom out
                self.zoom_out()
            
            elif key == ord('0'):  # Reset zoom
                self.reset_zoom()
            
            elif key == ord('1'):  # Mark as occupied
                if self.selected_roi is not None:
                    self.current_occupancy[self.selected_roi] = True
                    self.draw()
            
            elif key == ord('2'):  # Mark as empty
                if self.selected_roi is not None:
                    self.current_occupancy[self.selected_roi] = False
                    self.draw()
            
            elif key == ord('o') or key == ord('x'):  # Mark all as occupied
                self.current_occupancy = [True] * len(self.template_rois)
                self.draw()
            
            elif key == ord('e') or key == ord('a'):  # Mark all as empty (reset)
                self.current_occupancy = [False] * len(self.template_rois)
                self.draw()
            
            elif key == ord('c'):  # Copy from previous image
                if self.current_idx > 0:
                    prev_fname = os.path.basename(self.image_files[self.current_idx - 1])
                    if prev_fname in self.annotations['file_names']:
                        prev_idx = self.annotations['file_names'].index(prev_fname)
                        prev_occupancy = self.annotations['occupancy_list'][prev_idx].copy()
                        
                        # Handle potential mismatch
                        if len(prev_occupancy) != len(self.template_rois):
                            if len(prev_occupancy) < len(self.template_rois):
                                prev_occupancy.extend([False] * (len(self.template_rois) - len(prev_occupancy)))
                            else:
                                prev_occupancy = prev_occupancy[:len(self.template_rois)]
                        
                        self.current_occupancy = prev_occupancy
                        print(f"✓ Copied occupancy from previous image: {prev_fname}")
                        self.draw()
                    else:
                        print(f"⚠️  Previous image not yet labeled")
                else:
                    print(f"⚠️  Already at first image")
            
            elif key == 27:  # ESC - deselect
                self.selected_roi = None
                self.draw()
            
            elif key == 81 or key == 2:  # Left arrow
                self.save_current_image()
                self.load_image(self.current_idx - 1)
            
            elif key == 83 or key == 3:  # Right arrow
                self.save_current_image()
                self.load_image(self.current_idx + 1)
            
            elif key == 82 or key == 0:  # Home
                self.save_current_image()
                self.load_image(0)
            
            elif key == 84 or key == 1:  # End
                self.save_current_image()
                self.load_image(len(self.image_files) - 1)
            
            # Number keys 1-9 for quick selection
            elif 49 <= key <= 57:  # Keys 1-9
                spot_num = key - 49  # Convert to 0-indexed
                if spot_num < len(self.template_rois):
                    self.selected_roi = spot_num
                    self.draw()
        
        cv2.destroyAllWindows()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast template-based occupancy labeling')
    parser.add_argument('image_dir', type=str, help='Directory containing images')
    parser.add_argument('template', type=str, help='ROI template JSON file')
    parser.add_argument('--output', '-o', type=str, default='annotations.json',
                       help='Output JSON file (default: annotations.json)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TEMPLATE-BASED LABELER (FAST MODE)")
    print("=" * 60)
    print(f"\nImage directory: {args.image_dir}")
    print(f"Template: {args.template}")
    print(f"Output file: {args.output}")
    print("\nROIs are pre-defined. You only label occupancy!")
    print("Press 'H' for help")
    print("=" * 60)
    print()
    
    try:
        labeler = TemplateBasedLabeler(args.image_dir, args.template, args.output)
        labeler.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()