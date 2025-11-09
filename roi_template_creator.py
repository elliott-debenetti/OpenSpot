import cv2
import numpy as np
import json
import os
from pathlib import Path


class ROITemplateCreator:
    """
    Create a template of ROIs that can be reused across multiple images.
    Only defines the ROI locations - occupancy is labeled separately.
    """
    def __init__(self, reference_image_path, output_template='roi_template.json', edit_template=None):
        self.reference_image_path = reference_image_path
        self.output_template = output_template
        self.edit_template = edit_template
        self.editing_mode = False
        
        # Load reference image
        self.img = cv2.imread(reference_image_path)
        if self.img is None:
            raise ValueError(f"Could not load image: {reference_image_path}")
        
        self.display_img = None
        self.img_height, self.img_width = self.img.shape[:2]
        
        # Template ROIs (in normalized coordinates)
        self.template_rois = []
        
        # Drawing state
        self.current_roi_points = []  # Points being drawn (in image coords)
        self.selected_roi = None
        self.selected_point = None
        self.dragging = False
        self.hover_roi = None
        self.hover_point = None
        
        # UI state
        self.show_help = False
        self.show_grid_numbers = True
        
        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.pan_mode = False
        self.is_panning = False
        self.pan_start = None
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        
        self.window_name = 'ROI Template Creator'
        
        # Load existing template if it exists
        self.load_template()
    
    def load_template(self):
        """Load existing template from file."""
        template_to_load = self.edit_template if self.edit_template else self.output_template
        
        if os.path.exists(template_to_load):
            try:
                with open(template_to_load, 'r') as f:
                    data = json.load(f)
                    self.template_rois = data.get('rois', [])
                    
                    # Check if template was created for a different image size
                    template_width = data.get('image_width')
                    template_height = data.get('image_height')
                    
                    if self.edit_template:
                        self.editing_mode = True
                        print(f"\n{'='*60}")
                        print(f"📝 EDITING EXISTING TEMPLATE")
                        print(f"{'='*60}")
                        print(f"Template file: {template_to_load}")
                        print(f"Loaded: {len(self.template_rois)} ROIs")
                        
                        if template_width and template_height:
                            if template_width != self.img_width or template_height != self.img_height:
                                print(f"\n⚠️  WARNING: Image size mismatch!")
                                print(f"   Template created for: {template_width}x{template_height}")
                                print(f"   Current image size: {self.img_width}x{self.img_height}")
                                print(f"   ROIs will be scaled to fit the new image.")
                            else:
                                print(f"✓ Image dimensions match: {self.img_width}x{self.img_height}")
                        
                        print(f"{'='*60}\n")
                    else:
                        print(f"✓ Loaded existing template: {len(self.template_rois)} ROIs from {template_to_load}")
                        
            except Exception as e:
                print(f"Could not load template: {e}")
        elif self.edit_template:
            raise ValueError(f"Edit template file not found: {self.edit_template}")
    
    def save_template(self):
        """Save template to file."""
        template_data = {
            'reference_image': os.path.basename(self.reference_image_path),
            'image_width': self.img_width,
            'image_height': self.img_height,
            'num_rois': len(self.template_rois),
            'rois': self.template_rois
        }
        
        with open(self.output_template, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        if self.editing_mode:
            print(f"✓ Template updated: {len(self.template_rois)} ROIs → {self.output_template}")
        else:
            print(f"✓ Template saved: {len(self.template_rois)} ROIs → {self.output_template}")
    
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
    
    def image_to_normalized(self, point):
        """Convert image coordinates to normalized coordinates (0-1)."""
        x = point[0] / self.img_width
        y = point[1] / self.img_height
        return [x, y]
    
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
        
        # Draw existing template ROIs
        for i, roi in enumerate(self.template_rois):
            # Convert to image coordinates with zoom/pan
            pts = [self.normalized_to_image(p) for p in roi]
            pts = np.array(pts, dtype=np.int32)
            
            # Determine color based on selection
            if i == self.selected_roi:
                color = (255, 255, 0)  # Yellow for selected
                thickness = max(1, int(2 * self.zoom_level))
                alpha = 0.1
            elif i == self.hover_roi:
                color = (255, 165, 0)  # Orange for hover
                thickness = max(1, int(1 * self.zoom_level))
                alpha = 0.3
            else:
                color = (0, 255, 0)  # Green for all ROIs
                thickness = max(1, int(1 * self.zoom_level))
                alpha = 0.2
            
            # Draw ROI polygon with semi-transparent fill
            overlay = self.display_img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, alpha, self.display_img, 1 - alpha, 0, self.display_img)
            
            # Draw outline
            cv2.polylines(self.display_img, [pts], True, color, thickness)
            
            # Draw corner points
            point_radius = max(2, int(3 * self.zoom_level))
            selected_point_radius = max(3, int(5 * self.zoom_level))
            for j, pt in enumerate(pts):
                if i == self.selected_roi and j == self.selected_point:
                    cv2.circle(self.display_img, tuple(pt), selected_point_radius, (255, 255, 0), -1)
                elif i == self.hover_roi and j == self.hover_point:
                    cv2.circle(self.display_img, tuple(pt), point_radius + 1, (255, 165, 0), -1)
                else:
                    cv2.circle(self.display_img, tuple(pt), point_radius, color, -1)
            
            # Draw ROI number
            if self.show_grid_numbers:
                center = np.mean(pts, axis=0).astype(int)
                font_scale = max(0.4, 0.7 * self.zoom_level)
                font_thickness = max(1, int(2 * self.zoom_level))
                cv2.putText(self.display_img, str(i+1), tuple(center),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Draw ROI being created
        if len(self.current_roi_points) > 0:
            # Convert to zoomed/panned coordinates
            zoomed_pts = [self.apply_zoom_pan(p) for p in self.current_roi_points]
            pts = np.array(zoomed_pts, dtype=np.int32)
            
            if len(self.current_roi_points) > 1:
                thickness = max(1, int(2 * self.zoom_level))
                cv2.polylines(self.display_img, [pts], False, (255, 0, 255), thickness)
            
            point_radius = max(2, int(3 * self.zoom_level))
            for pt in zoomed_pts:
                cv2.circle(self.display_img, tuple(pt), point_radius, (255, 0, 255), -1)
            
            # Show instruction
            cv2.putText(self.display_img, 
                       f"Click point {len(self.current_roi_points)+1}/4",
                       (10, self.img_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Draw instructions
        self.draw_instructions()
        
        cv2.imshow(self.window_name, self.display_img)
    
    def draw_instructions(self):
        """Draw instruction text overlay."""
        if self.show_help:
            self.draw_help_overlay()
            return
        
        # Build status line with zoom and pan info
        zoom_text = f"Zoom: {self.zoom_level:.1f}x"
        pan_text = " | PAN MODE" if self.pan_mode else ""
        mode_text = " | EDITING" if self.editing_mode else ""
        
        instructions = [
            f"TEMPLATE CREATOR - {len(self.template_rois)} ROIs | {zoom_text}{pan_text}{mode_text}",
            "",
            "Click 4 corners to draw ROI | Del: Delete | S: Save | Q: Quit",
            "Drag corners to adjust | +/- or Scroll: Zoom | P: Pan | 0: Reset | N: Numbers | H: Help"
        ]
        
        y = 25
        for line in instructions:
            if line == "":
                y += 10
                continue
            
            # Background for text
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(self.display_img, (5, y-20), (w+15, y+5), (0, 0, 0), -1)
            
            cv2.putText(self.display_img, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 30
    
    def draw_help_overlay(self):
        """Draw detailed help overlay."""
        overlay = self.display_img.copy()
        height, width = overlay.shape[:2]
        
        # Semi-transparent background
        cv2.rectangle(overlay, (50, 50), (width-50, height-50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.display_img, 0.3, 0, self.display_img)
        
        help_text = [
            "ROI TEMPLATE CREATOR - HELP",
            "",
            "PURPOSE:",
            "  Create a reusable template of parking spot ROIs",
            "  Use this template across all images in your dataset",
            "  Edit existing templates to add/remove/adjust ROIs",
            "",
            "DRAWING ROIs:",
            "  • Click 4 corners to create each parking spot",
            "  • Right-click or ESC to cancel current ROI",
            "  • Draw all parking spots visible in the image",
            "",
            "EDITING ROIs:",
            "  • Click and drag corners to adjust shape",
            "  • Click inside ROI to select it",
            "  • Del/Backspace: Delete selected ROI",
            "  • Add new ROIs to existing template",
            "",
            "ZOOM & PAN:",
            "  • Mouse Wheel or +/-: Zoom in/out",
            "  • P: Toggle pan mode, then drag to pan",
            "  • Middle Mouse: Hold and drag to pan",
            "  • 0 (zero): Reset zoom and pan to default",
            "",
            "SAVING:",
            "  • S: Save template to JSON file",
            "  • Template will be used for occupancy labeling",
            "",
            "OTHER CONTROLS:",
            "  • N: Toggle ROI numbers on/off",
            "  • Q: Save and quit",
            "",
            "Press H again to close help..."
        ]
        
        y = 70
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
            y += 30 if "HELP" in line else 23
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.on_left_click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.on_right_click(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.on_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.on_left_release(x, y)
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle mouse button down - start panning
            self.is_panning = True
            self.pan_start = (x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            # Middle mouse button up - stop panning
            self.is_panning = False
            self.pan_start = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.on_mouse_wheel(x, y, flags)
    
    
    def on_mouse_wheel(self, x, y, flags):
        """Handle mouse wheel for zooming."""
        if flags > 0:  # Scroll up - zoom in
            self.zoom_in()
        else:  # Scroll down - zoom out
            self.zoom_out()
    
    def on_pan_move(self, x, y):
        """Handle panning with middle mouse drag or pan mode."""
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
        """Handle left mouse click."""
        # If in pan mode, start panning with left click
        if self.pan_mode:
            self.pan_start = (x, y)
            return
        
        # Check if clicking on an existing point
        for i, roi in enumerate(self.template_rois):
            pts = [self.normalized_to_image(p) for p in roi]
            for j, pt in enumerate(pts):
                dist = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
                if dist < max(8, 8 * self.zoom_level):
                    self.selected_roi = i
                    self.selected_point = j
                    self.dragging = True
                    return
        
        # Check if clicking inside an existing ROI
        for i, roi in enumerate(self.template_rois):
            pts = [self.normalized_to_image(p) for p in roi]
            pts = np.array(pts, dtype=np.int32)
            if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                self.selected_roi = i
                self.selected_point = None
                self.draw()
                return
        
        # Otherwise, add point to new ROI (convert screen coords to image coords)
        if len(self.current_roi_points) < 4:
            img_x, img_y = self.screen_to_image(x, y)
            self.current_roi_points.append([img_x, img_y])
            
            # If we have 4 points, create the ROI
            if len(self.current_roi_points) == 4:
                # Convert to normalized coordinates
                norm_roi = [self.image_to_normalized(p) for p in self.current_roi_points]
                self.template_rois.append(norm_roi)
                self.current_roi_points = []
                self.selected_roi = len(self.template_rois) - 1
                print(f"✓ ROI {len(self.template_rois)} created")
            
            self.draw()
    
    def on_right_click(self, x, y):
        """Handle right mouse click."""
        if len(self.current_roi_points) > 0:
            self.current_roi_points = []
            self.draw()
    
    def on_mouse_move(self, x, y):
        """Handle mouse movement."""
        # Handle panning
        if self.is_panning or (self.pan_mode and self.pan_start is not None):
            self.on_pan_move(x, y)
            return
        
        if self.dragging and self.selected_roi is not None and self.selected_point is not None:
            # Update point position - convert screen coords to image coords
            img_x, img_y = self.screen_to_image(x, y)
            norm_point = self.image_to_normalized([img_x, img_y])
            self.template_rois[self.selected_roi][self.selected_point] = norm_point
            self.draw()
        else:
            # Check for hover
            old_hover_roi = self.hover_roi
            old_hover_point = self.hover_point
            self.hover_roi = None
            self.hover_point = None
            
            for i, roi in enumerate(self.template_rois):
                pts = [self.normalized_to_image(p) for p in roi]
                for j, pt in enumerate(pts):
                    dist = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
                    if dist < max(8, 8 * self.zoom_level):
                        self.hover_roi = i
                        self.hover_point = j
                        break
                if self.hover_roi is not None:
                    break
            
            if old_hover_roi != self.hover_roi or old_hover_point != self.hover_point:
                self.draw()
    
    def on_left_release(self, x, y):
        """Handle left mouse button release."""
        self.dragging = False
        if self.pan_mode:
            self.pan_start = None
    
    def run(self):
        """Main loop for the template creator."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.draw()
        
        if not self.editing_mode:
            print("\n" + "=" * 60)
            print("ROI TEMPLATE CREATOR")
            print("=" * 60)
            print("Draw ROIs for all parking spots on this reference image.")
            print("These ROIs will be reused across all images in your dataset.")
            print("Press 'H' for help | 'S' to save | 'Q' to quit")
            print("=" * 60 + "\n")
        else:
            print("You can now modify the template:")
            print("  • Add new ROIs by clicking 4 corners")
            print("  • Edit ROIs by dragging corner points")
            print("  • Delete ROIs by selecting and pressing Del")
            print("  • Press 'S' to save | 'Q' to save and quit")
            print()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                self.save_template()
                break
            
            elif key == ord('s'):  # Save
                self.save_template()
            
            elif key == ord('h'):  # Help
                self.show_help = not self.show_help
                self.draw()
            
            elif key == ord('n'):  # Toggle numbers
                self.show_grid_numbers = not self.show_grid_numbers
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
            
            elif key in [8, 127]:  # Backspace or Delete
                if self.selected_roi is not None:
                    print(f"✗ Deleted ROI {self.selected_roi + 1}")
                    del self.template_rois[self.selected_roi]
                    self.selected_roi = None
                    self.draw()
            
            elif key == 27:  # ESC
                if len(self.current_roi_points) > 0:
                    self.current_roi_points = []
                    self.draw()
                else:
                    self.selected_roi = None
                    self.draw()
        
        cv2.destroyAllWindows()
        if self.editing_mode:
            print(f"\n✓ Template updated with {len(self.template_rois)} ROIs")
        else:
            print(f"\n✓ Template saved with {len(self.template_rois)} ROIs")
        return self.template_rois


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create or edit ROI template for parking spots')
    parser.add_argument('reference_image', type=str, 
                       help='Reference image to define ROIs on')
    parser.add_argument('--output', '-o', type=str, default='roi_template.json',
                       help='Output template file (default: roi_template.json)')
    parser.add_argument('--edit', '-e', type=str, default=None,
                       help='Edit an existing template file (loads ROIs from this file)')
    
    args = parser.parse_args()
    
    # Validate that edit file exists if specified
    if args.edit and not os.path.exists(args.edit):
        print(f"Error: Template file not found: {args.edit}")
        return
    
    try:
        creator = ROITemplateCreator(args.reference_image, args.output, args.edit)
        creator.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()