#!/usr/bin/env python3
"""
Parking Lot Homography Calibration Tool

This script allows users to interactively calibrate the transformation matrix
between a top-down view (Google Earth) and an angled security camera view of
a parking lot by drawing corresponding polygons on both images.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import Button
import cv2
import argparse
import pickle


class HomographyCalibrator:
    """Interactive calibration tool for homography estimation."""
    
    def __init__(self, topdown_image_path, camera_image_path):
        """
        Initialize the calibrator with two image paths.
        
        Args:
            topdown_image_path: Path to the top-down view image
            camera_image_path: Path to the angled camera view image
        """
        # Load images
        self.topdown_img = cv2.imread(topdown_image_path)
        self.camera_img = cv2.imread(camera_image_path)
        
        if self.topdown_img is None:
            raise ValueError(f"Could not load top-down image: {topdown_image_path}")
        if self.camera_img is None:
            raise ValueError(f"Could not load camera image: {camera_image_path}")
        
        # Convert BGR to RGB for matplotlib
        self.topdown_img = cv2.cvtColor(self.topdown_img, cv2.COLOR_BGR2RGB)
        self.camera_img = cv2.cvtColor(self.camera_img, cv2.COLOR_BGR2RGB)
        
        # Storage for polygon correspondences
        self.topdown_polygons = []
        self.camera_polygons = []
        
        # Current polygon being drawn
        self.current_topdown_polygon = []
        self.current_camera_polygon = []
        
        # Drawing state
        self.drawing_on = 'topdown'  # 'topdown' or 'camera'
        
        # Homography matrix
        self.homography_matrix = None
        
        # Setup matplotlib interface
        self.setup_interface()
    
    def setup_interface(self):
        """Create the interactive matplotlib interface."""
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.suptitle('Parking Lot Homography Calibration Tool', fontsize=14, fontweight='bold')
        
        # Create subplots
        self.ax_topdown = plt.subplot(1, 2, 1)
        self.ax_topdown.set_title('Top-Down View (Google Earth)\nClick to draw polygon | Scroll to zoom | Right-click drag to pan', fontsize=11)
        self.ax_topdown.imshow(self.topdown_img)
        self.ax_topdown.axis('on')  # Enable axis for zoom/pan
        self.ax_topdown.set_xticks([])
        self.ax_topdown.set_yticks([])
        
        self.ax_camera = plt.subplot(1, 2, 2)
        self.ax_camera.set_title('Camera View (Angled)\nClick to draw polygon | Scroll to zoom | Right-click drag to pan', fontsize=11)
        self.ax_camera.imshow(self.camera_img)
        self.ax_camera.axis('on')  # Enable axis for zoom/pan
        self.ax_camera.set_xticks([])
        self.ax_camera.set_yticks([])
        
        # Add buttons
        button_height = 0.04
        button_width = 0.10
        spacing = 0.015
        start_x = 0.12
        
        ax_finish = plt.axes([start_x, 0.02, button_width, button_height])
        self.btn_finish = Button(ax_finish, 'Finish Polygon')
        self.btn_finish.on_clicked(self.finish_polygon)
        
        ax_clear = plt.axes([start_x + button_width + spacing, 0.02, button_width, button_height])
        self.btn_clear = Button(ax_clear, 'Clear Current')
        self.btn_clear.on_clicked(self.clear_current_polygon)
        
        ax_remove = plt.axes([start_x + 2*(button_width + spacing), 0.02, button_width, button_height])
        self.btn_remove = Button(ax_remove, 'Remove Last')
        self.btn_remove.on_clicked(self.remove_last_polygon)
        
        ax_compute = plt.axes([start_x + 3*(button_width + spacing), 0.02, button_width, button_height])
        self.btn_compute = Button(ax_compute, 'Compute H')
        self.btn_compute.on_clicked(self.compute_homography)
        
        ax_save = plt.axes([start_x + 4*(button_width + spacing), 0.02, button_width, button_height])
        self.btn_save = Button(ax_save, 'Save Matrix')
        self.btn_save.on_clicked(self.save_homography)
        
        ax_test = plt.axes([start_x + 5*(button_width + spacing), 0.02, button_width, button_height])
        self.btn_test = Button(ax_test, 'Test Transform')
        self.btn_test.on_clicked(self.test_transform)
        
        ax_reset_zoom = plt.axes([start_x + 6*(button_width + spacing), 0.02, button_width, button_height])
        self.btn_reset_zoom = Button(ax_reset_zoom, 'Reset Zoom')
        self.btn_reset_zoom.on_clicked(self.reset_zoom)
        
        # Status text
        self.status_text = self.fig.text(0.5, 0.95, 'Draw corresponding polygons on both images', 
                                        ha='center', fontsize=10, style='italic')
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Store initial axis limits for zoom reset
        self.initial_xlim_topdown = self.ax_topdown.get_xlim()
        self.initial_ylim_topdown = self.ax_topdown.get_ylim()
        self.initial_xlim_camera = self.ax_camera.get_xlim()
        self.initial_ylim_camera = self.ax_camera.get_ylim()
        
        # Store polygon patches for visualization
        self.topdown_patches = []
        self.camera_patches = []
        self.current_topdown_points = []
        self.current_camera_points = []
    
    def on_click(self, event):
        """Handle mouse clicks for polygon drawing."""
        if event.inaxes == self.ax_topdown:
            if event.button == 1:  # Left click
                self.current_topdown_polygon.append([event.xdata, event.ydata])
                self.ax_topdown.plot(event.xdata, event.ydata, 'ro', markersize=5)
                
                # Draw line to previous point
                if len(self.current_topdown_polygon) > 1:
                    pts = np.array(self.current_topdown_polygon)
                    self.ax_topdown.plot(pts[-2:, 0], pts[-2:, 1], 'r-', linewidth=2)
                
                self.fig.canvas.draw()
                self.update_status(f"Top-down: {len(self.current_topdown_polygon)} points | " +
                                 f"Camera: {len(self.current_camera_polygon)} points")
        
        elif event.inaxes == self.ax_camera:
            if event.button == 1:  # Left click
                self.current_camera_polygon.append([event.xdata, event.ydata])
                self.ax_camera.plot(event.xdata, event.ydata, 'go', markersize=5)
                
                # Draw line to previous point
                if len(self.current_camera_polygon) > 1:
                    pts = np.array(self.current_camera_polygon)
                    self.ax_camera.plot(pts[-2:, 0], pts[-2:, 1], 'g-', linewidth=2)
                
                self.fig.canvas.draw()
                self.update_status(f"Top-down: {len(self.current_topdown_polygon)} points | " +
                                 f"Camera: {len(self.current_camera_polygon)} points")
    
    def on_scroll(self, event):
        """Handle mouse scroll for zooming."""
        if event.inaxes not in [self.ax_topdown, self.ax_camera]:
            return
        
        # Get the current axis
        ax = event.inaxes
        
        # Get current limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        # Get event location
        xdata = event.xdata
        ydata = event.ydata
        
        # Zoom factor
        zoom_factor = 1.3
        
        if event.button == 'up':
            # Zoom in
            scale_factor = 1 / zoom_factor
        elif event.button == 'down':
            # Zoom out
            scale_factor = zoom_factor
        else:
            return
        
        # Calculate new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        # Center zoom on cursor position
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
        new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
        new_ylim = [ydata - new_height * (1 - rely), ydata + new_height * rely]
        
        # Apply new limits
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        
        self.fig.canvas.draw()
    
    def reset_zoom(self, event):
        """Reset zoom to show entire images."""
        self.ax_topdown.set_xlim(self.initial_xlim_topdown)
        self.ax_topdown.set_ylim(self.initial_ylim_topdown)
        self.ax_camera.set_xlim(self.initial_xlim_camera)
        self.ax_camera.set_ylim(self.initial_ylim_camera)
        self.fig.canvas.draw()
        self.update_status("Zoom reset to full view")
    
    def finish_polygon(self, event):
        """Finish the current polygon pair and store it."""
        if len(self.current_topdown_polygon) < 3 or len(self.current_camera_polygon) < 3:
            self.update_status("Error: Each polygon needs at least 3 points!")
            return
        
        if len(self.current_topdown_polygon) != len(self.current_camera_polygon):
            self.update_status("Error: Polygons must have the same number of points!")
            return
        
        # Store polygons
        self.topdown_polygons.append(np.array(self.current_topdown_polygon))
        self.camera_polygons.append(np.array(self.current_camera_polygon))
        
        # Draw completed polygons
        poly_td = MplPolygon(self.current_topdown_polygon, fill=False, 
                            edgecolor='red', linewidth=2, alpha=0.7)
        poly_cam = MplPolygon(self.current_camera_polygon, fill=False, 
                             edgecolor='green', linewidth=2, alpha=0.7)
        
        self.ax_topdown.add_patch(poly_td)
        self.ax_camera.add_patch(poly_cam)
        
        # Add polygon numbers
        td_center = np.mean(self.current_topdown_polygon, axis=0)
        cam_center = np.mean(self.current_camera_polygon, axis=0)
        
        poly_num = len(self.topdown_polygons)
        self.ax_topdown.text(td_center[0], td_center[1], str(poly_num), 
                           color='red', fontsize=12, fontweight='bold',
                           ha='center', va='center')
        self.ax_camera.text(cam_center[0], cam_center[1], str(poly_num), 
                          color='green', fontsize=12, fontweight='bold',
                          ha='center', va='center')
        
        # Clear current polygons
        self.current_topdown_polygon = []
        self.current_camera_polygon = []
        
        self.fig.canvas.draw()
        self.update_status(f"Polygon pair {poly_num} saved! Total pairs: {len(self.topdown_polygons)}")
    
    def clear_current_polygon(self, event):
        """Clear the current polygon being drawn."""
        self.current_topdown_polygon = []
        self.current_camera_polygon = []
        
        # Redraw to remove temporary points
        self.redraw_all()
        self.update_status("Current polygon cleared")
    
    def remove_last_polygon(self, event):
        """Remove the last completed polygon pair."""
        if len(self.topdown_polygons) > 0:
            self.topdown_polygons.pop()
            self.camera_polygons.pop()
            self.redraw_all()
            self.update_status(f"Last polygon removed. Total pairs: {len(self.topdown_polygons)}")
        else:
            self.update_status("No polygons to remove!")
    
    def redraw_all(self):
        """Redraw all polygons."""
        # Store current zoom state
        xlim_td = self.ax_topdown.get_xlim()
        ylim_td = self.ax_topdown.get_ylim()
        xlim_cam = self.ax_camera.get_xlim()
        ylim_cam = self.ax_camera.get_ylim()
        
        self.ax_topdown.clear()
        self.ax_camera.clear()
        
        self.ax_topdown.set_title('Top-Down View (Google Earth)\nClick to draw polygon | Scroll to zoom | Right-click drag to pan', fontsize=11)
        self.ax_topdown.imshow(self.topdown_img)
        self.ax_topdown.axis('on')
        self.ax_topdown.set_xticks([])
        self.ax_topdown.set_yticks([])
        
        self.ax_camera.set_title('Camera View (Angled)\nClick to draw polygon | Scroll to zoom | Right-click drag to pan', fontsize=11)
        self.ax_camera.imshow(self.camera_img)
        self.ax_camera.axis('on')
        self.ax_camera.set_xticks([])
        self.ax_camera.set_yticks([])
        
        # Restore zoom state
        self.ax_topdown.set_xlim(xlim_td)
        self.ax_topdown.set_ylim(ylim_td)
        self.ax_camera.set_xlim(xlim_cam)
        self.ax_camera.set_ylim(ylim_cam)
        
        # Redraw all completed polygons
        for i, (td_poly, cam_poly) in enumerate(zip(self.topdown_polygons, self.camera_polygons)):
            poly_td = MplPolygon(td_poly, fill=False, edgecolor='red', linewidth=2, alpha=0.7)
            poly_cam = MplPolygon(cam_poly, fill=False, edgecolor='green', linewidth=2, alpha=0.7)
            
            self.ax_topdown.add_patch(poly_td)
            self.ax_camera.add_patch(poly_cam)
            
            td_center = np.mean(td_poly, axis=0)
            cam_center = np.mean(cam_poly, axis=0)
            
            self.ax_topdown.text(td_center[0], td_center[1], str(i+1), 
                               color='red', fontsize=12, fontweight='bold',
                               ha='center', va='center')
            self.ax_camera.text(cam_center[0], cam_center[1], str(i+1), 
                              color='green', fontsize=12, fontweight='bold',
                              ha='center', va='center')
        
        # Draw current incomplete polygons
        if len(self.current_topdown_polygon) > 0:
            pts = np.array(self.current_topdown_polygon)
            self.ax_topdown.plot(pts[:, 0], pts[:, 1], 'ro-', markersize=5, linewidth=2)
        
        if len(self.current_camera_polygon) > 0:
            pts = np.array(self.current_camera_polygon)
            self.ax_camera.plot(pts[:, 0], pts[:, 1], 'go-', markersize=5, linewidth=2)
        
        self.fig.canvas.draw()
    
    def compute_homography(self, event):
        """Compute the homography matrix from polygon correspondences."""
        if len(self.topdown_polygons) < 2:
            self.update_status("Error: Need at least 2 polygon pairs to compute homography!")
            return
        
        # Collect all point correspondences
        src_points = []  # Camera view points
        dst_points = []  # Top-down view points
        
        for cam_poly, td_poly in zip(self.camera_polygons, self.topdown_polygons):
            src_points.extend(cam_poly)
            dst_points.extend(td_poly)
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Compute homography using RANSAC
        self.homography_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        
        if self.homography_matrix is None:
            self.update_status("Error: Could not compute homography!")
            return
        
        # Calculate reprojection error
        inliers = np.sum(mask)
        total = len(mask)
        
        # Transform source points and compute error
        src_homogeneous = np.hstack([src_points, np.ones((len(src_points), 1))])
        transformed = (self.homography_matrix @ src_homogeneous.T).T
        transformed = transformed[:, :2] / transformed[:, 2:]
        
        errors = np.linalg.norm(transformed - dst_points, axis=1)
        mean_error = np.mean(errors[mask.ravel() == 1])
        
        self.update_status(f"Homography computed! Inliers: {inliers}/{total}, Mean error: {mean_error:.2f} pixels")
        
        print("\nHomography Matrix:")
        print(self.homography_matrix)
        print(f"\nReprojection Statistics:")
        print(f"  Inliers: {inliers}/{total} ({100*inliers/total:.1f}%)")
        print(f"  Mean error: {mean_error:.2f} pixels")
        print(f"  Max error: {np.max(errors[mask.ravel() == 1]):.2f} pixels")
    
    def test_transform(self, event):
        """Test the homography by transforming camera view polygons."""
        if self.homography_matrix is None:
            self.update_status("Error: Compute homography first!")
            return
        
        # Create new figure for visualization
        fig_test = plt.figure(figsize=(16, 8))
        fig_test.suptitle('Transformation Test - Warped Camera Polygons on Top-Down View', 
                         fontsize=14, fontweight='bold')
        
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title('Original Camera View Polygons', fontsize=11)
        ax1.imshow(self.camera_img)
        ax1.axis('off')
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title('Transformed Polygons Overlaid on Top-Down View', fontsize=11)
        ax2.imshow(self.topdown_img)
        ax2.axis('off')
        
        # Draw original polygons on camera view
        for i, cam_poly in enumerate(self.camera_polygons):
            poly = MplPolygon(cam_poly, fill=False, edgecolor='green', linewidth=2, alpha=0.7)
            ax1.add_patch(poly)
            center = np.mean(cam_poly, axis=0)
            ax1.text(center[0], center[1], str(i+1), color='green', 
                    fontsize=12, fontweight='bold', ha='center', va='center')
        
        # Transform and draw polygons on top-down view
        for i, cam_poly in enumerate(self.camera_polygons):
            # Transform polygon points
            src_homogeneous = np.hstack([cam_poly, np.ones((len(cam_poly), 1))])
            transformed = (self.homography_matrix @ src_homogeneous.T).T
            transformed = transformed[:, :2] / transformed[:, 2:]
            
            # Draw original top-down polygon (red)
            td_poly = self.topdown_polygons[i]
            poly_original = MplPolygon(td_poly, fill=False, edgecolor='red', 
                                      linewidth=2, alpha=0.5, linestyle='--', label='Original' if i==0 else '')
            ax2.add_patch(poly_original)
            
            # Draw transformed polygon (blue)
            poly_transformed = MplPolygon(transformed, fill=False, edgecolor='blue', 
                                         linewidth=2, alpha=0.7, label='Transformed' if i==0 else '')
            ax2.add_patch(poly_transformed)
            
            center_orig = np.mean(td_poly, axis=0)
            center_trans = np.mean(transformed, axis=0)
            
            ax2.text(center_orig[0], center_orig[1], str(i+1), color='red', 
                    fontsize=12, fontweight='bold', ha='center', va='center')
            ax2.text(center_trans[0], center_trans[1], str(i+1), color='blue', 
                    fontsize=12, fontweight='bold', ha='center', va='center')
        
        ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        
        self.update_status("Test transformation displayed in new window")
    
    def save_homography(self, event):
        """Save the homography matrix to a file."""
        if self.homography_matrix is None:
            self.update_status("Error: Compute homography first!")
            return
        
        data = {
            'homography_matrix': self.homography_matrix,
            'topdown_shape': self.topdown_img.shape,
            'camera_shape': self.camera_img.shape,
            'num_polygons': len(self.topdown_polygons)
        }
        
        with open('homography_matrix.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        np.savetxt('homography_matrix.txt', self.homography_matrix, fmt='%.6f')
        
        self.update_status("Homography matrix saved to homography_matrix.pkl and .txt")
        print("\nHomography matrix saved to:")
        print("  - homography_matrix.pkl (pickle format)")
        print("  - homography_matrix.txt (text format)")
    
    def update_status(self, message):
        """Update the status message."""
        self.status_text.set_text(message)
        self.fig.canvas.draw()
        print(message)
    
    def run(self):
        """Start the interactive calibration interface."""
        print("\n" + "="*70)
        print("PARKING LOT HOMOGRAPHY CALIBRATION TOOL")
        print("="*70)
        print("\nInstructions:")
        print("  1. Click on the TOP-DOWN view to mark polygon vertices")
        print("  2. Click on the CAMERA view to mark corresponding vertices")
        print("  3. Use the SAME number of points for both polygons")
        print("  4. Click 'Finish Polygon' when done with one correspondence")
        print("  5. Repeat for multiple polygon pairs (at least 2, more is better)")
        print("  6. Click 'Compute H' to calculate the homography matrix")
        print("  7. Click 'Test Transform' to visualize the results")
        print("  8. Click 'Save Matrix' to save the homography for later use")
        print("\nZoom Controls:")
        print("  - Scroll wheel: Zoom in/out (centered on cursor)")
        print("  - Right-click + drag: Pan around the image")
        print("  - 'Reset Zoom' button: Return to full view")
        print("\nTips:")
        print("  - Draw polygons on distinct features (parking spots, markings, etc.)")
        print("  - Use zoom to click precisely on corners")
        print("  - More polygon pairs = better accuracy")
        print("  - Aim for 4-8 polygon pairs across the entire parking lot")
        print("="*70 + "\n")
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive homography calibration tool for parking lot images'
    )
    parser.add_argument('topdown', help='Path to top-down view image (Google Earth)')
    parser.add_argument('camera', help='Path to angled camera view image')
    
    args = parser.parse_args()
    
    try:
        calibrator = HomographyCalibrator(args.topdown, args.camera)
        calibrator.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())