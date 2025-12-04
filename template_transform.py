#!/usr/bin/env python3
"""
Apply saved homography matrix to transform points from camera view to top-down view.

This script demonstrates how to use the calibrated homography matrix to transform
coordinates from the angled security camera view to the top-down parking lot view.
"""

import numpy as np
import cv2
import pickle
import argparse
import json
import os
from pathlib import Path


class HomographyTransformer:
    """Apply homography transformation to points and polygons."""
    
    def __init__(self, homography_file='homography_matrix.pkl'):
        """
        Load the saved homography matrix.
        
        Args:
            homography_file: Path to the pickle file containing the homography matrix
        """
        with open(homography_file, 'rb') as f:
            data = pickle.load(f)
        
        self.H = data['homography_matrix']
        self.topdown_shape = data['topdown_shape']
        self.camera_shape = data['camera_shape']
        
        print("Homography matrix loaded successfully!")
        print(f"Top-down image shape: {self.topdown_shape}")
        print(f"Camera image shape: {self.camera_shape}")
        print("\nHomography Matrix:")
        print(self.H)
    
    def transform_point(self, point):
        """
        Transform a single point from camera view to top-down view.
        
        Args:
            point: (x, y) coordinates in camera view
            
        Returns:
            (x, y) coordinates in top-down view
        """
        point = np.array([point[0], point[1], 1.0])
        transformed = self.H @ point
        transformed = transformed[:2] / transformed[2]
        return transformed
    
    def transform_points(self, points):
        """
        Transform multiple points from camera view to top-down view.
        
        Args:
            points: Array of shape (N, 2) with (x, y) coordinates in camera view
            
        Returns:
            Array of shape (N, 2) with (x, y) coordinates in top-down view
        """
        points = np.array(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        # Convert to homogeneous coordinates
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        
        # Apply homography
        transformed = (self.H @ homogeneous.T).T
        
        # Convert back to Cartesian coordinates
        transformed = transformed[:, :2] / transformed[:, 2:]
        
        return transformed
    
    def transform_polygon(self, polygon):
        """
        Transform a polygon from camera view to top-down view.
        
        Args:
            polygon: Array of shape (N, 2) with polygon vertices in camera view
            
        Returns:
            Array of shape (N, 2) with polygon vertices in top-down view
        """
        return self.transform_points(polygon)
    
    def transform_bounding_box(self, bbox):
        """
        Transform a bounding box from camera view to top-down view.
        
        Args:
            bbox: (x, y, width, height) in camera view
            
        Returns:
            Polygon vertices in top-down view (4 corners)
        """
        x, y, w, h = bbox
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ])
        return self.transform_points(corners)
    
    def warp_image(self, camera_image):
        """
        Warp the entire camera image to top-down view.
        
        Args:
            camera_image: Camera view image
            
        Returns:
            Warped image in top-down view
        """
        h, w = self.topdown_shape[:2]
        warped = cv2.warpPerspective(camera_image, self.H, (w, h))
        return warped
    
    def inverse_transform_point(self, point):
        """
        Transform a point from top-down view back to camera view.
        
        Args:
            point: (x, y) coordinates in top-down view
            
        Returns:
            (x, y) coordinates in camera view
        """
        H_inv = np.linalg.inv(self.H)
        point = np.array([point[0], point[1], 1.0])
        transformed = H_inv @ point
        transformed = transformed[:2] / transformed[2]
        return transformed
    
    def get_polygon_area_ratio(self, camera_polygon):
        """
        Calculate the ratio of polygon areas between camera and top-down views.
        This shows the scaling effect of the transformation.
        
        Args:
            camera_polygon: Polygon vertices in camera view
            
        Returns:
            Ratio of (top-down area) / (camera area)
        """
        camera_polygon = np.array(camera_polygon)
        topdown_polygon = self.transform_polygon(camera_polygon)
        
        # Calculate areas using the Shoelace formula
        def polygon_area(vertices):
            x = vertices[:, 0]
            y = vertices[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        camera_area = polygon_area(camera_polygon)
        topdown_area = polygon_area(topdown_polygon)
        
        if camera_area > 0:
            return topdown_area / camera_area
        else:
            return float('inf')
    
    def load_and_transform_annotations(self, annotations_file, image_dir=None):
        """
        Load annotations.json and transform all ROIs to top-down view.
        
        Args:
            annotations_file: Path to annotations.json
            image_dir: Optional directory containing the images
            
        Returns:
            Dictionary with transformed annotations
        """
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        print(f"\nLoaded {len(annotations['file_names'])} annotated images")
        print(f"Each image has {len(annotations['rois_list'][0])} ROIs")
        
        # Transform all ROIs from normalized coordinates to pixel coordinates and then to top-down
        transformed_annotations = {
            'file_names': annotations['file_names'].copy(),
            'rois_list_camera': [],  # Original camera view ROIs in pixels
            'rois_list_topdown': [],  # Transformed top-down ROIs
            'occupancy_list': annotations['occupancy_list'].copy()
        }
        
        camera_h, camera_w = self.camera_shape[:2]
        
        for img_idx, (fname, rois, occupancy) in enumerate(zip(
            annotations['file_names'], 
            annotations['rois_list'], 
            annotations['occupancy_list']
        )):
            camera_rois_pixels = []
            topdown_rois = []
            
            for roi in rois:
                # Convert from normalized (0-1) to pixel coordinates in camera view
                roi_pixels = np.array([[x * camera_w, y * camera_h] for x, y in roi])
                camera_rois_pixels.append(roi_pixels.tolist())
                
                # Transform to top-down view
                roi_topdown = self.transform_polygon(roi_pixels)
                topdown_rois.append(roi_topdown.tolist())
            
            transformed_annotations['rois_list_camera'].append(camera_rois_pixels)
            transformed_annotations['rois_list_topdown'].append(topdown_rois)
        
        print(f"✓ Transformed all ROIs to top-down view")
        return transformed_annotations
    
    def visualize_transformed_annotations(self, annotations_file, image_file, output_file=None):
        """
        Visualize original and transformed ROIs side-by-side.
        
        Args:
            annotations_file: Path to annotations.json
            image_file: Path to a camera image to warp and display
            output_file: Optional path to save the visualization
        """
        # Load annotations
        transformed_data = self.load_and_transform_annotations(annotations_file)
        
        # Load the camera image
        camera_img = cv2.imread(image_file)
        if camera_img is None:
            raise ValueError(f"Could not load image: {image_file}")
        
        # Get the filename to find corresponding annotations
        fname = os.path.basename(image_file)
        if fname not in transformed_data['file_names']:
            print(f"Warning: {fname} not found in annotations. Using first image's ROIs.")
            img_idx = 0
        else:
            img_idx = transformed_data['file_names'].index(fname)
        
        # Get ROIs for this image
        camera_rois = transformed_data['rois_list_camera'][img_idx]
        topdown_rois = transformed_data['rois_list_topdown'][img_idx]
        occupancy = transformed_data['occupancy_list'][img_idx]
        
        # Create warped image
        topdown_img = self.warp_image(camera_img)
        
        # Draw ROIs on camera view
        camera_display = camera_img.copy()
        for i, (roi, occupied) in enumerate(zip(camera_rois, occupancy)):
            pts = np.array(roi, dtype=np.int32)
            color = (0, 0, 255) if occupied else (0, 255, 0)  # Red=occupied, Green=empty
            cv2.polylines(camera_display, [pts], True, color, 2)
            
            # Draw spot number
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(camera_display, str(i), tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw ROIs on top-down view
        topdown_display = topdown_img.copy()
        for i, (roi, occupied) in enumerate(zip(topdown_rois, occupancy)):
            pts = np.array(roi, dtype=np.int32)
            color = (0, 0, 255) if occupied else (0, 255, 0)  # Red=occupied, Green=empty
            cv2.polylines(topdown_display, [pts], True, color, 2)
            
            # Draw spot number
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(topdown_display, str(i), tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add labels
        cv2.putText(camera_display, "Camera View", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(topdown_display, "Top-Down View (Transformed)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Resize images to same height for side-by-side display
        h1, w1 = camera_display.shape[:2]
        h2, w2 = topdown_display.shape[:2]
        
        target_height = max(h1, h2)
        scale1 = target_height / h1
        scale2 = target_height / h2
        
        camera_resized = cv2.resize(camera_display, (int(w1 * scale1), target_height))
        topdown_resized = cv2.resize(topdown_display, (int(w2 * scale2), target_height))
        
        # Combine side-by-side
        combined = np.hstack([camera_resized, topdown_resized])
        
        # Display
        window_name = 'ROI Transformation Visualization'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, combined)
        
        print(f"\nDisplaying {fname}")
        print(f"Green = Empty spots, Red = Occupied spots")
        print(f"Total ROIs: {len(camera_rois)}")
        print(f"Occupied: {sum(occupancy)}, Empty: {len(occupancy) - sum(occupancy)}")
        print("\nPress any key to close...")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if output_file:
            cv2.imwrite(output_file, combined)
            print(f"✓ Saved visualization to {output_file}")
        
        return combined
    
    def save_transformed_annotations(self, annotations_file, output_file='annotations_topdown.json'):
        """
        Transform annotations and save to a new file.
        
        Args:
            annotations_file: Path to original annotations.json
            output_file: Path to save transformed annotations
        """
        transformed_data = self.load_and_transform_annotations(annotations_file)
        
        # Create output in the same format, but with top-down ROIs
        # Normalize the top-down ROIs to 0-1 range
        topdown_h, topdown_w = self.topdown_shape[:2]
        
        output_data = {
            'file_names': transformed_data['file_names'],
            'rois_list': [],
            'occupancy_list': transformed_data['occupancy_list']
        }
        
        for topdown_rois in transformed_data['rois_list_topdown']:
            normalized_rois = []
            for roi in topdown_rois:
                normalized_roi = [[x / topdown_w, y / topdown_h] for x, y in roi]
                normalized_rois.append(normalized_roi)
            output_data['rois_list'].append(normalized_rois)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved transformed annotations to {output_file}")
        return output_data


def demo_usage():
    """Demonstrate how to use the HomographyTransformer."""
    print("\n" + "="*70)
    print("HOMOGRAPHY TRANSFORMER DEMO")
    print("="*70 + "\n")
    
    # Load the transformer
    try:
        transformer = HomographyTransformer('homography_matrix.pkl')
    except FileNotFoundError:
        print("Error: homography_matrix.pkl not found!")
        print("Please run the calibration script first to generate the homography matrix.")
        return
    
    print("\n" + "-"*70)
    print("EXAMPLE 1: Transform a single point")
    print("-"*70)
    
    # Example: Transform a point from camera view to top-down view
    camera_point = [320, 240]
    topdown_point = transformer.transform_point(camera_point)
    print(f"Camera point: {camera_point}")
    print(f"Top-down point: [{topdown_point[0]:.2f}, {topdown_point[1]:.2f}]")
    
    print("\n" + "-"*70)
    print("EXAMPLE 2: Transform a polygon (e.g., a parking spot)")
    print("-"*70)
    
    # Example: Transform a polygon (parking spot)
    parking_spot_camera = np.array([
        [300, 200],
        [400, 210],
        [410, 280],
        [310, 270]
    ])
    
    parking_spot_topdown = transformer.transform_polygon(parking_spot_camera)
    
    print("Camera view polygon:")
    print(parking_spot_camera)
    print("\nTop-down view polygon:")
    print(parking_spot_topdown.astype(int))
    
    # Calculate area ratio
    area_ratio = transformer.get_polygon_area_ratio(parking_spot_camera)
    print(f"\nArea ratio (top-down/camera): {area_ratio:.2f}x")
    
    print("\n" + "-"*70)
    print("EXAMPLE 3: Transform a bounding box")
    print("-"*70)
    
    # Example: Transform a bounding box
    bbox_camera = [100, 150, 80, 120]  # (x, y, width, height)
    bbox_corners_topdown = transformer.transform_bounding_box(bbox_camera)
    
    print(f"Camera bounding box: x={bbox_camera[0]}, y={bbox_camera[1]}, " +
          f"w={bbox_camera[2]}, h={bbox_camera[3]}")
    print("Top-down corners:")
    print(bbox_corners_topdown.astype(int))
    
    print("\n" + "-"*70)
    print("EXAMPLE 4: Batch transform multiple points")
    print("-"*70)
    
    # Example: Transform multiple points at once
    camera_points = np.array([
        [100, 100],
        [200, 150],
        [300, 200],
        [400, 250]
    ])
    
    topdown_points = transformer.transform_points(camera_points)
    
    print("Camera points:")
    print(camera_points)
    print("\nTop-down points:")
    print(topdown_points.astype(int))
    
    print("\n" + "="*70)
    print("USAGE IN YOUR CODE:")
    print("="*70)
    print("""
from parking_lot_transform import HomographyTransformer

# Load the transformer
transformer = HomographyTransformer('homography_matrix.pkl')

# Transform a point
topdown_point = transformer.transform_point([x, y])

# Transform a polygon
topdown_polygon = transformer.transform_polygon(camera_polygon)

# Transform a bounding box
topdown_corners = transformer.transform_bounding_box([x, y, w, h])

# Warp entire image (if needed)
topdown_image = transformer.warp_image(camera_image)

# Transform annotations from annotations.json
transformed = transformer.load_and_transform_annotations('annotations.json')

# Visualize transformed ROIs
transformer.visualize_transformed_annotations('annotations.json', 'camera_image.jpg')
    """)
    print("="*70)
    print("\nCOMMAND LINE USAGE:")
    print("="*70)
    print("""
# View transformed annotations
python template_transform.py --annotations annotations.json --image camera_image.jpg

# Save transformed annotations to a new file
python template_transform.py --annotations annotations.json --save-annotations topdown_annotations.json

# Save visualization to file
python template_transform.py --annotations annotations.json --image camera_image.jpg --save-viz output.jpg
    """)
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Apply saved homography transformation to points and polygons'
    )
    parser.add_argument('--demo', action='store_true', 
                       help='Run demonstration of transformer usage')
    parser.add_argument('--matrix', default='homography_matrix.pkl',
                       help='Path to saved homography matrix (default: homography_matrix.pkl)')
    parser.add_argument('--point', nargs=2, type=float, metavar=('X', 'Y'),
                       help='Transform a single point (x, y) from camera to top-down view')
    parser.add_argument('--polygon', type=str,
                       help='Transform a polygon given as comma-separated coordinates: x1,y1,x2,y2,...')
    parser.add_argument('--annotations', type=str,
                       help='Path to annotations.json file to transform and visualize')
    parser.add_argument('--image', type=str,
                       help='Path to camera image to visualize with annotations')
    parser.add_argument('--save-annotations', type=str,
                       help='Save transformed annotations to this file (default: annotations_topdown.json)')
    parser.add_argument('--save-viz', type=str,
                       help='Save visualization image to this file')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_usage()
        return 0
    
    try:
        transformer = HomographyTransformer(args.matrix)
        
        # Handle annotations transformation and visualization
        if args.annotations:
            if args.image:
                # Visualize annotations on an image
                transformer.visualize_transformed_annotations(
                    args.annotations, 
                    args.image,
                    args.save_viz
                )
            else:
                # Just transform and optionally save
                transformed = transformer.load_and_transform_annotations(args.annotations)
                print(f"\n✓ Loaded and transformed annotations")
                print(f"  Images: {len(transformed['file_names'])}")
                print(f"  ROIs per image: {len(transformed['rois_list_topdown'][0])}")
                
                if args.save_annotations:
                    transformer.save_transformed_annotations(
                        args.annotations,
                        args.save_annotations
                    )
                elif not args.image:
                    print("\nUse --image to visualize, or --save-annotations to save transformed ROIs")
        
        # Handle single point transformation
        elif args.point:
            result = transformer.transform_point(args.point)
            print(f"Camera point: ({args.point[0]}, {args.point[1]})")
            print(f"Top-down point: ({result[0]:.2f}, {result[1]:.2f})")
        
        # Handle polygon transformation
        elif args.polygon:
            coords = list(map(float, args.polygon.split(',')))
            if len(coords) % 2 != 0:
                print("Error: Polygon coordinates must be pairs of x,y values")
                return 1
            
            points = np.array(coords).reshape(-1, 2)
            result = transformer.transform_polygon(points)
            
            print("Camera polygon:")
            print(points)
            print("\nTop-down polygon:")
            print(result)
        
        else:
            print("Use --demo to see usage examples")
            print("Use --annotations <file> --image <file> to visualize transformed ROIs")
            print("Use --help for all options")
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Run the calibration script first to generate the homography matrix.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())