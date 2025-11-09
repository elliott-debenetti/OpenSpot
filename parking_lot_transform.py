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
    
    args = parser.parse_args()
    
    if args.demo:
        demo_usage()
        return 0
    
    try:
        transformer = HomographyTransformer(args.matrix)
        
        if args.point:
            result = transformer.transform_point(args.point)
            print(f"Camera point: ({args.point[0]}, {args.point[1]})")
            print(f"Top-down point: ({result[0]:.2f}, {result[1]:.2f})")
        
        if args.polygon:
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
        
        if not args.point and not args.polygon:
            print("Use --demo to see usage examples, or --help for options")
    
    except FileNotFoundError:
        print(f"Error: Could not find {args.matrix}")
        print("Run the calibration script first to generate the homography matrix.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
