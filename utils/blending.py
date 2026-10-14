import cv2
import numpy as np
from typing import Tuple

class AdvancedBlender:
    def __init__(self, config):
        self.config = config
        self.face_parser = self._init_face_parser() if config['swapping']['mask_type'] == 'face_parsing' else None
    
    def _init_face_parser(self):
        try:
            from face_parsing import FaceParser
            return FaceParser()
        except ImportError:
            print("Face parsing model not available, falling back to simple mask")
            return None
    
    def create_dynamic_mask(self, face_img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Create advanced blending mask using face parsing or adaptive techniques"""
        if self.face_parser is not None:
            return self._create_parsing_mask(face_img)
        else:
            return self._create_adaptive_mask(face_img, landmarks)
    
    def _create_parsing_mask(self, face_img: np.ndarray) -> np.ndarray:
        """Create precise mask using face parsing"""
        parsing = self.face_parser.parse(face_img)
        mask = np.zeros_like(face_img)
        
        # Include face skin, nose, eyes, eyebrows, mouth
        for part in [1, 2, 3, 4, 5, 6, 10]:
            mask[parsing == part] = 255
        
        # Apply Gaussian blur for smooth edges
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask / 255.0
    
    def _create_adaptive_mask(self, face_img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Create mask based on face landmarks"""
        h, w = face_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Create convex hull from landmarks
        hull = cv2.convexHull(landmarks)
        cv2.fillConvexPoly(mask, hull, 1.0)
        
        # Expand mask slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Soften edges
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        mask = np.clip(mask * 1.2, 0, 1)
        
        return np.stack([mask]*3, axis=2)
    
    def apply_color_correction(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Match color statistics between source and target"""
        src_mean, src_std = cv2.meanStdDev(src)
        dst_mean, dst_std = cv2.meanStdDev(dst)
        
        src = src.astype(np.float32)
        dst = dst.astype(np.float32)
        
        # Color transfer
        src = (src - src_mean) * (dst_std / (src_std + 1e-6)) + dst_mean
        return np.clip(src, 0, 255).astype(np.uint8)
    
    def blend_faces(self, background: np.ndarray, face: np.ndarray, 
                   mask: np.ndarray, bbox: Tuple[int]) -> np.ndarray:
        """Advanced blending with edge refinement"""
        x1, y1, x2, y2 = bbox
        bg_region = background[y1:y2, x1:x2]
        
        # Resize face and mask to match target region
        face = cv2.resize(face, (x2-x1, y2-y1))
        mask = cv2.resize(mask, (x2-x1, y2-y1))
        
        # Apply color correction if enabled
        if self.config['swapping']['color_correction']:
            face = self.apply_color_correction(face, bg_region)
        
        # Sharpen if enabled
        if self.config['swapping']['sharpen']:
            face = self._sharpen_image(face)
        
        # Blend using the mask
        blended = bg_region * (1 - mask) + face * mask
        background[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        # Refine edges
        if self.config['swapping']['blend_edges']:
            background = self._blend_edges(background, bbox, mask)
        
        return background
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle sharpening"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def _blend_edges(self, image: np.ndarray, bbox: Tuple[int], mask: np.ndarray) -> np.ndarray:
        """Special edge blending to reduce visible seams"""
        x1, y1, x2, y2 = bbox
        edge_width = int(self.config['swapping']['blend_thickness'] * min(x2-x1, y2-y1))
        
        # Create edge region
        edge_mask = cv2.Canny((mask * 255).astype(np.uint8), 100, 200) / 255.0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width, edge_width))
        edge_region = cv2.dilate(edge_mask, kernel)
        
        # Apply Gaussian blur to edge region
        region = image[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(region, (edge_width*2+1, edge_width*2+1), 0)
        
        # Blend edge region
        edge_region = edge_region[..., np.newaxis]
        region = region * (1 - edge_region) + blurred * edge_region
        image[y1:y2, x1:x2] = region.astype(np.uint8)
        
        return image