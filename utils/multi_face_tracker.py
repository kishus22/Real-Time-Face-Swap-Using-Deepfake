import numpy as np
from collections import OrderedDict

class FaceTracker:
    def __init__(self, max_faces=10, tracking_sensitivity=0.85):
        self.max_faces = max_faces
        self.tracking_sensitivity = tracking_sensitivity
        self.tracked_faces = OrderedDict()
        self.next_id = 0
        
    def update(self, current_faces):
        """Update tracked faces with new detections"""
        if len(current_faces) == 0:
            self.tracked_faces.clear()
            return []
        
        # Initialize tracked faces if empty
        if len(self.tracked_faces) == 0:
            for face in current_faces[:self.max_faces]:
                self._add_face(face)
            return list(self.tracked_faces.values())
        
        # Calculate similarity matrix (IOU + embedding distance)
        face_ids = list(self.tracked_faces.keys())
        similarities = np.zeros((len(face_ids), len(current_faces)))
        
        for i, (fid, old_face) in enumerate(self.tracked_faces.items()):
            for j, new_face in enumerate(current_faces):
                similarities[i, j] = self._face_similarity(old_face, new_face)
        
        # Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-similarities)
        
        updated_faces = {}
        used_new_faces = set()
        
        # Update existing faces
        for row, col in zip(row_ind, col_ind):
            if similarities[row, col] > self.tracking_sensitivity:
                fid = face_ids[row]
                updated_faces[fid] = current_faces[col]
                used_new_faces.add(col)
        
        # Add new faces
        for j, face in enumerate(current_faces):
            if j not in used_new_faces and len(updated_faces) < self.max_faces:
                fid = self._add_face(face)
                updated_faces[fid] = face
        
        # Update tracked faces
        self.tracked_faces = OrderedDict(sorted(updated_faces.items()))
        return list(self.tracked_faces.values())
    
    def _face_similarity(self, face1, face2):
        """Calculate similarity between two faces (0-1)"""
        # Bounding box IOU
        bbox1 = face1.bbox
        bbox2 = face2.bbox
        
        # Calculate intersection area
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        
        # Calculate union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - inter_area
        
        iou = inter_area / union_area
        
        # Embedding distance (if available)
        if hasattr(face1, 'embedding') and hasattr(face2, 'embedding'):
            emb_dist = np.dot(face1.embedding, face2.embedding)
        else:
            emb_dist = 0.5  # Neutral value if no embeddings
            
        # Combined similarity score
        return 0.6 * iou + 0.4 * emb_dist
    
    def _add_face(self, face):
        """Add a new face to tracking"""
        fid = self.next_id
        self.tracked_faces[fid] = face
        self.next_id += 1
        return fid