import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

class FaceEnhancer:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() and config['performance']['use_gpu'] else 'cpu'
        self.model = self._load_model()
        
    def _load_model(self):
        """Load face enhancement model (GFPGAN or similar)"""
        try:
            from gfpgan import GFPGANer
            model = GFPGANer(
                model_path='models/gfpgan/GFPGANv1.4.pth',
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            return model
        except ImportError:
            print("GFPGAN not available, face enhancement disabled")
            return None
        except Exception as e:
            print(f"Failed to load face enhancer: {e}")
            return None
    
    def enhance_face(self, face_image):
        """Enhance face image quality"""
        if self.model is None:
            return face_image
            
        try:
            # Convert face image to RGB
            if face_image.shape[2] == 4:  # RGBA image
                face_image = face_image[:, :, :3]
            
            # Normalize image
            face_image = face_image.astype(np.float32)[:, :, ::-1]  # BGR to RGB
            face_image = face_image / 255.0
            
            # Enhance face
            _, _, output = self.model.enhance(
                face_image,
                has_aligned=True,
                only_center_face=False,
                paste_back=True
            )
            
            # Convert back to BGR
            output = (output * 255).astype(np.uint8)
            output = output[:, :, ::-1]  # RGB to BGR
            
            return output
        except Exception as e:
            print(f"Face enhancement failed: {e}")
            return face_image