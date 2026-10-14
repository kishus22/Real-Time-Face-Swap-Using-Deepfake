import cv2
import argparse
from face_swapper import FaceSwapper

def main():
    parser = argparse.ArgumentParser(description="Real-time Face Swap")
    parser.add_argument("--target", type=str, required=True, 
                       help="Path to target face image")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save output video")
    parser.add_argument("--show", action="store_true",
                       help="Show real-time preview")
    args = parser.parse_args()
    
    # Initialize face swapper
    try:
        swapper = FaceSwapper(args.target)
        print("Face swapper initialized successfully")
    except Exception as e:
        print(f"Failed to initialize face swapper: {e}")
        return
    
    # Open video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting face swap. Press 'q' to quit...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Mirror the frame for more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = swapper.face_analyzer.get(frame)
            
            if len(faces) > 0:
                # Swap the first detected face
                frame = swapper.swap_face(frame, faces[0])
            
            # Display the resulting frame
            if args.show:
                cv2.imshow('Face Swap', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\nStopping face swap...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Face swap stopped")

if __name__ == "__main__":
    main()