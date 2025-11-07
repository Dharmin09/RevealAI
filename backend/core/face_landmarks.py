"""Facial landmark detection and manipulation-prone region identification."""
import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp_face_mesh = None
    MEDIAPIPE_AVAILABLE = False


# Key facial landmarks prone to manipulation
FACIAL_REGIONS = {
    'lips': {
        'landmarks': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 347, 0, 267, 13, 14, 15, 16, 37, 39, 40, 41, 42, 183, 184, 185, 186, 71, 72, 73, 74, 75, 150, 149, 148, 176, 177, 400, 401, 402],
        'name': 'Lips',
        'color': (0, 0, 255),  # Red
    },
    'eyes': {
        'landmarks': [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173, 263, 249, 390, 373, 374, 380, 381, 382, 362],
        'name': 'Eyes',
        'color': (0, 255, 0),  # Green
    },
    'mouth': {
        'landmarks': [11, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
        'name': 'Mouth',
        'color': (255, 0, 0),  # Blue
    },
    'chin': {
        'landmarks': [165, 164, 235, 454, 453],
        'name': 'Chin',
        'color': (255, 165, 0),  # Orange
    },
    'nose': {
        'landmarks': [1, 2, 3, 4, 5, 6, 48, 51, 57, 43, 46, 131, 134, 98, 97, 2, 326, 327, 294, 293, 331, 375, 279],
        'name': 'Nose',
        'color': (255, 255, 0),  # Cyan
    },
}


def detect_facial_landmarks(frame_rgb, scale=0.5):
    """Detect facial landmarks using MediaPipe Face Mesh.
    
    Args:
        frame_rgb: RGB frame (numpy array)
        scale: Scale factor for detection (0.5 = half resolution for speed)
    
    Returns:
        dict with 'landmarks' (dict of region->points), 'h', 'w', or None if detection fails
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    try:
        h, w = frame_rgb.shape[:2]
        
        # Scale frame down for faster detection
        if scale != 1.0:
            scaled_frame = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
            scale_h, scale_w = scaled_frame.shape[:2]
        else:
            scaled_frame = frame_rgb
            scale_h, scale_w = h, w
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(scaled_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0]
            
            # Extract all landmarks
            all_landmarks = []
            for lm in landmarks.landmark:
                # Scale back to original resolution
                x = int(lm.x * scale_w / scale)
                y = int(lm.y * scale_h / scale)
                all_landmarks.append((x, y))
            
            # Extract facial regions
            regions_dict = {}
            for region_name, region_data in FACIAL_REGIONS.items():
                region_landmarks = []
                for idx in region_data['landmarks']:
                    if idx < len(all_landmarks):
                        region_landmarks.append(all_landmarks[idx])
                if region_landmarks:
                    regions_dict[region_name] = np.array(region_landmarks, dtype=np.int32)
            
            return {
                'landmarks': regions_dict,
                'h': h,
                'w': w,
            }
    
    except Exception as e:
        print(f"[LANDMARK] Detection error: {e}")
        return None


def create_facial_region_mask(frame_shape, landmarks_dict, blur_sigma=15):
    """Create a mask highlighting manipulation-prone facial regions.
    
    Args:
        frame_shape: (height, width) of frame
        landmarks_dict: Dict from detect_facial_landmarks()
        blur_sigma: Gaussian blur sigma for soft edges
    
    Returns:
        2D float array (0-1) with high values on facial regions
    """
    if landmarks_dict is None:
        return np.ones(frame_shape[:2], dtype=np.float32)
    
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    regions = landmarks_dict.get('landmarks', {})
    
    # Create mask for each region with slight overlap padding
    for region_name, points in regions.items():
        if points is not None and len(points) > 2:
            # Create convex hull for smooth region boundaries
            hull = cv2.convexHull(points)
            # Draw filled polygon
            cv2.fillPoly(mask, [hull], 1.0, lineType=cv2.LINE_AA)
    
    # Apply Gaussian blur to create soft transitions
    if blur_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    
    return np.clip(mask, 0.0, 1.0)


def get_facial_region_emphasis(frame_rgb, blur_sigma=15):
    """Get a mask emphasizing manipulation-prone facial regions.
    
    Returns:
        2D float array (0-1) with high values where manipulation is likely
    """
    landmarks = detect_facial_landmarks(frame_rgb)
    if landmarks is None:
        # Fallback: emphasize center region (face typically centered)
        h, w = frame_rgb.shape[:2]
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_dist = np.sqrt(cy ** 2 + cx ** 2)
        mask = 1.0 - (dist / max_dist) ** 2
        return np.clip(mask, 0.0, 1.0)
    
    return create_facial_region_mask(frame_rgb.shape, landmarks, blur_sigma)
