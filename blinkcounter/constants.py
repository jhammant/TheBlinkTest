"""Constants for blink detection and classification."""

# dlib 68-landmark eye indices (used in face_tracker.py directly)
# Left eye: 36-41, Right eye: 42-47

# Eye Aspect Ratio (EAR) thresholds
EAR_BLINK_THRESHOLD = 0.22
EAR_HYSTERESIS = 0.03  # Band above threshold to confirm eye is open again

# Blink timing constraints
MIN_BLINK_DURATION_MS = 50   # Below = noise
MAX_BLINK_DURATION_MS = 400  # Above = voluntary eye closure

# Consecutive frames below threshold required to register a blink
CONSECUTIVE_FRAMES_FOR_BLINK = 2  # Requires 2+ frames below threshold (prevents head-movement false positives)

# Video processing
VIDEO_FRAME_SKIP = 3  # Process every Nth frame (at 30fps = 10 samples/sec)

# Face re-identification
FACE_MATCH_TOLERANCE = 0.7  # face_recognition encoding distance threshold (higher = more permissive)
FACE_ENCODING_UPDATE_INTERVAL = 30  # Re-encode faces every N processed frames

# Blink rate classification (blinks per minute)
CLASSIFICATION_VERY_LOW_MAX = 10
CLASSIFICATION_LOW_MAX = 15
CLASSIFICATION_NORMAL_MAX = 20
# Above NORMAL_MAX = High

# Colors for classifications (R, G, B)
COLOR_VERY_LOW = (220, 53, 69)    # Red
COLOR_LOW = (255, 165, 0)         # Orange
COLOR_NORMAL = (40, 167, 69)      # Green
COLOR_HIGH = (0, 123, 255)        # Blue

# Person labels
PERSON_LABELS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]  # A-Z
