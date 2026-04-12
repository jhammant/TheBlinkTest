"""Constants for blink detection and classification."""

# MediaPipe Face Mesh eye landmark indices
# Each eye has 6 key points: outer corner, upper1, upper2, inner corner, lower1, lower2
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Eye Aspect Ratio (EAR) thresholds
EAR_BLINK_THRESHOLD = 0.21
EAR_HYSTERESIS = 0.02  # Band above threshold to confirm eye is open again

# Blink timing constraints
MIN_BLINK_DURATION_MS = 50   # Below = noise
MAX_BLINK_DURATION_MS = 400  # Above = voluntary eye closure

# Consecutive frames below threshold required to register a blink
CONSECUTIVE_FRAMES_FOR_BLINK = 2

# Video processing
VIDEO_FRAME_SKIP = 3  # Process every Nth frame (at 30fps = 10 samples/sec)

# Face re-identification
FACE_MATCH_TOLERANCE = 0.6  # face_recognition encoding distance threshold
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
