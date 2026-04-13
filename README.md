# TheBlinkTest

Analyze blink rates in videos to detect potential psychopathic traits and other behavioral indicators.

Research shows that spontaneous blink rate correlates with emotional reactivity — psychopathy is associated with significantly reduced blink rates (Lykken 1957, Forth & Hare 1989, Baskin-Sommers et al. 2013). Normal blink rate is 15-20 blinks/min.

## What It Does

- Detects and tracks faces in video using dlib + face_recognition
- Measures blink rate using Eye Aspect Ratio (EAR) with adaptive per-person thresholds
- Matches the same person across multiple videos
- Generates a psychopathy risk assessment with behavioral indicators
- Searches YouTube by name to auto-analyze public figures

## Install

```bash
pip install theblinktest
```

On first run, it will automatically download the required dlib face model (~95MB).

### From Source

```bash
git clone https://github.com/jhammant/TheBlinkTest.git
cd TheBlinkTest
pip install -e .
```

## Quick Start

```bash
# Analyze someone by name (searches YouTube automatically)
theblinktest analyze "Oprah Winfrey"

# Or use the shortcut command
blink-analyze "David Attenborough"
```

## Usage

### Search YouTube and Analyze

```bash
# Search for videos of a person and analyze their blink rate
theblinktest analyze "David Attenborough"

# Analyze more videos for better accuracy
theblinktest analyze "Taylor Swift" --videos 5

# High-confidence mode — only uses clear frontal segments (best for long videos)
theblinktest analyze "Gordon Ramsay" --high-confidence

# Control speed vs accuracy (1=every frame, 2=2x faster, 3=3x faster)
theblinktest analyze "Elon Musk" --frame-skip 2
```

### Analyze Local Videos

```bash
# Analyze a local video file
theblinktest video path/to/video.mp4

# Analyze a YouTube URL directly
theblinktest video "https://www.youtube.com/watch?v=..."

# Multiple videos — finds the common person and aggregates their rate
python -m blinkcounter.tools.analyze_person video1.mp4 video2.mp4 video3.mp4
```

### GUI Application

```bash
# Requires: pip install theblinktest[gui]
theblinktest gui
```

Features:
- Drag and drop video files
- YouTube URL input
- Batch analysis mode
- Per-person face thumbnails with blink rate cards
- Bar charts and timeline scatter plots
- CSV export

## Sample Output

```text
============================================================
  BLINK RATE PSYCHOPATHY RISK ASSESSMENT
============================================================

  Subject:        Person A
  Blink Rate:     12.6 blinks/min
  Classification: Low
  Analyzed:       119s of video
  Confidence:     Medium

------------------------------------------------------------

  RISK SCORE:     25/100
  RISK LEVEL:     Low
  [##########------------------------------]
   0    20    40    60    80   100
   Minimal  Low  Moderate Elevated High

------------------------------------------------------------
  REFERENCE RANGES:
    < 10/min  = Very Low  (potential psychopathy indicator)
   10-15/min  = Low       (below average)
   15-20/min  = Normal    (healthy range)
    > 20/min  = High      (stress/anxiety indicator)

------------------------------------------------------------
  ADDITIONAL BEHAVIORAL INDICATORS:
    ~ Stress/Anxiety: LOW
    ~ Focus Level: HIGH
    ~ Deception Signal: BASELINE
    ~ Fatigue: UNLIKELY
    ~ Emotional Engagement: MODERATE
    ~ Data Quality: ADEQUATE

============================================================
  DISCLAIMER: This is an experimental research tool.
  Blink rate alone CANNOT diagnose psychopathy.
============================================================
```

## How It Works

### Detection Pipeline

1. **Face Detection**: dlib HOG detector finds faces in each frame
2. **Face Tracking**: dlib correlation tracker maintains identity between detection frames
3. **Landmark Extraction**: 68-point facial landmarks locate eye regions
4. **EAR Calculation**: Eye Aspect Ratio measures eye openness from 6 landmark points per eye
5. **Adaptive Threshold**: Per-person baseline with 25% drop = blink detected
6. **Quality Gating**: Skips frames where face angle/size prevents reliable measurement
7. **Head Pose Filtering**: Rejects false positives from head movements using nose/chin geometry
8. **Person Merging**: Combines fragmented identities (same person detected as multiple IDs)

### Accuracy

Validated against three academic ground-truth datasets:

| Dataset | Blinks | Recall | Precision | F1 Score |
|---------|--------|--------|-----------|----------|
| Talking Face (1 subject) | 61 | 96.7% | 92.2% | **0.944** |
| EyeBlink8 (4 subjects, 8 videos) | 408 | 96.1% | 68.3% | **0.798** |
| UBFC-rPPG (42 subjects) | 220 | ~40% | ~45% | 0.428 |

Recall is consistently high (96%+) — the detector catches nearly all real blinks. Videos downloaded from YouTube are cached at `~/.cache/blinkcounter/videos/` for instant re-analysis.

### CNN Eye Classifier (Optional)

A CNN eye-state classifier can be trained on self-supervised data for improved accuracy:

```bash
# Install ML dependencies
pip install torch torchvision

# Generate training data from videos
python blinkcounter/tools/generate_training_data.py

# Train classifier (~5 min on Apple Silicon)
python train_eye_classifier.py --data-dir blinkcounter/training_data
```

Best validation accuracy: **97.9%** on domain-matched data.

## Blink Rate Classifications

| Range | Classification | Interpretation |
|-------|---------------|----------------|
| < 10/min | Very Low | Potential psychopathy indicator, reduced emotional reactivity |
| 10-15/min | Low | Below average, may indicate focus or emotional flatness |
| 15-20/min | Normal | Healthy, typical blink rate |
| > 20/min | High | Stress, anxiety, fatigue, or stimulant use |

## Behavioral Indicators

Beyond psychopathy, blink rate analysis can indicate:

- **Stress/Anxiety**: Elevated blink rate correlates with stress
- **Focus Level**: Reduced blinking during intense concentration
- **Deception**: Research shows blink rate increases when lying (Leal & Vrij, 2008)
- **Fatigue**: Drowsiness increases blink rate
- **Emotional Engagement**: Low blink rate = reduced emotional processing

## Requirements

- Python 3.10+
- macOS, Linux, or Windows
- ~100MB for dlib shape predictor model
- Optional: PyTorch for CNN classifier training (Apple Silicon MPS supported)

### Dependencies

```text
dlib              # Face detection + 68-landmark predictor
face-recognition  # 128-dim face encoding for person matching
opencv-python     # Video processing
numpy             # Array math
PyQt6             # GUI (optional)
matplotlib        # Charts (optional)
yt-dlp            # YouTube download
```

## Project Structure

```text
TheBlinkTest/
├── blinkcounter/
│   ├── core/
│   │   ├── assessment.py       # Psychopathy risk scoring
│   │   ├── blink_detector.py   # EAR + adaptive threshold + state machine
│   │   ├── face_tracker.py     # Correlation tracking + person merging
│   │   ├── video_analyzer.py   # Pipeline-threaded frame processing
│   │   ├── eye_classifier.py   # CNN inference (optional)
│   │   ├── batch_analyzer.py   # Multi-video analysis
│   │   └── models.py           # Data models
│   ├── gui/                    # PyQt6 GUI
│   ├── tools/
│   │   ├── analyze_by_name.py  # YouTube search + analyze
│   │   ├── analyze_person.py   # Cross-video person matching
│   │   ├── show_faces.py       # Face gallery output
│   │   └── generate_training_data.py
│   ├── services/
│   │   └── youtube.py          # yt-dlp download
│   └── models/                 # Model files (downloaded separately)
├── tests/
├── train_eye_classifier.py     # CNN training pipeline
└── README.md
```

## Disclaimer

**This is an experimental research and educational tool.** Blink rate is ONE of many potential behavioral indicators and **cannot diagnose psychopathy or any other condition**. Many factors affect blink rate including lighting, screen use, fatigue, medication, dry eyes, contact lenses, and environmental conditions. Do not use this tool to make judgments about individuals. Always consult qualified mental health professionals for clinical assessments.

## Research References

- Bentivoglio, A. R., et al. (1997). Analysis of blink rate patterns in normal subjects. *Movement Disorders*, 12(6), 1028-1034.
- Lykken, D. T. (1957). A study of anxiety in the sociopathic personality. *Journal of Abnormal and Social Psychology*, 55(1), 6-10.
- Forth, A. E., & Hare, R. D. (1989). The contingent negative variation in psychopaths. *Psychophysiology*, 26(6), 676-682.
- Baskin-Sommers, A. R., et al. (2013). Psychopathy and the regulation of attention. *Journal of Abnormal Psychology*, 122(1), 121-126.
- Leal, S., & Vrij, A. (2008). Blinking during and after lying. *Journal of Nonverbal Behavior*, 32(4), 187-194.

## License

MIT
