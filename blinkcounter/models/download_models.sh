#!/bin/bash
# Download required model files for BlinkCounter
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Downloading dlib shape predictor (68 landmarks)..."
curl -L -o "$DIR/shape_predictor_68_face_landmarks.dat.bz2" \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 -f "$DIR/shape_predictor_68_face_landmarks.dat.bz2"

echo "Downloading MediaPipe face landmarker..."
curl -L -o "$DIR/face_landmarker.task" \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

echo "Downloading RT-BENE VGG16 blink model (pre-trained)..."
curl -L -o "$DIR/rt_bene_vgg16_allsubjects1.model" \
    "https://imperialcollegelondon.box.com/shared/static/wwky1um443vgz9oy90zllv0s7474a5dj.model"

echo "Done! Models downloaded to $DIR"
