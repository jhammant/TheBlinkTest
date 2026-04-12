#!/usr/bin/env bash
# Download RT-BENE eye image dataset for training the eye-state classifier.
#
# Downloads 3 subjects (~53 MB total) from https://zenodo.org/records/3685316
# to /tmp/blinkcounter_training/rt_bene/.
#
# Usage:
#   bash download_rt_bene_data.sh [--all]
#
# Pass --all to download all 17 subjects (~900 MB).

set -euo pipefail

DATA_DIR="/tmp/blinkcounter_training/rt_bene"
BASE_URL="https://zenodo.org/api/records/3685316/files"

# Subjects to download (small subset by default)
if [[ "${1:-}" == "--all" ]]; then
    SUBJECTS=(s000 s001 s002 s003 s004 s005 s006 s007 s008 s009 s010 s011 s012 s013 s014 s015 s016)
    echo "Downloading ALL 17 subjects..."
else
    SUBJECTS=(s007 s009 s012)
    echo "Downloading 3 subjects (pass --all for all 17)..."
fi

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Download subjects CSV
echo "Downloading rt_bene_subjects.csv..."
curl -sL "${BASE_URL}/rt_bene_subjects.csv/content" -o rt_bene_subjects.csv

for sid in "${SUBJECTS[@]}"; do
    # Download label CSV
    CSV_FILE="${sid}_blink_labels.csv"
    if [[ ! -f "$CSV_FILE" ]]; then
        echo "Downloading ${CSV_FILE}..."
        curl -sL "${BASE_URL}/${CSV_FILE}/content" -o "$CSV_FILE"
    else
        echo "Skipping ${CSV_FILE} (already exists)"
    fi

    # Download eye images tar
    TAR_FILE="${sid}_noglasses_eyes.tar"
    DIR_NAME="${sid}_noglasses"
    if [[ ! -d "$DIR_NAME" ]]; then
        if [[ ! -f "$TAR_FILE" ]]; then
            echo "Downloading ${TAR_FILE}..."
            curl -L "${BASE_URL}/${TAR_FILE}/content" -o "$TAR_FILE"
        fi
        echo "Extracting ${TAR_FILE}..."
        tar xf "$TAR_FILE"
        rm -f "$TAR_FILE"  # Clean up tar after extraction
    else
        echo "Skipping ${DIR_NAME} (already extracted)"
    fi
done

echo ""
echo "Done! Data is at: ${DATA_DIR}"
echo "Total images:"
find "$DATA_DIR" -name "*.png" | wc -l
