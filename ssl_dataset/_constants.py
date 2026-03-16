# ssl_dataset/_constants.py
#
# Single source of truth for the entire ssl_dataset package.
# Every other file imports from here — no magic numbers anywhere else.

# ---------------------------------------------------------------------------
# HuggingFace repository identifiers
# ---------------------------------------------------------------------------
# Each sub-library has its own HuggingFace repo so they can be downloaded
# independently. Users who only need landmarks don't have to download skeleton
# images, and vice versa.

HF_AUTHOR = "Jayasha"

HF_REPO_LANDMARKS   = f"{HF_AUTHOR}/ssl-landmarks"
HF_REPO_SKELETON    = f"{HF_AUTHOR}/ssl-skeleton"
HF_REPO_PREPROCESSED = f"{HF_AUTHOR}/ssl-preprocessed"

# ---------------------------------------------------------------------------
# Dataset dimensions — never hardcode these anywhere else
# ---------------------------------------------------------------------------

NUM_CLASSES        = 55   # total sign classes (0–54)
NUM_STATIC_CLASSES = 37   # classes 0–36  (single images)
NUM_DYNAMIC_CLASSES = 18  # classes 37–54 (video sequences)

STATIC_CLASS_IDS  = list(range(0, 37))   # [0, 1, ..., 36]
DYNAMIC_CLASS_IDS = list(range(37, 55))  # [37, 38, ..., 54]

SAMPLES_PER_STATIC_CLASS  = 100  # images per static sign
SAMPLES_PER_DYNAMIC_CLASS = 20   # videos per dynamic sign

NUM_FRAMES    = 30   # frames per video (static signs are replicated to 30)
NUM_LANDMARKS = 21   # MediaPipe hand landmarks per frame
NUM_COORDS    = 3    # x, y, z per landmark

# Flattened feature sizes (used by MLP and preprocessed sub-library)
SKELETON_IMAGE_SIZE     = 28          # each skeleton image is 28×28 pixels
SKELETON_IMAGE_CHANNELS = 3           # RGB
SKELETON_FLAT_SIZE = (
    NUM_FRAMES
    * SKELETON_IMAGE_SIZE
    * SKELETON_IMAGE_SIZE
    * SKELETON_IMAGE_CHANNELS
)  # = 30 × 28 × 28 × 3 = 70,560  (MLP input)

LANDMARK_FLAT_SIZE = NUM_FRAMES * NUM_LANDMARKS * NUM_COORDS
# = 30 × 21 × 3 = 1,890  (LSTM/Transformer input per sample)

# ---------------------------------------------------------------------------
# Train / validation / test split ratios
# These match the exact ratios used in the original thesis so that anyone
# reproducing the research gets the same splits.
# ---------------------------------------------------------------------------

SPLIT_TRAIN = 0.70
SPLIT_VAL   = 0.15
SPLIT_TEST  = 0.15
SPLIT_SEED  = 42   # fixed seed → reproducible splits every time

# ---------------------------------------------------------------------------
# Class label map  (class index → Sinhala sign label)
# ---------------------------------------------------------------------------
# Static signs (0–36): vowels, consonants, numbers, one phrase
# Dynamic signs (37–54): additional vowels, consonants, words/phrases

CLASS_LABELS = {
    # --- Static signs (single images) ---
    0:  'අ',
    1:  'ආ',
    2:  'ඇ',
    3:  'ඉ',
    4:  'ඊ',
    5:  'උ',
    6:  'එ',
    7:  'ඒ',
    8:  'ක්',
    9:  'ග්',
    10: 'ට්',
    11: 'ද්',
    12: 'ත්',
    13: 'ඩ්',
    14: 'න්',
    15: 'ප්',
    16: 'බ්',
    17: 'ම්',
    18: 'ය්',
    19: 'ර්',
    20: 'ල්',
    21: 'ව්',
    22: 'ස්',
    23: 'හ්',
    24: 'ං',
    25: 'ච්',
    26: 'ෆ',
    27: '1',
    28: '2',
    29: '3',
    30: '4',
    31: '5',
    32: '6',
    33: '7',
    34: '8',
    35: '9',
    36: 'සුභ',
    # --- Dynamic signs (video sequences) ---
    37: 'ඈ',
    38: 'ඔ',
    39: 'ඕ',
    40: 'ජ',
    41: 'ණ',
    42: 'ළ',
    43: 'ඟ',
    44: 'ඳ',
    45: 'ඬ',
    46: '10',
    47: 'උදෑසනක්',
    48: 'රාත්‍රියක්',
    49: 'සන්ධ්‍යාවක්',
    50: 'අම්මා',
    51: 'තාත්තා',
    52: 'මම',
    53: 'මට',
    54: 'මගේ',
}

# Reverse lookup: label string → class index
LABEL_TO_CLASS = {v: k for k, v in CLASS_LABELS.items()}

# ASCII-safe slugs for file system and HuggingFace paths.
# We cannot use Sinhala characters directly in folder/file names on all OSes,
# so every class gets a stable ASCII slug: "class_000", "class_001", etc.
# The slug is zero-padded to 3 digits so alphabetical sort = numerical sort.
CLASS_SLUGS = {i: f"class_{i:03d}" for i in range(NUM_CLASSES)}

# ---------------------------------------------------------------------------
# Local cache directory
# ---------------------------------------------------------------------------
# Downloaded files are cached here so they are not re-downloaded on every use.
# Using pathlib.Path ensures this works on Windows, macOS, and Linux.

import pathlib

CACHE_DIR_ROOT       = pathlib.Path.home() / ".cache" / "ssl_dataset"
CACHE_DIR_LANDMARKS  = CACHE_DIR_ROOT / "landmarks"
CACHE_DIR_SKELETON   = CACHE_DIR_ROOT / "skeleton"
CACHE_DIR_PREPROCESSED = CACHE_DIR_ROOT / "preprocessed"
