import os
import numpy as np

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR  = os.path.join(SCRIPT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train_data")
TEST_DIR  = os.path.join(DATA_DIR, "test_data")

def list_train_subjects():
    """Return a sorted list of subject folder names in data/train_data/."""
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"train_data folder not found at: {TRAIN_DIR}")

    subjects = [
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ]
    return sorted(subjects)

def load_train_subject_roi(subject, roi_tag=None):
    """
    Load training image IDs and fMRI data for a given subject.

    Current behavior:
      - Loads lh_training_fmri.npy and rh_training_fmri.npy
      - Concatenates hemispheres along voxel dimension
      - Derives image IDs from training_images filenames by parsing the NSD ID

    Directory structure assumed:

      data/train_data/<subject>/<inner_subj>/
        training_split/
          training_fmri/
            lh_training_fmri.npy
            rh_training_fmri.npy
          training_images/
            train-0001_nsd-00013.png
            ...

    Returns:
        img_ids: (N_train,) array of NSD image IDs (integers)
        fmri:    (N_train, N_voxels_total) array
    """

    # ---- Locate subject root and inner subject folder ----
    subj_root = os.path.join(TRAIN_DIR, subject)
    print(f"Subject root: {subj_root}")

    if not os.path.isdir(subj_root):
        raise FileNotFoundError(f"Subject folder not found: {subj_root}")

    inner_dirs = [
        d for d in os.listdir(subj_root)
        if os.path.isdir(os.path.join(subj_root, d))
    ]
    if not inner_dirs:
        raise RuntimeError(f"No inner subject folder found inside {subj_root}.")
    if len(inner_dirs) > 1:
        print("Warning: multiple inner directories found, using the first one:", inner_dirs)

    inner_subj = inner_dirs[0]
    subj_dir = os.path.join(subj_root, inner_subj)
    print(f"Using inner subject folder: {subj_dir}")

    # ---- Paths to training_fmri and training_images ----
    fmri_dir = os.path.join(subj_dir, "training_split", "training_fmri")
    img_dir  = os.path.join(subj_dir, "training_split", "training_images")

    lh_path = os.path.join(fmri_dir, "lh_training_fmri.npy")
    rh_path = os.path.join(fmri_dir, "rh_training_fmri.npy")

    if not os.path.isfile(lh_path) or not os.path.isfile(rh_path):
        raise FileNotFoundError(
            f"Could not find lh/rh training fMRI files at:\n  {lh_path}\n  {rh_path}"
        )

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"training_images folder not found at: {img_dir}")

    # ---- Load fMRI for both hemispheres ----
    lh_fmri = np.load(lh_path)
    rh_fmri = np.load(rh_path)

    print("Loaded hemisphere fMRI:")
    print("  LH shape:", lh_fmri.shape)
    print("  RH shape:", rh_fmri.shape)

    if lh_fmri.shape[0] != rh_fmri.shape[0]:
        raise RuntimeError(
            f"Mismatch in number of trials between LH and RH: "
            f"{lh_fmri.shape[0]} vs {rh_fmri.shape[0]}"
        )

    # Concatenate hemispheres along voxel dimension
    fmri = np.concatenate([lh_fmri, rh_fmri], axis=1)
    n_trials = fmri.shape[0]

    # ---- Derive image IDs from training_images filenames ----
    png_files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith(".png")
    )

    if not png_files:
        raise RuntimeError(f"No .png files found in training_images: {img_dir}")

    if len(png_files) != n_trials:
        print("Warning: number of images != number of fMRI trials.")
        print("  N images:", len(png_files))
        print("  N fMRI rows:", n_trials)

    # Parse NSD ID from filenames like: train-0001_nsd-00013.png
    img_ids = []
    for fname in png_files:
        try:
            # Split on 'nsd-' and strip extension
            nsd_part = fname.split("nsd-")[1]
            nsd_id_str = os.path.splitext(nsd_part)[0]
            nsd_id = int(nsd_id_str)
        except Exception as e:
            raise RuntimeError(
                f"Unexpected training image filename format: {fname}"
            ) from e
        img_ids.append(nsd_id)

    img_ids = np.array(img_ids, dtype=int)

    print("\nLoaded arrays:")
    print("  img_ids shape:", img_ids.shape)
    print("  fmri shape:", fmri.shape)

    return img_ids, fmri


if __name__ == "__main__":
    print("Looking inside train_data/ ...")
    print("TRAIN_DIR:", TRAIN_DIR)

    subjects = list_train_subjects()
    print("Found train subjects:")
    for s in subjects:
        print("  ", s)

    if not subjects:
        raise RuntimeError("No subject folders found in data/train_data/.")

    test_subject = subjects[0]
    print(f"\nTesting load for subject: {test_subject}")

    img_ids_train, fmri_train = load_train_subject_roi(subject=test_subject, roi_tag="VC")

    print("\nFinal check:")
    print("img_ids_train shape:", img_ids_train.shape)
    print("fmri_train shape:", fmri_train.shape)
