import os
import sys

import numpy as np
import pandas as pd

from memory.hmsg.utils.constants import COCO_STUFF_CLASSES, MATTERPORT_LABELS_160, MATTERPORT_LABELS_40, MATTERPORT_GT_LABELS, OPENVOCAB_MATTERPORT_LABELS
from memory.hmsg.utils.clip_utils import get_text_feats_multiple_templates


def compute_label_feats(
        clip_model,
        clip_feat_dim,
        label_feat_path,
        classes,
        pre_computed_feats_path="text_feats.npy"):
    """Either load precomputed label features or run clip to obtain those."""
    stem, ext = os.path.splitext(pre_computed_feats_path)
    dim_tagged_path = os.path.join(label_feat_path, f"{stem}_dim{clip_feat_dim}{ext}")
    default_path = os.path.join(label_feat_path, pre_computed_feats_path)

    for candidate in (dim_tagged_path, default_path):
        if os.path.exists(candidate) and ".npy" in candidate:
            try:
                loaded = np.load(candidate)
                if loaded.ndim == 2 and loaded.shape[0] == len(classes) and loaded.shape[1] == clip_feat_dim:
                    return loaded, classes
                print(
                    f"[label_feats] Ignoring {candidate} due to shape mismatch "
                    f"(got {loaded.shape}, expected ({len(classes)}, {clip_feat_dim}))."
                )
            except Exception as e:
                print(f"[label_feats] Could not load {candidate} ({e}); will recompute.")

    # Compute, store to dim-tagged cache (avoids clobbering LFS-tracked files), and return
    text_feats = get_text_feats_multiple_templates(
        classes, clip_model, clip_feat_dim)
    np.save(dim_tagged_path, text_feats)
    return text_feats, classes


def get_label_feats(
        clip_model,
        clip_feat_dim,
        obj_labels,
        label_feat_path=None):
    label_feat_path = os.path.dirname(os.path.abspath(__file__))
    if obj_labels == "COCO_STUFF_CLASSES":
        classes = list(COCO_STUFF_CLASSES.values())
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, classes, "text_feats_COCO_STUFF_CLASSES.npy")
    elif obj_labels == "MATTERPORT_LABELS_160":
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, MATTERPORT_LABELS_160, "text_feats_MATTERPORT_LABELS_160.npy")
    elif obj_labels == "MATTERPORT_LABELS_40":
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, MATTERPORT_LABELS_40, "text_feats_MATTERPORT_LABELS_40.npy")
    elif obj_labels == "MATTERPORT_GT_LABELS":
        classes = list(MATTERPORT_GT_LABELS.values())
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, classes, "text_feats_MATTERPORT_GT_LABELS.npy")
    elif obj_labels == "OPENVOCAB_MATTERPORT_LABELS":
        classes = list()
        for key, val in OPENVOCAB_MATTERPORT_LABELS.items():
            classes.append(key)
            classes.extend(val)
        classes = list(set(classes))
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, classes, "text_feats_OPENVOCAB_MATTERPORT_LABELS.npy")
    elif obj_labels == "HM3DSEM_LABELS":
        # TODO: change this
        label_feat_path = "memory/hmsg/labels"
        classes_matrix = pd.read_csv(
            os.path.join(
                label_feat_path,
                "HM3D_CountsOfObjectTypes.csv"),
            header=0,
            sep=";")
        classes = list(classes_matrix[classes_matrix.keys()[0]].values)
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, classes, "text_feats_HM3DSEM_LABELS.npy")
    elif obj_labels == "IMAGENET21K_LABELS":
        # TODO: change this
        label_feat_path = "memory/hmsg/labels"
        classes_matrix = pd.read_csv(
            os.path.join(
                label_feat_path,
                "imagenet21k.csv"),
            header=0,
            sep=";")
        classes = list(classes_matrix[classes_matrix.keys()[0]].values)
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, classes, "text_feats_IMAGENET21K_LABELS.npy")
    elif obj_labels == "SCANNET200":
        # TODO: change this
        label_feat_path = "memory/hmsg/labels"
        classes_matrix = pd.read_csv(
            os.path.join(
                label_feat_path,
                "scannet200.csv"),
            header=0,
            sep=";")
        classes = list(classes_matrix[classes_matrix.keys()[0]].values)
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, classes, "text_feats_SCANNET200_LABELS.npy")
    elif obj_labels == "SCANNET20":
        # TODO: change this
        label_feat_path = "memory/hmsg/labels"
        classes_matrix = pd.read_csv(
            os.path.join(
                label_feat_path,
                "scannet20.csv"),
            header=0,
            sep=";")
        classes = list(classes_matrix[classes_matrix.keys()[0]].values)
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, classes, "text_feats_SCANNET20_LABELS.npy")
    elif obj_labels == "FINALLABEL":
        # TODO: change this
        label_feat_path = "memory/hmsg/labels"
        classes_matrix = pd.read_csv(
            os.path.join(
                label_feat_path,
                "final_label.csv"),
            header=0,
            sep=";")
        classes = list(classes_matrix[classes_matrix.keys()[0]].values)
        text_feats, classes = compute_label_feats(
            clip_model, clip_feat_dim, label_feat_path, classes, "text_feats_FINALLABEL_LABELS.npy")
    return text_feats, classes
