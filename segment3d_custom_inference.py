#!/usr/bin/env python3
"""
Run Segment3D on a custom point‑cloud stored in a PLY file.

This script wraps the Segment3D inference pipeline so that it can be
applied to arbitrary PLY files containing XYZ coordinates and RGB
colors.  It loads a pre‑trained Segment3D checkpoint, processes the
input point cloud into a format understood by the model (using the
same voxelisation and normalisation steps as the official demo),
runs the network to obtain a set of binary instance masks, then
assigns each point to the highest scoring mask to produce a single
instance label per point.  The final instance labels are saved as a
NumPy array alongside an optional colourised PLY file for visual
inspection.

Example usage:

.. code-block:: bash

    # activate your conda environment first
    python segment3d_custom_inference.py \
        --ply /path/to/your_scene.ply \
        --checkpoint checkpoints/segment3d.ckpt \
        --output segmentation_labels.npy \
        --save_ply segmented_scene.ply

The script assumes that you have already cloned the Segment3D
repository and installed all required dependencies (PyTorch,
MinkowskiEngine, hydra, omegaconf, open3d, torch_scatter, cuml).  It
must be executed from within the root of the Segment3D repository so
that the relative `conf` directory and Python modules can be
imported.

Author: OpenAI assistant
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import torch

from hydra.experimental import initialize, compose
from omegaconf import OmegaConf

# Import helper functions from Segment3D.  These modules live in the
# repository and must be available on your PYTHONPATH.  When this
# script is run from the repository root the imports will work
# naturally; otherwise adjust sys.path accordingly.
from demo_utils import get_model, prepare_data
from torch_scatter import scatter_mean  # type: ignore


def get_mask_and_scores(cfg, mask_cls: torch.Tensor, mask_pred: torch.Tensor):
    """Filter masks and compute a per‑mask confidence score.

    This function follows the implementation in `demo.py`.  It
    removes empty masks, applies a sigmoid to the mask logits to
    produce a heatmap, and then computes the score as the dot
    product between class logits and mean mask heat.  Finally it
    selects the top‑k scoring masks (or all masks when topk_per_image
    is -1).

    Args:
        cfg: Hydra config dictionary.
        mask_cls: Tensor of shape (num_masks,) containing the class logits.
        mask_pred: Tensor of shape (num_points, num_masks) with mask logits.

    Returns:
        scores: 1D tensor containing scores for each retained mask.
        result_pred_mask: 2D tensor of shape (num_points, num_selected)
            containing binary masks after filtering.
    """
    # Remove empty masks
    result_pred_mask = (mask_pred > 0).float()
    mask_pred = mask_pred[:, result_pred_mask.sum(0) > 0]
    mask_cls = mask_cls[result_pred_mask.sum(0) > 0]
    result_pred_mask = result_pred_mask[:, result_pred_mask.sum(0) > 0]

    # Convert logits to probabilities
    heatmap = mask_pred.float().sigmoid()

    # Score each mask by combining class confidence with average heat
    mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
        result_pred_mask.sum(0) + 1e-6
    )
    score = mask_cls * mask_scores_per_image

    # Determine how many masks to keep
    if cfg.general.topk_per_image == -1:
        topk_count = len(score)
    elif cfg.general.topk_per_image < 1:
        # When topk_per_image is a float in (0,1), treat it as a fraction
        # of the total number of masks
        topk_count = int(np.ceil(cfg.general.topk_per_image * len(score)))
    else:
        topk_count = min(cfg.general.topk_per_image, len(score))

    score, topk_indices = score.topk(topk_count, sorted=True)
    result_pred_mask = result_pred_mask[:, topk_indices]
    return score, result_pred_mask


def get_full_res_mask(mask: torch.Tensor,
                      inverse_map: torch.Tensor,
                      point2segment_full: Optional[torch.Tensor]):
    """Upsample predicted masks back to the original point resolution.

    Args:
        mask: Tensor of shape (num_low_res_points, num_masks) – binary masks
            at the down‑sampled resolution used during inference.
        inverse_map: Tensor mapping indices of down‑sampled points back to
            the original point indices.
        point2segment_full: Optional mapping from original points to
            segmentation segments when using precomputed segments.

    Returns:
        Binary mask tensor of shape (num_masks, num_full_points).
    """
    # Expand from sparse tensor to full point cloud resolution
    mask = mask[inverse_map]  # shape: (num_full_points, num_masks)
    if point2segment_full is not None:
        # When segment indices are provided, first average per segment
        mask = scatter_mean(mask, point2segment_full.squeeze(0), dim=0)
        mask = (mask > 0.5).float()
        mask = mask[point2segment_full.squeeze(0)]
    return mask


def parse_predictions(cfg,
                      outputs: dict,
                      point2segment: Optional[torch.Tensor],
                      point2segment_full: Optional[torch.Tensor],
                      raw_coordinates: torch.Tensor,
                      inverse_map: torch.Tensor):
    """Convert raw model outputs into confidence scores and binary masks.

    This function handles the optional DBSCAN post‑processing as in
    `demo.py`.  It returns a 1D tensor of per‑mask scores and a
    boolean mask tensor of shape (num_masks, num_full_points).
    """
    # Extract class logits and mask logits from the model output
    logits = outputs["pred_logits"][0][:, 0].detach().cpu()
    masks = outputs["pred_masks"][0].detach().cpu()

    # When training on segments the model outputs segment‑level masks
    # that need to be expanded to point level
    if cfg.model.train_on_segments:
        masks = outputs["pred_masks"][0].detach().cpu()[point2segment.cpu()].squeeze(0)
    else:
        masks = outputs["pred_masks"][0].detach().cpu()

    # Optional DBSCAN to split disconnected clusters within a mask
    if cfg.general.use_dbscan:
        try:
            from cuml.cluster import DBSCAN  # pylint: disable=import-error
        except ImportError as exc:
            raise ImportError(
                "cuml is required for DBSCAN post‑processing. "
                "Please install it (e.g. conda install -c rapidsai -c nvidia -c "
                "conda-forge cuml cuda_version=11.3 python=3.10)"
            ) from exc
        new_logits = []
        new_masks = []
        for curr_query in range(masks.shape[1]):
            curr_mask = masks[:, curr_query] > 0
            # Skip empty predictions
            if raw_coordinates[curr_mask].shape[0] == 0:
                continue
            # Cluster the mask using DBSCAN on XYZ coordinates
            clusters = (
                DBSCAN(
                    eps=cfg.general.dbscan_eps,
                    min_samples=cfg.general.dbscan_min_points,
                    verbose=0,
                )
                .fit(raw_coordinates[curr_mask].cuda())
                .labels_
            )
            clusters = clusters.get()
            # Build a mask per cluster
            new_mask = np.zeros(curr_mask.shape, dtype=int)
            new_mask[curr_mask] = clusters + 1
            for cluster_id in np.unique(clusters):
                if cluster_id == -1:
                    continue  # ignore noise
                # Only keep reasonably sized clusters
                if (new_mask == cluster_id + 1).sum() > cfg.data.remove_small_group:
                    new_logits.append(logits[curr_query])
                    new_masks.append(
                        torch.from_numpy(
                            masks[:, curr_query].numpy() * (new_mask == cluster_id + 1)
                        )
                    )
        logits = new_logits
        masks = new_masks

    # Filter and score the masks
    scores, masks = get_mask_and_scores(cfg, torch.stack(logits).cpu(), torch.stack(masks).T)
    # Upsample to full resolution and convert to boolean
    masks_binary = get_full_res_mask(masks, inverse_map, point2segment_full)
    masks_binary = masks_binary.permute(1, 0).bool()
    return scores, masks_binary


def assign_instance_labels(scores: torch.Tensor, masks_binary: torch.Tensor) -> np.ndarray:
    """Assign each point to the highest scoring mask that covers it.

    Given a set of binary masks and their scores, this function
    computes a per‑point instance label such that each point belongs to
    at most one instance.  If a point is covered by multiple masks it
    is assigned to the mask with the highest score.  Points not
    covered by any mask receive a label of -1.

    Args:
        scores: 1D tensor of length N_masks.
        masks_binary: Boolean tensor of shape (N_masks, N_points).

    Returns:
        labels: 1D NumPy array of length N_points with integer instance
            indices (0‑based).  Unlabelled points are assigned -1.
    """
    num_masks, num_points = masks_binary.shape
    labels = -np.ones(num_points, dtype=np.int32)
    best_scores = -np.inf * np.ones(num_points, dtype=np.float32)

    # Iterate over masks in descending order of score
    sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
    for idx, mask_idx in enumerate(sorted_indices):
        mask = masks_binary[mask_idx].cpu().numpy()
        # Update labels where this mask has higher score than current best
        update = (mask.astype(bool)) & (scores[mask_idx].item() > best_scores)
        labels[update] = idx  # use position in sorted list as label
        best_scores[update] = scores[mask_idx].item()

    return labels


def main():
    parser = argparse.ArgumentParser(description="Inference for Segment3D on a custom PLY file")
    parser.add_argument("--ply", required=True, type=str, help="Path to input PLY file containing XYZ and RGB")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to the pre‑trained Segment3D checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output path for NumPy array of instance labels (e.g. labels.npy)",
    )
    parser.add_argument(
        "--save_ply",
        default=None,
        type=str,
        help=(
            "Optional path to write a colourised PLY file where each instance"
            " is assigned a random colour"
        ),
    )
    parser.add_argument(
        "--num_queries", default=400, type=int, help="Number of queries (masks) to predict"
    )
    parser.add_argument(
        "--topk",
        default=-1,
        type=float,
        help=(
            "Number of masks to retain.  Set -1 to keep all; a value in (0,1] to"
            " keep that fraction of masks; or an integer to keep a fixed number."
        ),
    )
    parser.add_argument(
        "--dbscan",
        action="store_true",
        help="Apply DBSCAN post‑processing to split disconnected masks",
    )
    parser.add_argument(
        "--eps",
        default=0.05,
        type=float,
        help="DBSCAN epsilon parameter (radius) when --dbscan is enabled",
    )
    parser.add_argument(
        "--min_points",
        default=5,
        type=int,
        help="Minimum number of points for a DBSCAN cluster to be kept",
    )
    parser.add_argument(
        "--remove_small_group",
        default=15,
        type=int,
        help="Ignore clusters with fewer than this many points after DBSCAN",
    )
    args = parser.parse_args()

    # Resolve absolute paths
    ply_path = Path(args.ply).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    save_ply_path = Path(args.save_ply).expanduser().resolve() if args.save_ply else None

    # Ensure we are in the Segment3D repository root so that hydra config files are found
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    # Load hydra config.  We use version_base=None for backward compatibility
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config_base_instance_segmentation.yaml")

    # Override configuration for inference on custom data
    cfg.general.train_mode = False
    cfg.general.train_on_segments = False
    cfg.general.eval_on_segments = False
    cfg.general.use_dbscan = args.dbscan
    cfg.general.dbscan_eps = args.eps
    cfg.general.dbscan_min_points = args.min_points
    cfg.general.topk_per_image = args.topk
    cfg.general.checkpoint = str(checkpoint_path)
    cfg.general.test_scene = "custom"
    cfg.model.num_queries = args.num_queries
    cfg.data.remove_small_group = args.remove_small_group

    # Instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg)
    model.eval()
    model.to(device)

    # Load the input point cloud
    if not ply_path.is_file():
        raise FileNotFoundError(f"Input PLY file not found: {ply_path}")
    # Try reading as a mesh first.  If no faces are present, fall back to point cloud
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    if not mesh.has_vertices():
        raise RuntimeError(f"Failed to read vertices from {ply_path}; ensure the file is a valid PLY")
    if not mesh.has_vertex_colors():
        # Some PLY files store only XYZ; colours are required by Segment3D
        raise RuntimeError(
            f"Input PLY {ply_path} must contain per‑vertex RGB colours."
        )
    # Convert a pure point cloud into a TriangleMesh if necessary
    if len(mesh.triangles) == 0:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = pcd.points
        mesh.vertex_colors = pcd.colors

    # Prepare data for the network (voxelisation and normalisation)
    data, point2segment, point2segment_full, raw_coords, inverse_map = prepare_data(cfg, mesh, None, device)

    # Forward pass
    with torch.no_grad():
        outputs = model(data, point2segment=None, raw_coordinates=raw_coords)

    # Parse raw predictions into scores and boolean masks
    scores, masks_binary = parse_predictions(cfg, outputs, None, point2segment_full, raw_coords, inverse_map)

    # Assign each point to the highest scoring instance
    labels = assign_instance_labels(scores, masks_binary)

    # Save labels as NumPy array
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, labels)
    print(f"Saved instance labels to {output_path}")

    # Optionally write a coloured PLY for visualisation
    if save_ply_path:
        # Assign a random colour per instance.  Points with label -1 remain white.
        unique_instances = np.unique(labels)
        instance_colours = {
            inst: np.random.rand(3)
            for inst in unique_instances
            if inst >= 0
        }
        colours = np.zeros_like(np.asarray(mesh.vertex_colors))
        for idx, inst in enumerate(labels):
            if inst >= 0:
                colours[idx] = instance_colours[inst]
            else:
                colours[idx] = np.array([1.0, 1.0, 1.0])  # white background
        out_mesh = o3d.geometry.PointCloud()
        out_mesh.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        out_mesh.colors = o3d.utility.Vector3dVector(colours)
        o3d.io.write_point_cloud(str(save_ply_path), out_mesh)
        print(f"Saved colourised point cloud to {save_ply_path}")


if __name__ == "__main__":
    main()
