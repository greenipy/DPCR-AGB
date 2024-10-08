import logging
import os
from itertools import product
from math import log10, ceil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from matplotlib.cm import get_cmap
from plyfile import PlyData, PlyElement

from torch_points3d.utils.config import is_list

log = logging.getLogger(__name__)


class Visualizer:
    """Initialize the Visualizer class.
    Parameters:
        viz_conf (OmegaConf Dictionary) -- stores all config for the visualizer
        num_batches (dict) -- This dictionary maps stage_name to #batches
        batch_size (int) -- Current batch size usef
        save_dir (str) -- The path used by hydra to store the experiment

    This class is responsible to save visuals into different formats. Currently supported formats are:
        ply -- Either an ascii or binary ply file, with the labels and gt stored as columns
        tensorboard -- Visualize point cloud in tensorboard
        wandb -- Upload point cloud to wandb. WARNING: This can become very slow, both in training and on the web.
            Make sure you properly limit the num_samples_per_epoch and wandb_max_points.
        csv -- creates a csv with predictions

    The configuration looks like this:
        visualization:
            activate: False # Whether to activate the visualizer
            format: ["ply", "tensorboard"] # 'pointcloud' is deprecated, use 'ply' instead
            num_samples_per_epoch: 2 # If negative, it will save all elements
            deterministic: True # False -> Randomly sample elements from epoch to epoch
            deterministic_seed: 0 # Random seed used to generate consistant keys if deterministic is True
            saved_keys: # Mapping from Data Object to structured numpy
                pos: [['x', 'float'], ['y', 'float'], ['z', 'float']]
                y: [['l', 'float']]
                pred: [['p', 'float']]
            indices: # List of indices to be saved (support "train", "test", "val")
                train: [0, 3]
            # Format specific options:
            ply_format: binary_big_endian # PLY format (support "binary_big_endian", "binary_little_endian", "ascii")
            tensorboard_mesh: # Mapping from mesh name and propety use to color
                label: 'y'
                prediction: 'pred'
            wandb_max_points: 10000 # Limits the size of the cloud that gets uploaded by random sampling.
                                    # "-1" saves the entire cloud
            wandb_cmap: # Applies a color map to the point cloud. Allows custom coloring of different classes.
                - [0, 0, 0] # class 0
                - [255, 255, 255] # class 1
                - [128, 128, 128] # class 2
    """

    def __init__(self, viz_conf, num_batches, batch_size, save_dir, tracker):
        # From configuration and dataset
        for stage_name, stage_num_sample in num_batches.items():
            setattr(self, "{}_num_batches".format(stage_name), stage_num_sample)
        self._batch_size = batch_size
        self._activate = viz_conf.activate
        self._format = [viz_conf.format] if not is_list(viz_conf.format) else viz_conf.format
        self._num_samples_per_epoch = int(viz_conf.num_samples_per_epoch)
        self._deterministic = viz_conf.deterministic
        self._seed = viz_conf.deterministic_seed if viz_conf.deterministic_seed is not None else 0
        self._tracker = tracker

        self._saved_keys = viz_conf.saved_keys
        self._tensorboard_mesh = {}
        self.save_dir = save_dir
        self._viz_path = os.path.join(save_dir, "viz")

        # Internal state
        self._stage = None
        self._current_epoch = None

        # format-specific initialization
        if "ply" in self._format:
            self._ply_format = viz_conf.ply_format if viz_conf.ply_format is not None else "binary_big_endian"

        if "tensorboard" in self._format:
            if not tracker._use_tensorboard:
                log.warning("Tensorboard visualization specified, but tensorboard isn't active.")
            else:
                self._tensorboard_mesh = viz_conf.tensorboard_mesh

                # SummaryWriter for tensorboard loging
                self._writer = tracker._writer

        if "wandb" in self._format:
            if not self._tracker._wandb:
                log.warning("Wandb visualization specified, but Wandb isn't active.")
            else:
                self._wandb_cmap = viz_conf.get("wandb_cmap", None)
                self._max_points = viz_conf.wandb_max_points if viz_conf.get("wandb_max_points",
                                                                             None) is not None else -1

        if "gpkg" in self._format or "csv" in self._format:
            self.dfs = []

        self._indices = {}
        self._contains_indices = False

        try:
            indices = getattr(viz_conf, "indices", None)
        except:
            indices = None

        if indices:
            for split in ["train", "test", "val"]:
                if split in indices:
                    split_indices = indices[split]
                    self._indices[split] = np.asarray(split_indices)
                    self._contains_indices = True

    def finalize_epoch(self, loaders):
        if "gpkg" in self._format or "csv" in self._format:
            df = pd.concat(self.dfs)

            for loader in loaders:

                dataset = loader.dataset
                for area_name in np.unique(df.area):
                    area_pred = df.query("area == @area_name").set_index("label_idx").drop("area", axis=1)

                    if "csv" in self._format:
                        dff = area_pred.query(f"stage == '{self._stage}'")
                        del dff["stage"]
                        file = self.save_dir / Path(f"{area_name}_{self._stage}_preds.csv")
                        dff.to_csv(file, mode='a', header=not file.exists())

                    if "gpkg" in self._format:
                        file = self.save_dir / Path(f"{area_name}_preds.gpkg")
                        area_label = dataset.areas.get(area_name, None)
                        if area_label is None:
                            continue
                        area_label = area_label["labels"]
                        area_df = area_label.join(area_pred, rsuffix="_pred", how="inner")
                        area_df = area_df.query(f"stage == '{self._stage}'")
                        del area_df["stage"]
                        area_df.to_file(file, mode='a' if file.exists() else "w")

    def get_indices(self, stage):
        """This function is responsible to calculate the indices to be saved"""
        if self._contains_indices:
            return
        stage_num_batches = getattr(self, "{}_num_batches".format(stage))
        total_items = (stage_num_batches - 1) * self._batch_size
        if stage_num_batches > 0:
            if self._num_samples_per_epoch < 0:  # All elements should be saved.
                if stage_num_batches > 0:
                    self._indices[stage] = np.arange(total_items)
                else:
                    self._indices[stage] = None
            else:
                if self._num_samples_per_epoch > total_items:
                    log.warning("Number of samples to save is higher than the number of available elements")
                self._indices[stage] = self._rng.permutation(total_items)[: self._num_samples_per_epoch]

    @property
    def is_active(self):
        return self._activate

    def reset(self, epoch, stage):
        """This function is responsible to restore the visualizer
        to start a new epoch on a new stage
        """
        self._current_epoch = epoch
        self._seen_batch = 0
        self._stage = stage
        if self._deterministic:
            self._rng = np.random.default_rng(self._seed)
        else:
            self._rng = np.random.default_rng()
        if self._activate:
            self.get_indices(stage)

    def _extract_from_PYG(self, item, pos_idx):
        num_samples = item.batch.shape[0]
        batch_mask = item.batch == pos_idx
        out_data = {}
        for k in item.keys:
            if torch.is_tensor(item[k]) and (k in self._saved_keys.keys() or k in self._tensorboard_mesh.values()):
                if item[k].shape[0] == num_samples:
                    out_data[k] = item[k][batch_mask]
        return out_data

    def _extract_from_dense(self, item, pos_idx):
        assert (  # TODO only true if task is segmentation
                item.y.shape[0] == item.pos.shape[0]
        ), "y and pos should have the same number of samples. Something is probably wrong with your data to visualise"
        num_samples = item.y.shape[0]
        out_data = {}
        for k in item.keys:
            if torch.is_tensor(item[k]) and (k in self._saved_keys.keys() or k in self._tensorboard_mesh.values()):
                if item[k].shape[0] == num_samples:
                    out_data[k] = item[k][pos_idx]
        return out_data

    def _dict_to_structured_npy(self, item):
        item.keys()
        out = []
        dtypes = []
        for k, v in item.items():
            v_npy = v.detach().cpu().numpy()
            if len(v_npy.shape) == 1:
                v_npy = v_npy[..., np.newaxis]
            for dtype in self._saved_keys[k]:
                dtypes.append(dtype)
            out.append(v_npy)

        out = np.concatenate(out, axis=-1)
        dtypes = np.dtype([tuple(d) for d in dtypes])
        return np.asarray([tuple(o) for o in out], dtype=dtypes)

    def save_visuals(self, model, loader):
        """This function is responsible to save the data into .ply objects
        Parameters:
            model -- Contains the model including visuals
            loader - Contains the dataloader
        Make sure the saved_keys  within the config maps to the Data attributes.
        """
        if self._stage in self._indices:
            visuals = model.get_current_visuals()

            if any(format in self._format for format in ["csv", "gpkg", "ply"]):
                dataset = loader.dataset
                pred = {}
                data_vis = model.data_visual
                if model.has_reg_targets:
                    preds = model.get_reg_output().detach().cpu().numpy()
                    for i, pred_name in enumerate(dataset.reg_targets):
                        pred[pred_name] = preds[:, i]
                pred["area"] = data_vis.area_name

                label_idx_ = [idx[0] for idx in data_vis.label_idx]
                pred["label_idx"] = label_idx_

                df = pd.DataFrame(pred)

                if "gpkg" in self._format or "csv" in self._format:
                    df["stage"] = self._stage
                    self.dfs.append(df)

                is_ply = "ply" in self._format
                if is_ply:
                    viz_path = Path(self._viz_path)
                    viz_path.mkdir(exist_ok=True)
                    for i, (_, sample) in enumerate(df.iterrows()):
                        area_name = sample["area"]
                        area_path = viz_path / area_name
                        area_path.mkdir(exist_ok=True)
                        label_idx = sample['label_idx']
                        file = area_path / f"{label_idx}.ply"
                        out_item = self._extract_from_PYG(data_vis, i)
                        out_item = self._dict_to_structured_npy(out_item)
                        self.save_ply(out_item, f"{area_name}_{label_idx}", file)

            if all(format not in self._format for format in ["wandb", "tensorboard"]):
                return
            stage_num_batches = getattr(self, "{}_num_batches".format(self._stage))
            batch_indices = self._indices[self._stage] // self._batch_size
            pos_indices = self._indices[self._stage] % self._batch_size
            for idx in np.argwhere(self._seen_batch == batch_indices).flatten():
                pos_idx = pos_indices[idx]
                for visual_name, item in visuals.items():
                    if hasattr(item, "batch") and item.batch is not None:  # The PYG dataloader has been used
                        out_item = self._extract_from_PYG(item, pos_idx)
                    else:
                        out_item = self._extract_from_dense(item, pos_idx)

                    if "tensorboard" in self._format and self._tracker._use_tensorboard:
                        self.save_tensorboard(out_item, visual_name, stage_num_batches)

                    out_item = self._dict_to_structured_npy(out_item)
                    gt_name = "{}_{}_{}_gt".format(self._current_epoch, self._seen_batch, pos_idx)
                    pred_name = "{}_{}_{}".format(self._current_epoch, self._seen_batch, pos_idx)

                    if "wandb" in self._format and self._tracker._wandb:
                        self.save_wandb(out_item, gt_name, pred_name)

            self._seen_batch += 1

    def save_ply(self, npy_array, visual_name, filename):

        el = PlyElement.describe(npy_array, visual_name)
        if self._ply_format == "ascii":
            PlyData([el], text=True).write(filename)
        elif self._ply_format == "binary_little_endian":
            PlyData([el], byte_order="<").write(filename)
        elif self._ply_format == "binary_big_endian":
            PlyData([el], byte_order=">").write(filename)
        else:
            PlyData([el]).write(filename)

    def save_tensorboard(self, out_item, visual_name, stage_num_batches):
        pos = out_item["pos"].detach().cpu().unsqueeze(0)
        colors = get_cmap("tab10")
        config_dict = {"material": {"size": 0.3}}

        for label, k in self._tensorboard_mesh.items():
            value = out_item[k].detach().cpu()

            if len(value.shape) == 2 and value.shape[1] == 3:
                if value.min() >= 0 and value.max() <= 1:
                    value = (255 * value).type(torch.uint8).unsqueeze(0)
                else:
                    value = value.type(torch.uint8).unsqueeze(0)
            elif len(value.shape) == 1 and value.shape[0] == 1:
                value = np.tile((255 * colors(value.numpy() % 10))[:, 0:3].astype(np.uint8), (pos.shape[0], 1)).reshape(
                    (1, -1, 3)
                )
            elif len(value.shape) == 1 or value.shape[1] == 1:
                value = (255 * colors(value.numpy() % 10))[:, 0:3].astype(np.uint8).reshape((1, -1, 3))
            else:
                continue

            self._writer.add_mesh(
                self._stage + "/" + visual_name + "/" + label,
                pos,
                colors=value,
                config_dict=config_dict,
                global_step=(self._current_epoch - 1) * (10 ** ceil(log10(stage_num_batches + 1))) + self._seen_batch,
            )

    def gen_bb_corners(self, points):
        points_min = np.min(points, axis=0)
        points_max = np.max(points, axis=0)
        points_min_max = np.stack([points_min, points_max], axis=0)

        bb_points = []
        for x, y, z in [i for i in product(range(2), repeat=3)]:  # 2^3 binary combination table
            bb_points.append([points_min_max[x, 0], points_min_max[y, 1], points_min_max[z, 2]])
        return bb_points

    def apply_cmap(self, val):
        out = np.zeros((val.shape[0], 3), dtype=int)
        for label, color in enumerate(self._wandb_cmap):
            out[val == label] = color
        return out

    PRED_COLOR = [255, 0, 0]  # red
    GT_COLOR = [124, 255, 0]  # green

    # https://docs.wandb.ai/guides/track/log/media#3d-visualizations
    def save_wandb(self, out_item, gt_name, pred_name):
        if self._max_points > 0:
            out_item = out_item[self._rng.permutation(len(out_item))[: self._max_points]]
        if self._wandb_cmap is None:
            assert (out_item["p"].max() + 1) <= 14, "Wandb classes must be in 1-14"
            assert (out_item["l"].max() + 1) <= 14, "Wandb classes must be in 1-14"

            pred_points = np.stack([out_item["x"], out_item["y"], out_item["z"], out_item["p"] + 1], axis=1)
            gt_points = np.stack([out_item["x"], out_item["y"], out_item["z"], out_item["l"] + 1], axis=1)
        else:
            pred_colors = self.apply_cmap(out_item["p"])
            gt_colors = self.apply_cmap(out_item["l"])
            pred_points = np.stack(
                [out_item["x"], out_item["y"], out_item["z"], pred_colors[:, 0], pred_colors[:, 1], pred_colors[:, 2]],
                axis=1,
            )
            gt_points = np.stack(
                [out_item["x"], out_item["y"], out_item["z"], gt_colors[:, 0], gt_colors[:, 1], gt_colors[:, 2]], axis=1
            )

        corners = self.gen_bb_corners(pred_points)

        pred_scene = wandb.Object3D(
            {
                "type": "lidar/beta",
                "points": pred_points,
                "boxes": np.array(  # draw 3d boxes
                    [
                        {
                            "corners": corners,
                            "label": pred_name,
                            "color": self.PRED_COLOR,
                        }
                    ]
                ),
            }
        )
        gt_scene = wandb.Object3D(
            {
                "type": "lidar/beta",
                "points": gt_points,
                "boxes": np.array(  # draw 3d boxes
                    [
                        {
                            "corners": corners,
                            "label": gt_name,
                            "color": self.GT_COLOR,
                        }
                    ]
                ),
            }
        )

        gt_scene_name = "{}/gt".format(self._stage)
        pred_scene_name = "{}/pred".format(self._stage)
        wandb.log({pred_scene_name: pred_scene, gt_scene_name: gt_scene, "epoch": self._current_epoch})
