# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torchmetrics
from torchmetrics.utilities.data import dim_zero_cat

from maploc.utils.wrappers import Transform2D

from .utils import deg2rad, rotmat2d


def location_error(xy, xy_gt):
    return torch.norm(xy - xy_gt.to(xy), dim=-1)


def angle_error(t, t_gt):
    error = torch.abs(t % 360 - t_gt.to(t) % 360)
    error = torch.minimum(error, 360 - error)
    return error


class Location2DRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, key="tile_T_cam_max", subkey=None, *args, **kwargs):
        self.threshold = threshold
        self.key = key
        self.subkey = subkey
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        if isinstance(pred[self.key][self.subkey], Transform2D):
            xy_p = pred[self.key][self.subkey].t
        else:
            xy_p = pred[self.key][self.subkey]
        xy_gt = data["tile_T_cam"][self.subkey].t
        assert xy_gt.shape == xy_p.shape
        error = location_error(xy_p, xy_gt)
        super().update((error <= self.threshold).float())


class AngleRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, key="tile_T_cam_max", subkey=None, *args, **kwargs):
        self.threshold = threshold
        self.key = key
        self.subkey = subkey
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        error = angle_error(
            pred[self.key][self.subkey].angle, data["tile_T_cam"][self.subkey].angle
        )
        super().update((error <= self.threshold).float())


class MeanMetricWithRecall(torchmetrics.Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("value", default=[], dist_reduce_fx="cat")

    def compute(self):
        return dim_zero_cat(self.value).mean(0)

    def get_errors(self):
        return dim_zero_cat(self.value)

    def recall(self, thresholds):
        error = self.get_errors()
        thresholds = error.new_tensor(thresholds)
        return (error.unsqueeze(-1) < thresholds).float().mean(0) * 100


class ExhaustiveEntropy(MeanMetricWithRecall):
    def __init__(self, key="log_probs", subkey=None, *args, **kwargs):
        self.key = key
        self.subkey = subkey
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        log_probs = pred[self.key]
        if self.subkey is not None:
            log_probs = log_probs[self.subkey]
        probs = log_probs.exp()
        entropy = -torch.sum(probs * (probs + 1e-9).log())
        n = torch.prod(torch.tensor(probs.shape)).to(entropy)
        norm_entropy = entropy / torch.log(n)  # [0, 1]
        value = norm_entropy
        self.value.append(value.float())
        # super().update(norm_entropy.float())


class AngleError(MeanMetricWithRecall):
    def __init__(self, key, subkey):
        super().__init__()
        self.key = key
        self.subkey = subkey

    def update(self, pred, data):
        value = angle_error(
            pred[self.key][self.subkey].angle, data["tile_T_cam"][self.subkey].angle
        )
        if value.numel():
            self.value.append(value)


class Location2DError(MeanMetricWithRecall):
    def __init__(self, key, subkey):
        super().__init__()
        self.key = key
        self.subkey = subkey

    def update(self, pred, data):
        xy_gt = data["tile_T_cam"][self.subkey].t

        if isinstance(pred[self.key][self.subkey], Transform2D):
            xy_p = pred[self.key][self.subkey].t
        else:
            xy_p = pred[self.key][self.subkey]

        assert xy_gt.shape == xy_p.shape
        value = location_error(xy_p, xy_gt)
        if value.numel():
            self.value.append(value)


class LateralLongitudinalError(MeanMetricWithRecall):
    def __init__(self, key="tile_T_cam_max", subkey=None):
        super().__init__()
        self.key = key
        self.subkey = subkey

    def update(self, pred, data):
        yaw = deg2rad(90 - data["tile_T_cam"][self.subkey].angle).squeeze(-1)
        shift = pred[self.key][self.subkey].t - data["tile_T_cam"][self.subkey].t
        shift = (rotmat2d(yaw) @ shift.unsqueeze(-1)).squeeze(-1)
        error = torch.abs(shift)
        value = error.view(-1, 2)
        if value.numel():
            self.value.append(value)
