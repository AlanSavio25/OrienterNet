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
    def __init__(self, threshold, key="tile_T_cam_max", *args, **kwargs):
        self.threshold = threshold
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        if isinstance(pred[self.key], Transform2D):
            xy_p = pred[self.key].t
        else:
            xy_p = pred[self.key]
        xy_gt = data["tile_T_cam"].t
        assert xy_gt.shape == xy_p.shape
        error = location_error(xy_p, xy_gt)
        super().update((error <= self.threshold).float())


class AngleRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, key="tile_T_cam_max", *args, **kwargs):
        self.threshold = threshold
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        error = angle_error(pred[self.key].angle, data["tile_T_cam"].angle)
        super().update((error <= self.threshold).float())


class SamplesRecall(torchmetrics.MeanMetric):
    def __init__(
        self, xy_thresh, yaw_thresh, key="tile_T_cam_samples", *args, **kwargs
    ):
        self.xy_thresh = xy_thresh
        self.yaw_thresh = yaw_thresh
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        gt = data["tile_T_cam"]
        samples = pred[self.key]
        if not isinstance(samples, Transform2D):
            samples = Transform2D(samples)
        dr, dt = (samples.inv() @ gt).magnitude()
        super().update(
            ((dr <= self.yaw_thresh).squeeze(-1) & (dt <= self.xy_thresh))
            .float()
            .mean()
        )


class ExhaustiveEntropy(torchmetrics.MeanMetric):
    def __init__(self, key="log_probs", *args, **kwargs):
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        log_probs = pred[self.key]
        probs = log_probs.exp()
        entropy = -torch.sum(probs * (probs + 1e-9).log())
        n = torch.prod(torch.tensor(probs.shape)).to(entropy)
        norm_entropy = entropy / torch.log(n)  # between 0 and 1
        super().update(norm_entropy.float())


class RansacPointDistributionEntropy(torchmetrics.MeanMetric):
    def __init__(self, key="prob_points", *args, **kwargs):
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        probs = pred[self.key].sum(-3)
        probs = probs / probs.sum()  # make probs add to 1
        entropy = -torch.sum(probs * (probs + 1e-9).log())
        n = torch.prod(torch.tensor(probs.shape)).to(entropy)
        norm_entropy = entropy / torch.log(n)
        super().update(norm_entropy.float())


class RansacPoseDistributionEntropy(torchmetrics.MeanMetric):
    def __init__(self, key="pose_scores", *args, **kwargs):
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        pose_scores = pred[self.key]
        pose_scores = pose_scores[pose_scores > 0]
        num_bins = int(1e3)
        hist = torch.histc(
            pose_scores.flatten(),
            bins=num_bins,
            min=0,
            max=pred["gt_pose_score"].squeeze().item(),
        )
        probs = hist / (hist.sum() + 1e-9)
        entropy = -torch.sum(probs * (probs + 1e-9).log())
        n = torch.tensor(len(probs))
        norm_entropy = entropy / torch.log(n)
        super().update(norm_entropy.float())


class RansacPoseScoresMean(torchmetrics.MeanMetric):
    def __init__(self, key="pose_scores", *args, **kwargs):
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        pose_scores = pred[self.key]
        pose_scores = pose_scores[pose_scores > 0]
        pose_scores /= pred["gt_pose_score"]  # scores in range [0,1]
        mean = pose_scores.mean()
        super().update(mean.float())


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


class AngleError(MeanMetricWithRecall):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def update(self, pred, data):
        value = angle_error(pred[self.key].angle, data["tile_T_cam"].angle)
        if value.numel():
            self.value.append(value)


class Location2DError(MeanMetricWithRecall):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def update(self, pred, data):
        if isinstance(pred[self.key], Transform2D):
            xy_p = pred[self.key].t
        else:
            xy_p = pred[self.key]

        xy_gt = data["tile_T_cam"].t
        assert xy_gt.shape == xy_p.shape
        value = location_error(xy_p, xy_gt)
        if value.numel():
            self.value.append(value)


class LateralLongitudinalError(MeanMetricWithRecall):
    def __init__(self, key="tile_T_cam_max"):
        super().__init__()
        self.key = key

    def update(self, pred, data):
        yaw = deg2rad(90 - data["tile_T_cam"].angle).squeeze(-1)
        shift = pred[self.key].t - data["tile_T_cam"].t
        shift = (rotmat2d(yaw) @ shift.unsqueeze(-1)).squeeze(-1)
        error = torch.abs(shift)
        value = error.view(-1, 2)
        if value.numel():
            self.value.append(value)
