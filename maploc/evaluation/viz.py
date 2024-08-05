# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import defaultdict
import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from maploc.utils.wrappers import Transform2D

from ..osm.viz import Colormap, plot_nodes
from ..utils.io import write_torch_image
from ..utils.viz_2d import features_to_RGB, plot_images, save_plot
from ..utils.viz_localization import (
    add_circle_inset,
    likelihood_overlay,
    plot_dense_rotations,
    plot_pose,
    plot_bev,
)


def plot_example_single(
    idx,
    model,
    pred,
    data,
    results,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
    return_plots=False,
    overlay_bev=True,
):

    if results is None:
        results = defaultdict(float)
    # map_T_cam (or m_T_c): Transform of cam in pixel space.
    # map_t_cam (or m_t_c): only translation.
    # m_r_c (yaw): only rotation. East-facing, counter-clockwise rotation
    plots = []

    scene, name = data["scene"], data["name"]

    keys = list(model.model.conf.bev_mapper.z_max)
    if "xy_max_error_chain" in results:
        keys += ["chain"]

    for index, k in enumerate(keys):

        tile_T_cam_gt = data["tile_T_cam"][k]
        # When chaining, we choose map and bev of the finest resolution, because pred outputs in finest map space.
        if k == "chain":
            bev_ppm = model.model.conf.pixel_per_meter[0]
        else:
            bev_ppm = model.model.conf.pixel_per_meter[index]

        # map_T_cam_gt = Transform2D.to_pixels(
        #     tile_T_cam_gt, 1 / data["bev_ppm"]
        # )
        map_T_cam_gt = Transform2D.to_pixels(tile_T_cam_gt, 1 / bev_ppm)

        m_t_c_gt = map_T_cam_gt.t.squeeze(0)  # ij_gt
        yaw_gt = map_T_cam_gt.angle.squeeze(0)  # m_r_c_gt

        if k == "chain":
            tile_t_gps = data["tile_t_gps"][32.0].squeeze(0)
        else:
            tile_t_gps = data["tile_t_gps"][k].squeeze(0)

        if show_fused and "ij_fused" in pred[k]:
            m_t_c_pred = pred[k]["ij_fused"]
            yaw_p = pred.get("yaw_fused")
        else:
            m_T_c_pred = pred[k]["map_T_cam_max"]
            m_t_c_pred = m_T_c_pred.t.squeeze(0)  # ij_p
            yaw_p = m_T_c_pred.angle.squeeze(0)  # m_r_c_pred

        image = data["image"].permute(1, 2, 0)
        if "valid" in data:
            image = image.masked_fill(~data["valid"].unsqueeze(-1), 0.3)

        lp_ijt = lp_ij = pred[k]["log_probs"]
        if show_fused and "log_probs_fused" in pred[k]:
            lp_ijt = lp_ij = pred[k]["log_probs_fused"]
        elif not show_masked_prob and "scores_unmasked" in pred[k]:
            lp_ijt = lp_ij = pred[k]["scores_unmasked"]
        has_rotation = lp_ijt.ndim == 3
        if has_rotation:
            lp_ij = lp_ijt.max(-1).values
        if lp_ij.min() > -np.inf:
            lp_ij = lp_ij.clip(min=np.percentile(lp_ij, 1))
        prob = lp_ij.exp()

        if k == "chain":
            feats_map = pred[32.0]["features_map"]
        else:
            feats_map = pred[k]["features_map"]
        (feats_map_rgb,) = features_to_RGB(feats_map.numpy())

        if k == "chain":
            k_str = k
        else:
            k_str = str(int(k))
        text1 = rf'$\Delta xy$: {results[f"xy_max_error_{k_str}"]:.1f}m'
        if has_rotation:
            text1 += rf', $\Delta\theta$: {results[f"yaw_max_error_{k_str}"]:.1f}°'

        if show_fused and "xy_fused_error" in results:
            text1 += rf', $\Delta xy_{{fused}}$: {results["xy_fused_error"]:.1f}m'
            text1 += rf', $\Delta\theta_{{fused}}$: {results["yaw_fused_error"]:.1f}°'
        if show_dir_error and "directional_error" in results:
            err_lat, err_lon = results["directional_error"]  # TODO: adapt to dict
            text1 += rf",  $\Delta$lateral/longitundinal={err_lat:.1f}m/{err_lon:.1f}m"
        if "xy_gps_error" in results:
            text1 += rf',  $\Delta xy_{{GPS}}$: {results["xy_gps_error"]:.1f}m'

        maps_viz = []
        maps_titles = []
        if "semantic_map" in data:
            if k == "chain":
                rasters = data["semantic_map"][32.0]
            else:
                rasters = data["semantic_map"][k]

            map_viz = Colormap.apply(rasters)
            maps_titles.append("semantic map")
            maps_viz.append(map_viz)
        if "aerial_map" in data:
            aerial_map = data["aerial_map"].permute(1, 2, 0) / 255.0
            maps_titles.append("aerial map")
            maps_viz.append(aerial_map.numpy())

        # If a map is too large
        maps_viz = [
            (
                torch.nn.functional.interpolate(
                    torch.from_numpy(m).moveaxis(-1, -3).unsqueeze(1),
                    size=tuple(prob.shape),
                )
                .squeeze(1)
                .moveaxis(-3, -1)
                .numpy()
                if tuple(prob.shape) != tuple(m.shape[:2])
                else m
            )
            for m in maps_viz
        ]

        overlay = likelihood_overlay(prob.numpy(), maps_viz[0].mean(-1, keepdims=True))

        logl = lp_ij.numpy()
        overlay, logl, feats_map_rgb = [
            np.swapaxes(x, 0, 1) for x in (overlay, logl, feats_map_rgb)
        ]
        for i, mv in enumerate(maps_viz):
            maps_viz[i] = np.swapaxes(mv, 0, 1)

        # overlay_aerial = aerial_map

        # Create aerial verification plots
        # map_viz_im = Image.fromarray((map_viz*255.).astype(np.uint8)).convert('RGBA') DELETED

        plot_images(
            [image, *maps_viz, overlay, logl, feats_map_rgb],
            titles=[text1, *maps_titles, "likelihood", "log-likelihood", "neural map"],
            origins=["upper", *["lower"] * len(maps_viz), "lower", "lower", "lower"],
            dpi=75,
            cmaps="jet",
        )
        fig = plt.gcf()
        axes = fig.axes
        axes[1].images[0].set_interpolation("none")
        axes[2].images[0].set_interpolation("none")
        Colormap.add_colorbar()

        # if "semantic_map" in pred:
        #     plot_nodes(1, rasters[2], refactored=True)

        if overlay_bev:
            # TODO: when chaining, the bev overlay should be the max depth bev.
            # currently, the chain is in the smallest depth's resolution (finest).
            if k == "chain":
                (bev,) = features_to_RGB(
                    pred[32.0]["features_bev"].numpy(),
                    masks=[pred[32.0]["valid_bev"].numpy()],
                )
            else:
                (bev,) = features_to_RGB(
                    pred[k]["features_bev"].numpy(),
                    masks=[pred[k]["valid_bev"].numpy()],
                )
            bev = np.swapaxes(bev, 0, 1)
            plot_bev(bev, uv=m_t_c_pred, yaw=yaw_p, zorder=10, ax=axes[1])

        if show_gps and tile_t_gps is not None:
            m_t_gps = Transform2D.to_pixels(
                tile_t_gps,
                1 / bev_ppm,
            )
            plot_pose(
                [1] + ([2] if len(maps_viz) > 1 else []),
                m_t_gps,
                c="blue",
                refactored=True,
            )
        plot_pose(
            [1] + ([2] if len(maps_viz) > 1 else []),
            m_t_c_gt,
            yaw_gt,
            c="red",
            refactored=True,
        )
        plot_pose(
            [1] + ([2] if len(maps_viz) > 1 else []),
            m_t_c_pred,
            yaw_p,
            c="k",
            refactored=True,
        )

        plot_dense_rotations(
            2 if len(maps_viz) == 1 else 3, lp_ijt.exp(), refactored=True
        )
        # inset_center = m_t_c_pred if results["xy_max_error"] < 5 else m_t_c_gt

        # Doesn't work for refactored axes conventions
        # axins = add_circle_inset(axes[2], inset_center, refactored=True)
        # axins.scatter(*ij_gt, lw=1, c="red", ec="k", s=50, zorder=15)

        axes[0].text(
            0.003,
            0.003,
            f"{scene}/{name}",
            transform=axes[0].transAxes,
            fontsize=3,
            va="bottom",
            ha="left",
            color="w",
        )

        if out_dir is not None:
            name_ = name.replace("/", "_")
            p = str(
                out_dir
                / f"{idx}_{results[f'xy_max_error_{k_str}']:.1f}_{scene}_{name_}_{k}_{{}}.png"
            )

            save_plot(p.format("pred"))
            plt.close()

        # Don't plot bev or scales for chain - they are not "chained"
        if k == "chain":
            continue

        if return_plots:
            import matplotlib

            matplotlib.use("Agg")
            # save the fig to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            plot = Image.open(buf)
            plots.append(to_tensor(plot))
        else:
            plt.show()

        # plot_images(
        #     [image, *maps_viz],
        #     titles=[],
        #     origins=["upper", *["lower"] * len(maps_viz)],
        #     dpi=75,
        #     cmaps="jet",
        #     )
        # # plot_pose(
        # #     [1] + ([2] if len(maps_viz) > 1 else []),
        # #     m_t_c_gt,
        # #     yaw_gt,
        # #     c="red",
        # #     refactored=True,
        # # )
        # plt.savefig("TrainingData1.png")

        if len(maps_viz) > 1:
            pass
            # TODO: plot a new row containing each map norm

            # semantic_feats_map = pred["semantic_map"]["map_features"][0]
            # aerial_feats_map = pred["aerial_map"]
            # (feats_map_rgb_semantic,) = features_to_RGB(semantic_feats_map.numpy())
            # (feats_map_rgb_aerial,) = features_to_RGB(aerial_feats_map.numpy())
            # feats_map_rgb_semantic, feats_map_rgb_aerial = [
            #     np.swapaxes(x, 0, 1) for x in (feats_map_rgb_semantic, feats_map_rgb_aerial)
            # ]
            # plot_images(
            #     [feats_map_rgb_semantic, feats_map_rgb_aerial],
            #     titles=["semantic neural map", "aerial neural map"],
            #     origins=["lower", "lower"],
            #     dpi=75,
            #     cmaps="jet",
            # )

            # if return_plots:
            #     # save the fig to a buffer
            #     buf = io.BytesIO()
            #     plt.savefig(buf, format="png")
            #     plt.close(fig)
            #     buf.seek(0)
            #     plot = Image.open(buf)
            #     plots.append(to_tensor(plot))

            # else:
            #     plt.show()

        if fig_for_paper:
            # !cp ../datasets/MGL/{scene}/images/{name}.jpg {out_dir}/{scene}_{name}.jpg
            plot_images([map_viz])
            plt.gca().images[0].set_interpolation("none")
            plot_nodes(0, rasters[2])
            plot_pose([0], m_t_c_gt, yaw_gt, c="red")
            plot_pose([0], m_t_c_pred, yaw_p, c="k")
            save_plot(p.format("map"))
            plt.close()
            plot_images([lp_ij], cmaps="jet")
            plot_dense_rotations(0, lp_ijt.exp())
            save_plot(p.format("loglikelihood"), dpi=100)
            plt.close()
            plot_images([overlay])
            plt.gca().images[0].set_interpolation("none")
            axins = add_circle_inset(plt.gca(), inset_center)
            axins.scatter(*m_t_c_gt, lw=1, c="red", ec="k", s=50)
            save_plot(p.format("likelihood"))
            plt.close()
            write_torch_image(
                p.format("neuralmap").replace("pdf", "jpg"), feats_map_rgb
            )
            write_torch_image(p.format("image").replace("pdf", "jpg"), image.numpy())

        scales_scores = pred[k]["pixel_scales"]  # [..., 2:-7]
        z_max = k
        if z_max == 256.0:
            scales_scores[..., -10:] = 0  # 256m
        elif z_max == 128.0:
            scales_scores[..., -10:] = 0  # 128m
        elif z_max == 64.0:
            scales_scores[..., :2] = scales_scores[..., -10:] = 0  # 64m
        elif z_max == 32.0:
            scales_scores[..., :6] = scales_scores[..., -7:] = 0  # 32m
        # max_scoring_scale = scales_scores.max(-1).indices  # scale with highest score

        log_prob = torch.nn.functional.log_softmax(scales_scores, dim=-1)
        scales_exp = torch.sum(
            log_prob.exp() * torch.arange(scales_scores.shape[-1]), -1
        )
        max_score = log_prob.max(-1).values.exp()
        total_score = torch.logsumexp(scales_scores, -1)

        feats_q = pred[k]["features_bev"]
        mask_bev = pred[k]["valid_bev"]
        prior = None
        if "semantic_map" in pred[k] and "log_prior" in pred[k]["semantic_map"]:
            prior = pred[k]["semantic_map"]["log_prior"][0].sigmoid()
        if "bev" in pred[k] and "confidence" in pred[k]["bev"]:
            # In multiscale weighted training, the confidences are exp scaled so we log for viz
            conf_q = pred[k]["bev"]["confidence"]
            if model.model.conf.add_temperature:
                conf_q = torch.log(conf_q)  # add 1e-10?
        else:
            conf_q = torch.norm(feats_q, dim=0)
        conf_q = conf_q.masked_fill(~mask_bev, np.nan)
        (feats_q_rgb,) = features_to_RGB(feats_q.numpy(), masks=[mask_bev.numpy()])
        # feats_map_rgb, feats_q_rgb, = features_to_RGB(
        #     feats_map.numpy(), feats_q.numpy(), masks=[None, mask_bev])
        norm_map = torch.norm(feats_map, dim=0)
        conf_q, feats_q_rgb, norm_map = [
            np.swapaxes(x, 0, 1) for x in [conf_q, feats_q_rgb, norm_map]
        ]

        if prior is not None:
            prior = np.swapaxes(prior, 0, 1)
        origins = ["lower", "lower", "lower"] + ([] if prior is None else ["lower"])
        plot_images(
            [conf_q, feats_q_rgb, norm_map] + ([] if prior is None else [prior]),
            titles=[
                "BEV confidence",
                "BEV features",
                "map norm",
            ]
            + ([] if prior is None else ["map prior"]),
            origins=origins,
            dpi=50,
            cmaps="jet",
        )

        if out_dir is not None:
            save_plot(p.format("bev"))
            plt.close()

        if return_plots:
            # save the fig to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            plot = Image.open(buf)
            plots.append(to_tensor(plot))
        else:
            plt.show()

        f_image = pred["features_image"]
        if f_image.shape[-3] == model.model.conf.latent_dim:
            start = 0
            end = model.model.conf.latent_dim
        else:
            start = index * model.model.conf.latent_dim
            end = (index + 1) * model.model.conf.latent_dim
        (feats_image,) = features_to_RGB(f_image[start:end, ...].numpy())
        origins = ["upper", "upper", "upper", "upper"]
        plot_images(
            [feats_image, scales_exp, max_score, total_score],
            titles=[
                "Image Features",
                "Expected scale",
                "Max Score",
                "Total score",
            ],
            origins=origins,
            dpi=50,
            cmaps="jet",
        )
        if out_dir is not None:
            save_plot(p.format("scales"))
            plt.close()

        if return_plots:
            # save the fig to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            plot = Image.open(buf)
            plots.append(to_tensor(plot))
        else:
            plt.show()

    return plots


def plot_example_sequential(
    idx,
    model,
    pred,
    data,
    results,
    out_dir=None,
    fig_for_paper=False,
    show_gps=False,
    show_fused=False,
    show_dir_error=False,
    show_masked_prob=False,
):
    return
