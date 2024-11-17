
import csv
import sys

import gradio as gr
import matplotlib.pyplot as plt

from maploc.inference import OrienterNetv2
from maploc.osm.tiling import TileManager
from maploc.osm.viz import Colormap, GeoPlotter, plot_nodes
from maploc.utils.viz_2d import features_to_RGB, plot_images
from maploc.utils.viz_localization import (
    likelihood_overlay,
    plot_dense_rotations,
)

csv.field_size_limit(sys.maxsize)
gr.utils.sanitize_value_for_csv = lambda v: v

def run(image, address, tile_size_meters, num_rotations, do_hierarchical, topk):
    """Load inputs and return estimated pose """
    image_path = image.name
    
    # Initialize model
    model = OrienterNetv2(num_rotations=int(num_rotations))


    try:
        
        plots = model.run(image_path, address or None, int(tile_size_meters), do_hierarchical, topk)
    except ValueError as e:
        raise gr.Error(str(e))

    # # Pre-process inputs
    # try:
    #     image, camera, roll_pitch, proj, bbox = inference.preprocess_inputs(
    #         image_path,
    #         prior_address=address or None,
    #         tile_size_meters=int(tile_size_meters),
    #     )
    # except ValueError as e:
    #     raise gr.Error(str(e))

    # tiler = TileManager.from_bbox(proj, bbox + 10, model.config.data.pixel_per_meter)
    # canvas = tiler.query(bbox)
    # map_viz = Colormap.apply(canvas.raster)

    # plot_images([image, map_viz], titles=["input image", "OpenStreetMap raster"], pad=2)
    # plot_nodes(1, canvas.raster[2], fontsize=6, size=10)
    # fig1 = plt.gcf()

    # # Run model inference
    # try:
    #     uv, yaw, prob, neural_map, image_rectified = model.localize(
    #         image, camera, canvas, gravity=gravity, hierarchical_topk=heirarchical_topk
    #     )
    # except RuntimeError as e:
    #     raise gr.Error(str(e))

    # Visualize the predictions
    overlay = likelihood_overlay(prob.numpy().max(-1), map_viz.mean(-1, keepdims=True))
    (neural_map_rgb,) = features_to_RGB(neural_map.numpy())
    plot_images([overlay, neural_map_rgb], titles=["heatmap", "neural map"], pad=2)
    ax = plt.gcf().axes[0]
    ax.scatter(*canvas.to_uv(bbox.center), s=5, c="red")
    plot_dense_rotations(ax, prob, w=0.005, s=1 / 25)
    add_circle_inset(ax, uv)
    fig2 = plt.gcf()

    # # Plot as interactive figure
    # latlon = proj.unproject(canvas.to_xy(uv))
    # bbox_latlon = proj.unproject(canvas.bbox)
    # plot = GeoPlotter(zoom=16.5)
    # plot.raster(map_viz, bbox_latlon, opacity=0.5)
    # plot.raster(likelihood_overlay(prob.numpy().max(-1)), proj.unproject(bbox))
    # plot.points(proj.latlonalt[:2], "red", name="location prior", size=10)
    # plot.points(latlon, "black", name="argmax", size=10, visible="legendonly")
    # plot.bbox(bbox_latlon, "blue", name="map tile")

    # coordinates = f"(latitude, longitude) = {tuple(map(float, latlon))}"
    # coordinates += f"\nheading angle = {yaw:.2f}°"
    # return fig1, fig2, plot.fig, coordinates
    
    return *list(plots.values())


examples = [
    ["assets/query_zurich_1.JPG", "ETH CAB Zurich", 128, 256],
    ["assets/query_vancouver_1.JPG", "Vancouver Waterfront Station", 128, 256],
    ["assets/query_vancouver_2.JPG", None, 128, 256],
    ["assets/query_vancouver_3.JPG", None, 128, 256],
]

# TODO: add model architecture.
description = """
<h1 align="center">
  <ins>OrienterNetv2</ins>
  <br>
  Improved Visual Localization in 2D Public Maps
  <br>
  with Multi-Scale Neural Matching </h1>
<h3 align="center">
    <a href="https://psarlin.com/orienternet" target="_blank">Project Page</a> |
</h3>
<p align="center">
OrienterNetv2 finds the position and orientation of any image using publicly OpenStreetMap maps and/or Satellite imagery.
This work is my Master Thesis which focused on improving the accuracy, efficiency, and scalability of <a href="https://psarlin.com/orienternet" target="_blank">OrienterNet</a>.
Click on one of the provided examples or upload your own image!
</p>
"""

# <a href="https://arxiv.org/pdf/2304.02009.pdf" target="_blank">Paper</a> |
# <a href="https://github.com/facebookresearch/OrienterNet" target="_blank">Code</a> |
# <a href="https://youtu.be/wglW8jnupSs" target="_blank">Video</a>

app = gr.Interface(
    fn=run,
    inputs=[
        gr.File(file_types=["image"]),
        gr.Textbox(
            label="Prior Location",
            info="Required if the image metadata (EXIF) does not contain a GPS prior."
            "Enter an address, building, street, or city name.",
        ),
        gr.Radio(
            [64, 128, 256, 512, 1024, 2048, 4096],
            value=1024,
            label="Search radius (meters)",
            info="Depends on how coarse the prior location is.",
        ),
        gr.Radio(
            [64, 128, 256],
            value=128,
            label="Number of rotations",
            info="Reduce for faster results",
        ),
        # gr.Checkbox(
        #     value=True,
        #     label="Use coarse Prior Model",
        #     info="Effective for searching large areas and/or incorporating visual info from upto 256m in front of camera."
        # ),
        gr.Checkbox(
            value=True,
            label="Hierarchical Localization",
            info="Reduce inference time by hierarchically narrowing down search space"
        ),
        gr.Radio(
            [1, 2, 3, 5, 10],
            value=3,
            label="Hierarchical Search Top-K",
            info="Higher K values cover larger areas but require more time.",
        ),
    ],
    outputs=[
        gr.Plot(label="Inputs"),
        gr.Plot(label="Outputs"),
        gr.Plot(label="Interactive map"),
        gr.Textbox(label="Predicted coordinates"),
    ],
    description=description,
    examples=examples,
    cache_examples=True,
)
# TODO: add Details
app.launch(share=False)