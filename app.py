import os
import numpy as np
import cv2
import kiui
import trimesh
import torch
import rembg
from datetime import datetime
import subprocess
import gradio as gr

try:
    # running on Hugging Face Spaces
    import spaces

except ImportError:
    # running locally, use a dummy space
    class spaces:
        class GPU:
            def __init__(self, duration=60):
                self.duration = duration
            def __call__(self, func):
                return func


from flow.model import Model
from flow.configs.schema import ModelConfig
from flow.utils import get_random_color, recenter_foreground
from vae.utils import postprocess_mesh

# download checkpoints
from huggingface_hub import hf_hub_download
flow_ckpt_path = hf_hub_download(repo_id="nvidia/PartPacker", filename="flow.pt")
vae_ckpt_path = hf_hub_download(repo_id="nvidia/PartPacker", filename="vae.pt")

TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)
MAX_SEED = np.iinfo(np.int32).max
bg_remover = rembg.new_session()

# model config
model_config = ModelConfig(
    vae_conf="vae.configs.part_woenc",
    vae_ckpt_path=vae_ckpt_path,
    qknorm=True,
    qknorm_type="RMSNorm",
    use_pos_embed=False,
    dino_model="dinov2_vitg14",
    hidden_dim=1536,
    flow_shift=3.0,
    logitnorm_mean=1.0,
    logitnorm_std=1.0,
    latent_size=4096,
    use_parts=True,
)

# instantiate model
model = Model(model_config).eval().cuda().bfloat16()

# load weight
ckpt_dict = torch.load(flow_ckpt_path, weights_only=True)
model.load_state_dict(ckpt_dict, strict=True)

# process function
@spaces.GPU(duration=120)
def process(input_image, input_num_steps=30, input_cfg_scale=7.5, grid_res=384, seed=42, randomize_seed=True):

    # seed
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)
    kiui.seed_everything(seed)

    # output path
    os.makedirs("output", exist_ok=True)
    output_glb_path = f"output/partpacker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"

    # input image
    input_image = np.array(input_image) # uint8

    # bg removal if there is no alpha channel
    if input_image.shape[-1] == 3:
        input_image = rembg.remove(input_image, session=bg_remover)  # [H, W, 4]
    mask = input_image[..., -1] > 0
    image = recenter_foreground(input_image, mask, border_ratio=0.1)
    image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])  # white background

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float().cuda()
    data = {"cond_images": image_tensor}

    with torch.inference_mode():
        results = model(data, num_steps=input_num_steps, cfg_scale=input_cfg_scale)

    latent = results["latent"]

    # query mesh

    data_part0 = {"latent": latent[:, : model.config.latent_size, :]}
    data_part1 = {"latent": latent[:, model.config.latent_size :, :]}

    with torch.inference_mode():
        results_part0 = model.vae(data_part0, resolution=grid_res)
        results_part1 = model.vae(data_part1, resolution=grid_res)

    vertices, faces = results_part0["meshes"][0]
    mesh_part0 = trimesh.Trimesh(vertices, faces)
    mesh_part0.vertices = mesh_part0.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part0 = postprocess_mesh(mesh_part0, 5e4)
    parts = mesh_part0.split(only_watertight=False)

    vertices, faces = results_part1["meshes"][0]
    mesh_part1 = trimesh.Trimesh(vertices, faces)
    mesh_part1.vertices = mesh_part1.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part1 = postprocess_mesh(mesh_part1, 5e4)
    parts.extend(mesh_part1.split(only_watertight=False))

    # split connected components and assign different colors
    for j, part in enumerate(parts):
        # each component uses a random color
        part.visual.vertex_colors = get_random_color(j, use_float=True)

    mesh = trimesh.Scene(parts)
    # export the whole mesh
    mesh.export(output_glb_path)

    return seed, image, output_glb_path

# gradio UI

_TITLE = '''PartPacker: Efficient Part-level 3D Object Generation via Dual Volume Packing'''

_DESCRIPTION = '''
<div>
<a style="display:inline-block" href="https://research.nvidia.com/labs/dir/partpacker/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/NVlabs/PartPacker"><img src='https://img.shields.io/github/stars/NVlabs/PartPacker?style=social'/></a>
</div>

* Each part is visualized with a random color, and can be separated in the GLB file.
* If the output is not satisfactory, please try different random seeds!
'''

block = gr.Blocks(title=_TITLE).queue()
with block:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('# ' + _TITLE)
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            # input image
            input_image = gr.Image(label="Image", type='pil')
            # inference steps
            input_num_steps = gr.Slider(label="Inference steps", minimum=1, maximum=100, step=1, value=30)
            # cfg scale
            input_cfg_scale = gr.Slider(label="CFG scale", minimum=2, maximum=10, step=0.1, value=7.5)
            # grid resolution
            input_grid_res = gr.Slider(label="Grid resolution", minimum=256, maximum=512, step=1, value=384)
            # random seed
            seed = gr.Slider(label="Random seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            # gen button
            button_gen = gr.Button("Generate")


        with gr.Column(scale=4):
            with gr.Tab("3D Model"):
                # glb file
                output_model = gr.Model3D(label="Geometry", height=380)

            with gr.Tab("Input Image"):
                # background removed image
                output_image = gr.Image(interactive=False, show_label=False)
                

        with gr.Column(scale=1):
            gr.Examples(
                examples=[
                    ["examples/barrel.png"],
                    ["examples/cactus.png"],
                    ["examples/cyan_car.png"],
                    ["examples/pickup.png"],
                    ["examples/swivelchair.png"],
                    ["examples/warhammer.png"],
                ],
                inputs=[input_image],
                cache_examples=False,
            )

        button_gen.click(process, inputs=[input_image, input_num_steps, input_cfg_scale, input_grid_res, seed, randomize_seed], outputs=[seed, output_image, output_model])

block.launch()