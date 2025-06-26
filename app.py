import os
import numpy as np
import cv2
import kiui
import trimesh
import torch
import rembg
from datetime import datetime

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

# get random seed
def get_random_seed(randomize_seed, seed):
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)
    return seed

# process image
def process_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image file at: {image_path}. It might be corrupted or in an unsupported format.")

    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # bg removal if there is no alpha channel
        image = rembg.remove(image, session=bg_remover)  # [H, W, 4]
    mask = image[..., -1] > 0
    image = recenter_foreground(image, mask, border_ratio=0.1)
    image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_AREA)
    return image

# process generation
def process_3d(input_image, num_steps, cfg_scale, grid_res, seed, simplify_mesh, target_num_faces):

    # seed
    kiui.seed_everything(seed)

    # output path
    os.makedirs("output", exist_ok=True)
    output_glb_path = f"output/partpacker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"

    # input image (assume processed to RGBA uint8)
    image = input_image.astype(np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])  # white background
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float().cuda()

    data = {"cond_images": image_tensor}

    with torch.inference_mode():
        results = model(data, num_steps=num_steps, cfg_scale=cfg_scale)

    latent = results["latent"]

    # query mesh

    data_part0 = {"latent": latent[:, : model.config.latent_size, :]}
    data_part1 = {"latent": latent[:, model.config.latent_size :, :]}

    with torch.inference_mode():
        results_part0 = model.vae(data_part0, resolution=grid_res)
        results_part1 = model.vae(data_part1, resolution=grid_res)

    if not simplify_mesh:
        target_num_faces = -1

    vertices, faces = results_part0["meshes"][0]
    mesh_part0 = trimesh.Trimesh(vertices, faces)
    mesh_part0.vertices = mesh_part0.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part0 = postprocess_mesh(mesh_part0, target_num_faces)
    parts = mesh_part0.split(only_watertight=False)

    vertices, faces = results_part1["meshes"][0]
    mesh_part1 = trimesh.Trimesh(vertices, faces)
    mesh_part1.vertices = mesh_part1.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part1 = postprocess_mesh(mesh_part1, target_num_faces)
    parts.extend(mesh_part1.split(only_watertight=False))

    # split connected components and assign different colors
    for j, part in enumerate(parts):
        # each component uses a random color
        part.visual.vertex_colors = get_random_color(j, use_float=True)

    mesh = trimesh.Scene(parts)
    # export the whole mesh
    mesh.export(output_glb_path)

    return output_glb_path

def main():
    # --- Inference Parameters ---
    # You can change these parameters directly in the code
    input_image_path = "examples/rabbit.png"
    num_steps = 50
    cfg_scale = 7.0
    grid_res = 384
    randomize_seed = True
    seed = 42
    simplify_mesh = False
    target_num_faces = 100000
    # --------------------------

    print(f"Processing image: {input_image_path}")
    processed_image = process_image(input_image_path)

    seed = get_random_seed(randomize_seed, seed)
    print(f"Using seed: {seed}")

    print("Generating 3D model...")
    output_path = process_3d(
        processed_image,
        num_steps,
        cfg_scale,
        grid_res,
        seed,
        simplify_mesh,
        target_num_faces,
    )

    print(f"3D model saved to: {output_path}")

if __name__ == "__main__":
    main()