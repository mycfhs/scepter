# PYTHONPATH=. python scepter/tools/webui.py --cfg scepter/methods/studio/scepter_ui.yaml
from torchvision.utils import save_image
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger
from scepter.modules.inference.largen_inference import LargenInference

from scepter.studio.inference.inference_ui.largen_ui import LargenUI

from PIL import Image
import numpy as np
from pytorch_lightning import seed_everything

# category = "backpack"
# prompt = "a photo of a hellokitty backpack"
# img_name = "116527"

# category = "berry_bowl"
# prompt = "a photo of a silver Blueberry bowl"
# img_name = "308261"


category = "can"
prompt = "a blue square can"
img_name = "4"


tar_path  = "/home/zl/yyc_workspace/Fooocus_diffusers"
ref_path  = "/home/zl/yyc_workspace/Fooocus_diffusers/train_data"

ref_image = Image.open(f"{ref_path}/{category}/image/00.jpg").resize((1024, 1024)).convert("RGB")
ref_mask = Image.open(f"{ref_path}/{category}/mask/00.png").resize((1024, 1024)).convert("L")

# output_height, output_width = ref_image.size
output_height, output_width = 1024, 1024

tar_image = Image.open(f"{tar_path}/{img_name}_1.jpg").resize((1024, 1024)).convert("RGB")
tar_mask  = Image.open(f"{tar_path}/{img_name}_2.jpg").resize((1024, 1024)).convert("L")

# category_2 = "duck_toy"
# tar_image = Image.open(f"{ref_path}/{category_2}/image/00.jpg").convert("RGB")
# tar_mask = Image.open(f"{ref_path}/{category_2}/mask/00.png").convert("L")

# tar_image = Image.open(f"asset/images/inpainting_text_ref/ex4_scene_im.jpg").convert("RGB")
# tar_mask = Image.open(f"asset/images/inpainting_text_ref/ex4_scene_mask.jpg").convert("L")
# ref_image = Image.open(f"asset/images/inpainting_text_ref/ex4_subject_im.jpg").convert("RGB")
# ref_mask = Image.open(f"asset/images/inpainting_text_ref/ex4_subject_mask.jpg").convert("L")


# ref_image = subject_image['background'].convert('RGB')
# ref_mask = subject_image['layers'][0].split()[-1].convert('L')
ref_image = np.asarray(ref_image)
ref_mask = np.asarray(ref_mask)
ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

tar_image = np.asarray(tar_image)
tar_mask = np.asarray(tar_mask)
tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)


data = LargenUI.data_preprocess_inpaint(
    None, tar_image, tar_mask, ref_image, ref_mask, True, 1.1, output_height, output_width
)


# init file system - modelscope
# FS.init_fs_client(Config(load=False, cfg_dict={'NAME': 'ModelscopeFs', 'TEMP_DIR': 'cache/data'}))
FS.init_fs_client(
    Config(load=False, cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "cache/cache_data"})
)   # 新版本改名字了hhh。 ui里面保存到cache data。我们就用之前下载好的，不然得重新下载。 这个在scepter_ui.yaml里面
FS.init_fs_client(
    Config(load=False, cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "cache/cache_data"})
)   # load参数不知道需不需要单独设置

# init model config
logger = get_logger(name='scepter')
cfg = Config(cfg_file='scepter/methods/studio/inference/largen/largen_pro.yaml')
largen_infer = LargenInference(logger)
largen_infer.init_from_cfg(cfg)

input_config = {
    "image": None,
    "original_size_as_tuple": [output_height, output_width],
    "target_size_as_tuple": [output_height, output_width],
    "aesthetic_score": 6.0,
    "negative_aesthetic_score": 2.5,
    # "prompt": "a photo of a backpack",
    "prompt": prompt,
    "negative_prompt": "",
    "prompt_prefix": "",
    "crop_coords_top_left": [0, 0],
    "sample": "ddim",
    "sample_steps": 50,
    "guide_scale": 7.5,
    "guide_rescale": 0,
    "discretization": "trailing",
    "refine_sample": "ddim",
    "refine_guide_scale": 7.5,
    "refine_guide_rescale": 0.5,
    "refine_discretization": "trailing",
}

for seed in range(100):
    seed_everything(seed)
    # start inference
    output = largen_infer(
        input=input_config,
        num_samples=1,
        intermediate_callback=None,
        refine_strength=0,
        cat_uc=True,
        largen_state=True,
        largen_task="Text_Subject_Guided_Inpainting",
        largen_image_scale=1,
        largen_tar_image=data[0],
        largen_tar_mask=data[1],
        largen_masked_image=data[2],
        largen_ref_image=data[3],
        largen_ref_mask=data[4],
        largen_ref_clip=data[5],
        largen_base_image=data[6],
        largen_extra_sizes=data[7],
        largen_bbox_yyxx=data[8],
    )

    save_image(output["images"], f"{prompt}_{seed}.png")
