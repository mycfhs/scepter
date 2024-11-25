# PYTHONPATH=. python scepter/tools/webui.py --cfg scepter/methods/studio/scepter_ui.yaml
from torchvision.utils import save_image
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger
from scepter.modules.inference.largen_inference import LargenInference
from scepter.studio.inference.inference_ui.largen_ui import LargenUI
import os
from PIL import Image
import numpy as np
import yaml
import shutil
from tqdm import tqdm
from pytorch_lightning import seed_everything

# 去scepter.scepter.modules.utils.distribute.__init__ 里面修改device_id

# 读取YAML文件
with open("/home/zl/yyc_workspace/Fooocus_diffusers/categories_final.yml", "r") as yml_file:
    data = yaml.safe_load(yml_file)
seed = 42
category_dict = data["category_dict"]
output_height, output_width = 1024, 1024
category_num = 30

tar_path = "/home/zl/yyc_workspace/Fooocus_diffusers/bg_images"
ref_path = "/home/zl/yyc_workspace/Fooocus_diffusers/train_data"

all_bg = [png[:-6] for png in os.listdir(tar_path) if png.endswith("_1.png")]
category_list = [i for i in os.listdir(ref_path) if i in category_dict.keys()]

# init file system - modelscope
# FS.TEMP_DIRinit_fs_client(Config(load=False, cfg_dict={'NAME': 'ModelscopeFs', 'TEMP_DIR': 'cache/data'}))
FS.TEMP_DIRinit_fs_client(
    Config(load=False, cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "cache/cache_data"})
)  # 新版本改名字了hhh。 ui里面保存到cache data。我们就用之前下载好的，不然得重新下载。 这个在scepter_ui.yaml里面
FS.TEMP_DIRinit_fs_client(
    Config(load=False, cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "cache/cache_data"})
) 

# init model config
logger = get_logger(name="scepter")
cfg = Config(cfg_file="scepter/methods/studio/inference/largen/largen_pro.yaml")
largen_infer = LargenInference(logger)
largen_infer.init_from_cfg(cfg)


for index, category in enumerate(tqdm(category_list)):
    # if index < 6:
    #     continue
    save_path = f"./output/lar_ti_edit/{category}"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    folder_path = os.path.join(ref_path, category)

    input_shape = Image.open(f"{ref_path}/{category}/image/00.jpg").size

    ref_image = Image.open(f"{ref_path}/{category}/image/00.jpg").convert("RGB")
    ref_mask = Image.open(f"{ref_path}/{category}/mask/00.png").convert("L")

    prompt_id = 0
    edit_prompt_num = len(category_dict[category]) - 1

    for img_name in all_bg[index::category_num]:
        # prompt = category_dict[category][0]
        prompt = category_dict[category][1 + prompt_id % edit_prompt_num]
        prompt_id += 1
        seed_everything(seed)
        prompt = prompt.replace("sks", "")
        
        img_name = str(img_name)

        tar_image = Image.open(f"{tar_path}/{img_name}_1.png").resize((1024, 1024)).convert("RGB")
        tar_mask = Image.open(f"{tar_path}/{img_name}_2.png").resize((1024, 1024)).convert("L")

        # ref_image = subject_image['background'].convert('RGB')
        # ref_mask = subject_image['layers'][0].split()[-1].convert('L')
        ref_image = np.asarray(ref_image)
        ref_mask = np.asarray(ref_mask)
        ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

        tar_image = np.asarray(tar_image)
        tar_mask = np.asarray(tar_mask)
        tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)

        (
            largen_tar_image,
            largen_tar_mask,
            largen_masked_image,
            largen_ref_image,
            largen_ref_mask,
            largen_ref_clip,
            largen_base_image,
            largen_extra_sizes,
            largen_bbox_yyxx,
        ) = LargenUI.data_preprocess_inpaint(
            None,
            tar_image,
            tar_mask,
            ref_image,
            ref_mask,
            True,
            1.3,
            output_height,
            output_width,
        )

        input_config = {
            "image": None,
            "original_size_as_tuple": input_shape,
            "target_size_as_tuple": [1024, 1024],
            "aesthetic_score": 6.0,
            "negative_aesthetic_score": 2.5,
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
        # start inference
        output = largen_infer(
            input=input_config,
            num_samples=1,
            intermediate_callback=None,
            refine_strength=0,
            cat_uc=True,
            largen_state=True,
            largen_task="Text_Subject_Guided_Inpainting",
            largen_image_scale=0.5, 
            largen_tar_image=largen_tar_image,
            largen_tar_mask=largen_tar_mask,
            largen_masked_image=largen_masked_image,
            largen_ref_image=largen_ref_image,
            largen_ref_mask=largen_ref_mask,
            largen_ref_clip=largen_ref_clip,
            largen_base_image=largen_base_image,
            largen_extra_sizes=largen_extra_sizes,
            largen_bbox_yyxx=largen_bbox_yyxx,
        )

        save_image(output["images"], f"./{save_path}/{img_name}+{prompt}+{seed}.jpg")
