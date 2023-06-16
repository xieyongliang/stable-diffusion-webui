import datetime
import json
import os
import sys
import threading
import time

import gradio as gr
import torch
import tqdm

import modules.interrogate
import modules.memmon
import modules.styles
import modules.devices as devices
from modules import localization, script_loading, errors, ui_components, shared_items, cmd_args
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir  # noqa: F401
from ldm.models.diffusion.ddpm import LatentDiffusion
from typing import Optional

import boto3
from botocore.exceptions import ClientError
import requests
import glob
from datetime import datetime, timedelta, timezone

demo = None

parser = cmd_args.parser

script_loading.preload_extensions(extensions_dir, parser)
script_loading.preload_extensions(extensions_builtin_dir, parser)

if os.environ.get('IGNORE_CMD_ARGS_ERRORS', None) is None:
    cmd_opts = parser.parse_args()
else:
    cmd_opts, _ = parser.parse_known_args()


restricted_opts = {
    "samples_filename_pattern",
    "directories_filename_pattern",
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_extras_samples",
    "outdir_grids",
    "outdir_txt2img_grids",
    "outdir_save",
    "outdir_init_images"
}

ui_reorder_categories = [
    "inpaint",
    "sampler",
    "checkboxes",
    "hires_fix",
    "dimensions",
    "cfg",
    "seed",
    "batch",
    "override_settings",
    "scripts",
]

# https://huggingface.co/datasets/freddyaboulton/gradio-theme-subdomains/resolve/main/subdomains.json
gradio_hf_hub_themes = [
    "gradio/glass",
    "gradio/monochrome",
    "gradio/seafoam",
    "gradio/soft",
    "freddyaboulton/dracula_revamped",
    "gradio/dracula_test",
    "abidlabs/dracula_test",
    "abidlabs/pakistan",
    "dawood/microsoft_windows",
    "ysharma/steampunk"
]


cmd_opts.disable_extension_access = (cmd_opts.share or cmd_opts.listen or cmd_opts.server_name) and not cmd_opts.enable_insecure_extension_access

devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
    (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])

devices.dtype = torch.float32 if cmd_opts.no_half else torch.float16
devices.dtype_vae = torch.float32 if cmd_opts.no_half or cmd_opts.no_half_vae else torch.float16

device = devices.device
weight_load_location = None if cmd_opts.lowram else "cpu"

batch_cond_uncond = cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram and not cmd_opts.medvram
xformers_available = False
config_filename = cmd_opts.ui_settings_file

os.makedirs(cmd_opts.hypernetwork_dir, exist_ok=True)
hypernetworks = {}
loaded_hypernetworks = []

if not cmd_opts.train:
    api_endpoint = os.environ['api_endpoint']
    industrial_model = ''
    default_options = {}
    sagemaker_endpoint_component = None
    sd_model_checkpoint_component = None
    create_train_dreambooth_component = None
    sd_hypernetwork_component = None
else:
    api_endpoint = cmd_opts.api_endpoint

response = requests.get(url=f'{api_endpoint}/sd/industrialmodel')
if response.status_code == 200:
    industrial_model = response.text
else:
    model_name = 'stable-diffusion-webui'
    model_description = model_name
    inputs = {
        'model_algorithm': 'stable-diffusion-webui',
        'model_name': model_name,
        'model_description': model_description,
        'model_extra': '{"visible": "false"}',
        'model_samples': '',
        'file_content': {
                'data': [(lambda x: int(x))(x) for x in open(os.path.join(script_path, 'logo.ico'), 'rb').read()]
        }
    }

    response = requests.post(url=f'{api_endpoint}/industrialmodel', json = inputs)
    if response.status_code == 200:
        body = json.loads(response.text)
        industrial_model = body['id']
    else:
        print(response.text)

def reload_embeddings(request: gr.Request):
    from modules import sd_hijack
    global embeddings

    embeddings = {}

    if cmd_opts.pureui:
        username = get_webui_username(request)
        params = {
            'module': 'embeddings',
            'username': username
        }
        response = requests.get(url=f'{api_endpoint}/sd/models', params=params)
        if response.status_code == 200:
            for embedding_item in json.loads(response.text):
                basename, fullname = os.path.split(embedding_item)
                embeddings[os.path.splitext(embedding_item)[0]] = f'/tmp/models/embeddings/{fullname}'
    else:
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
        embeddings = sd_hijack.model_hijack.embedding_db.word_embeddings.values()


def reload_hypernetworks(request: gr.Request = None):
    from modules.hypernetworks import hypernetwork
    global hypernetworks

    hypernetworks = {}

    if cmd_opts.pureui:
        if request:
            username = get_webui_username(request)
            params = {
                'module': 'hypernetwork',
                'username': username
            }
            response = requests.get(url=f'{api_endpoint}/sd/models', params=params)
            if response.status_code == 200:
                for hypernetwork_item in json.loads(response.text):
                    basename, fullname = os.path.split(hypernetwork_item)
                    hypernetworks[os.path.splitext(hypernetwork_item)[0]] = f'/tmp/models/hypernetworks/{fullname}'
    else:
        hypernetworks = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)
        hypernetwork.load_hypernetwork(opts.sd_hypernetwork)

class State:
    skipped = False
    interrupted = False
    job = ""
    job_no = 0
    job_count = 0
    processing_has_refined_job_count = False
    job_timestamp = '0'
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0
    id_live_preview = 0
    textinfo = None
    time_start = None
    server_start = None
    _server_command_signal = threading.Event()
    _server_command: Optional[str] = None

    @property
    def need_restart(self) -> bool:
        # Compatibility getter for need_restart.
        return self.server_command == "restart"

    @need_restart.setter
    def need_restart(self, value: bool) -> None:
        # Compatibility setter for need_restart.
        if value:
            self.server_command = "restart"

    @property
    def server_command(self):
        return self._server_command

    @server_command.setter
    def server_command(self, value: Optional[str]) -> None:
        """
        Set the server command to `value` and signal that it's been set.
        """
        self._server_command = value
        self._server_command_signal.set()

    def wait_for_server_command(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Wait for server command to get set; return and clear the value and signal.
        """
        if self._server_command_signal.wait(timeout):
            self._server_command_signal.clear()
            req = self._server_command
            self._server_command = None
            return req
        return None

    def request_restart(self) -> None:
        self.interrupt()
        self.server_command = "restart"

    def skip(self):
        self.skipped = True

    def interrupt(self):
        self.interrupted = True

    def nextjob(self):
        if opts.live_previews_enable and opts.show_progress_every_n_steps == -1:
            self.do_set_current_image()

        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0

    def dict(self):
        obj = {
            "skipped": self.skipped,
            "interrupted": self.interrupted,
            "job": self.job,
            "job_count": self.job_count,
            "job_timestamp": self.job_timestamp,
            "job_no": self.job_no,
            "sampling_step": self.sampling_step,
            "sampling_steps": self.sampling_steps,
        }

        return obj

    def begin(self):
        self.sampling_step = 0
        self.job_count = -1
        self.processing_has_refined_job_count = False
        self.job_no = 0
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.current_latent = None
        self.current_image = None
        self.current_image_sampling_step = 0
        self.id_live_preview = 0
        self.skipped = False
        self.interrupted = False
        self.textinfo = None
        self.time_start = time.time()

        devices.torch_gc()

    def end(self):
        self.job = ""
        self.job_count = 0

        devices.torch_gc()

    def set_current_image(self):
        """sets self.current_image from self.current_latent if enough sampling steps have been made after the last call to this"""
        if not parallel_processing_allowed:
            return

        if self.sampling_step - self.current_image_sampling_step >= opts.show_progress_every_n_steps and opts.live_previews_enable and opts.show_progress_every_n_steps != -1:
            self.do_set_current_image()

    def do_set_current_image(self):
        if self.current_latent is None:
            return

        import modules.sd_samplers
        if opts.show_progress_grid:
            self.assign_current_image(modules.sd_samplers.samples_to_image_grid(self.current_latent))
        else:
            self.assign_current_image(modules.sd_samplers.sample_to_image(self.current_latent))

        self.current_image_sampling_step = self.sampling_step

    def assign_current_image(self, image):
        self.current_image = image
        self.id_live_preview += 1


state = State()
state.server_start = time.time()

styles_filename = cmd_opts.styles_file
prompt_styles = modules.styles.StyleDatabase(styles_filename)

interrogator = modules.interrogate.InterrogateModels("interrogate")

face_restorers = []


class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None, comment_before='', comment_after=''):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.refresh = refresh

        self.comment_before = comment_before
        """HTML text that will be added after label in UI"""

        self.comment_after = comment_after
        """HTML text that will be added before label in UI"""

    def link(self, label, url):
        self.comment_before += f"[<a href='{url}' target='_blank'>{label}</a>]"
        return self

    def js(self, label, js_func):
        self.comment_before += f"[<a onclick='{js_func}(); return false'>{label}</a>]"
        return self

    def info(self, info):
        self.comment_after += f"<span class='info'>({info})</span>"
        return self

    def needs_restart(self):
        self.comment_after += " <span class='info'>(requires restart)</span>"
        return self




def options_section(section_identifier, options_dict):
    for v in options_dict.values():
        v.section = section_identifier

    return options_dict


def list_checkpoint_tiles():
    import modules.sd_models
    return modules.sd_models.checkpoint_tiles()


def refresh_checkpoints(sagemaker_endpoint=None,username=''):
    import modules.sd_models
    modules.sd_models.list_models(sagemaker_endpoint,username)
    checkpoints = modules.sd_models.checkpoints_list
    return checkpoints


def list_samplers():
    import modules.sd_samplers
    return modules.sd_samplers.all_samplers


hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}
tab_names = []

options_templates = {}

options_templates = {}

sagemaker_endpoints = []

sd_models = []

def list_sagemaker_endpoints():
    global sagemaker_endpoints

    return sagemaker_endpoints

def list_sd_models():
    global sd_models

    return sd_models + [ x['filename'] for x in huggingface_models ]

def intersection(lst1, lst2):
    set1 = set(lst1)
    set2 = set(lst2)

    intersec = set1.intersection(set2)
    return list(intersec)

def get_available_sagemaker_endpoints(item):
    attrs = item.get('attributes', '')
    if attrs == '':
        return ''

    return attrs.get('sagemaker_endpoints', '')

def refresh_sagemaker_endpoints(username):
    global industrial_model, api_endpoint, sagemaker_endpoints

    sagemaker_endpoints = []

    if not username:
        return sagemaker_endpoints

    if industrial_model != '':
        params = {
            'industrial_model': industrial_model
        }
        response = requests.get(url=f'{api_endpoint}/endpoint', params=params)
        if response.status_code == 200:
            for endpoint_item in json.loads(response.text):
                sagemaker_endpoints.append(endpoint_item['EndpointName'])

    # to filter user's available endpoints
    inputs = {
        'action': 'get',
        'username': username
    }
    response = requests.post(url=f'{api_endpoint}/sd/user', json=inputs)
    if response.status_code == 200 and response.text != '':
        data = json.loads(response.text)
        eps = get_available_sagemaker_endpoints(data)
        if eps != '':
            sagemaker_endpoints = intersection(eps.split(','), sagemaker_endpoints)

    return sagemaker_endpoints

def refresh_sd_models(username):
    global api_endpoint, sd_models

    names = set()

    if not username:
        return sd_models

    params = {
        'module': 'sd_models'
    }
    params['username'] = username

    response = requests.get(url=f'{api_endpoint}/sd/models', params=params)
    if response.status_code == 200:
        model_list = json.loads(response.text)
        for model in model_list:
            names.add(model)

    sd_models = list(names)

    return sd_models

options_templates.update(options_section(('saving-images', "Saving images/grids"), {
    "samples_save": OptionInfo(True, "Always save all generated images"),
    "samples_format": OptionInfo('png', 'File format for images'),
    "samples_filename_pattern": OptionInfo("", "Images filename pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    "save_images_add_number": OptionInfo(True, "Add number to filename when saving", component_args=hide_dirs),

    "grid_save": OptionInfo(True, "Always save all generated image grids"),
    "grid_format": OptionInfo('png', 'File format for grids'),
    "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
    "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
    "grid_prevent_empty_spots": OptionInfo(False, "Prevent empty spots in grid (when set to autodetect)"),
    "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),

    "enable_pnginfo": OptionInfo(True, "Save text information about generation parameters as chunks to png files"),
    "save_txt": OptionInfo(False, "Create a text file next to every image with generation parameters."),
    "save_images_before_face_restoration": OptionInfo(False, "Save a copy of image before doing face restoration."),
    "save_images_before_highres_fix": OptionInfo(False, "Save a copy of image before applying highres fix."),
    "save_images_before_color_correction": OptionInfo(False, "Save a copy of image before applying color correction to img2img results"),
    "save_mask": OptionInfo(False, "For inpainting, save a copy of the greyscale mask"),
    "save_mask_composite": OptionInfo(False, "For inpainting, save a masked composite"),
    "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "webp_lossless": OptionInfo(False, "Use lossless compression for webp images"),
    "export_for_4chan": OptionInfo(True, "Save copy of large images as JPG").info("if the file size is above the limit, or either width or height are above the limit"),
    "img_downscale_threshold": OptionInfo(4.0, "File size limit for the above option, MB", gr.Number),
    "target_side_length": OptionInfo(4000, "Width/height limit for the above option, in pixels", gr.Number),
    "img_max_size_mp": OptionInfo(200, "Maximum image size", gr.Number).info("in megapixels"),

    "use_original_name_batch": OptionInfo(True, "Use original name for output filename during batch process in extras tab"),
    "use_upscaler_name_as_suffix": OptionInfo(False, "Use upscaler name as filename suffix in the extras tab"),
    "save_selected_only": OptionInfo(True, "When using 'Save' button, only save a single selected image"),
    "save_init_img": OptionInfo(False, "Save init images when using img2img"),

    "temp_dir":  OptionInfo("", "Directory for temporary images; leave empty for default"),
    "clean_temp_dir_at_start": OptionInfo(False, "Cleanup non-default temporary directory when starting webui"),

}))

options_templates.update(options_section(('saving-paths', "Paths for saving"), {
    "outdir_samples": OptionInfo("", "Output directory for images; if empty, defaults to three directories below", component_args=hide_dirs),
    "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'Output directory for txt2img images', component_args=hide_dirs),
    "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'Output directory for img2img images', component_args=hide_dirs),
    "outdir_extras_samples": OptionInfo("outputs/extras-images", 'Output directory for images from extras tab', component_args=hide_dirs),
    "outdir_grids": OptionInfo("", "Output directory for grids; if empty, defaults to two directories below", component_args=hide_dirs),
    "outdir_txt2img_grids": OptionInfo("outputs/txt2img-grids", 'Output directory for txt2img grids', component_args=hide_dirs),
    "outdir_img2img_grids": OptionInfo("outputs/img2img-grids", 'Output directory for img2img grids', component_args=hide_dirs),
    "outdir_save": OptionInfo("log/images", "Directory for saving images using the Save button", component_args=hide_dirs),
    "outdir_init_images": OptionInfo("outputs/init-images", "Directory for saving init images when using img2img", component_args=hide_dirs),
}))

options_templates.update(options_section(('saving-to-dirs', "Saving to a directory"), {
    "save_to_dirs": OptionInfo(True, "Save images to a subdirectory"),
    "grid_save_to_dirs": OptionInfo(True, "Save grids to a subdirectory"),
    "use_save_to_dirs_for_ui": OptionInfo(False, "When using \"Save\" button, save images to a subdirectory"),
    "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1, **hide_dirs}),
}))

options_templates.update(options_section(('upscaling', "Upscaling"), {
    "ESRGAN_tile": OptionInfo(192, "Tile size for ESRGAN upscalers.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}).info("0 = no tiling"),
    "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap for ESRGAN upscalers.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}).info("Low values = visible seam"),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"], "Select which Real-ESRGAN models to show in the web UI.", gr.CheckboxGroup, lambda: {"choices": shared_items.realesrgan_models_names()}),
    "upscaler_for_img2img": OptionInfo(None, "Upscaler for img2img", gr.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers]}),
}))

options_templates.update(options_section(('face-restoration', "Face restoration"), {
    "face_restoration_model": OptionInfo("CodeFormer", "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
    "code_former_weight": OptionInfo(0.5, "CodeFormer weight", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}).info("0 = maximum effect; 1 = minimum effect"),
    "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
}))

options_templates.update(options_section(('system', "System"), {
    "show_warnings": OptionInfo(False, "Show warnings in console."),
    "memmon_poll_rate": OptionInfo(8, "VRAM usage polls per second during generation.", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}).info("0 = disable"),
    "samples_log_stdout": OptionInfo(False, "Always print all generation info to standard output"),
    "multiple_tqdm": OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job."),
    "print_hypernet_extra": OptionInfo(False, "Print extra hypernetwork information to console."),
    "list_hidden_files": OptionInfo(True, "Load models/files in hidden directories").info("directory is hidden if its name starts with \".\""),
}))

options_templates.update(options_section(('training', "Training"), {
    "unload_models_when_training": OptionInfo(False, "Move VAE and CLIP to RAM when training if possible. Saves VRAM."),
    "pin_memory": OptionInfo(False, "Turn on pin_memory for DataLoader. Makes training slightly faster but can increase memory usage."),
    "save_optimizer_state": OptionInfo(False, "Saves Optimizer state as separate *.optim file. Training of embedding or HN can be resumed with the matching optim file."),
    "save_training_settings_to_txt": OptionInfo(True, "Save textual inversion and hypernet settings to a text file whenever training starts."),
    "dataset_filename_word_regex": OptionInfo("", "Filename word regex"),
    "dataset_filename_join_string": OptionInfo(" ", "Filename join string"),
    "training_image_repeats_per_epoch": OptionInfo(1, "Number of repeats for a single input image per epoch; used only for displaying epoch number", gr.Number, {"precision": 0}),
    "training_write_csv_every": OptionInfo(500, "Save an csv containing the loss to log directory every N steps, 0 to disable"),
    "training_xattention_optimizations": OptionInfo(False, "Use cross attention optimizations while training"),
    "training_enable_tensorboard": OptionInfo(False, "Enable tensorboard logging."),
    "training_tensorboard_save_images": OptionInfo(False, "Save generated images within tensorboard."),
    "training_tensorboard_flush_every": OptionInfo(120, "How often, in seconds, to flush the pending tensorboard events and summaries to disk."),
}))

options_templates.update(options_section(('sd', "Stable Diffusion"), {
    "sagemaker_endpoint": OptionInfo(None, "SaegMaker endpoint", gr.Dropdown, lambda: {"choices": list_sagemaker_endpoints()}, refresh=refresh_sagemaker_endpoints),
    "sd_model_checkpoint": OptionInfo(None, "Stable Diffusion checkpoint", gr.Dropdown, lambda: {"choices": list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "sd_checkpoint_cache": OptionInfo(0, "Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_vae_checkpoint_cache": OptionInfo(0, "VAE Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_vae": OptionInfo("Automatic", "SD VAE", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list).info("choose VAE model: Automatic = use one with same filename as checkpoint; None = use VAE from checkpoint"),
    "sd_vae_as_default": OptionInfo(True, "Ignore selected VAE for stable diffusion checkpoints that have their own .vae.pt next to them"),
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for img2img", gr.Slider, {"minimum": 0.5, "maximum": 1.5, "step": 0.01}),
    "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors."),
    "img2img_fix_steps": OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies.").info("normally you'd do less with less denoising"),
    "img2img_background_color": OptionInfo("#ffffff", "With img2img, fill image's transparent parts with this color.", ui_components.FormColorPicker, {}),
    "enable_quantization": OptionInfo(False, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds. Requires restart to apply."),
    "enable_emphasis": OptionInfo(True, "Enable emphasis").info("use (text) to make model pay more attention to text and [text] to make it pay less attention"),
    "enable_batch_seeds": OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image"),
    "comma_padding_backtrack": OptionInfo(20, "Prompt word wrap length limit", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1}).info("in tokens - for texts shorter than specified, if they don't fit into 75 token limit, move them to the next 75 token chunk"),
    "CLIP_stop_at_last_layers": OptionInfo(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#clip-skip").info("ignore last layers of CLIP nrtwork; 1 ignores none, 2 ignores one layer"),
    "upcast_attn": OptionInfo(False, "Upcast cross attention layer to float32"),
    "randn_source": OptionInfo("GPU", "Random number generator source.", gr.Radio, {"choices": ["GPU", "CPU"]}).info("changes seeds drastically; use CPU to produce the same picture across different vidocard vendors"),
}))

options_templates.update(options_section(('optimizations', "Optimizations"), {
    "cross_attention_optimization": OptionInfo("Automatic", "Cross attention optimization", gr.Dropdown, lambda: {"choices": shared_items.cross_attention_optimizations()}),
    "s_min_uncond": OptionInfo(0, "Negative Guidance minimum sigma", gr.Slider, {"minimum": 0.0, "maximum": 4.0, "step": 0.01}).link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9177").info("skip negative prompt for some steps when the image is almost ready; 0=disable, higher=faster"),
    "token_merging_ratio": OptionInfo(0.0, "Token merging ratio", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}).link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9256").info("0=disable, higher=faster"),
    "token_merging_ratio_img2img": OptionInfo(0.0, "Token merging ratio for img2img", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}).info("only applies if non-zero and overrides above"),
    "token_merging_ratio_hr": OptionInfo(0.0, "Token merging ratio for high-res pass", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}).info("only applies if non-zero and overrides above"),
}))

options_templates.update(options_section(('compatibility', "Compatibility"), {
    "use_old_emphasis_implementation": OptionInfo(False, "Use old emphasis implementation. Can be useful to reproduce old seeds."),
    "use_old_karras_scheduler_sigmas": OptionInfo(False, "Use old karras scheduler sigmas (0.1 to 10)."),
    "no_dpmpp_sde_batch_determinism": OptionInfo(False, "Do not make DPM++ SDE deterministic across different batch sizes."),
    "use_old_hires_fix_width_height": OptionInfo(False, "For hires fix, use width/height sliders to set final resolution rather than first pass (disables Upscale by, Resize width/height to)."),
    "dont_fix_second_order_samplers_schedule": OptionInfo(False, "Do not fix prompt schedule for second order samplers."),
}))

options_templates.update(options_section(('interrogate', "Interrogate Options"), {
    "interrogate_keep_models_in_memory": OptionInfo(False, "Keep models in VRAM"),
    "interrogate_return_ranks": OptionInfo(False, "Include ranks of model tags matches in results.").info("booru only"),
    "interrogate_clip_num_beams": OptionInfo(1, "BLIP: num_beams", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length": OptionInfo(24, "BLIP: minimum description length", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length": OptionInfo(48, "BLIP: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_dict_limit": OptionInfo(1500, "CLIP: maximum number of lines in text file").info("0 = No limit"),
    "interrogate_clip_skip_categories": OptionInfo([], "CLIP: skip inquire categories", gr.CheckboxGroup, lambda: {"choices": modules.interrogate.category_types()}, refresh=modules.interrogate.category_types),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.5, "deepbooru: score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(True, "deepbooru: sort tags alphabetically").info("if not: sort by score"),
    "deepbooru_use_spaces": OptionInfo(True, "deepbooru: use spaces in tags").info("if not: use underscores"),
    "deepbooru_escape": OptionInfo(True, "deepbooru: escape (\\) brackets").info("so they are used as literal brackets and not for emphasis"),
    "deepbooru_filter_tags": OptionInfo("", "deepbooru: filter out those tags").info("separate by comma"),
}))

options_templates.update(options_section(('extra_networks', "Extra Networks"), {
    "extra_networks_show_hidden_directories": OptionInfo(True, "Show hidden directories").info("directory is hidden if its name starts with \".\"."),
    "extra_networks_hidden_models": OptionInfo("When searched", "Show cards for models in hidden directories", gr.Radio, {"choices": ["Always", "When searched", "Never"]}).info('"When searched" option will only show the item when the search string has 4 characters or more'),
    "extra_networks_default_view": OptionInfo("cards", "Default view for Extra Networks", gr.Dropdown, {"choices": ["cards", "thumbs"]}),
    "extra_networks_default_multiplier": OptionInfo(1.0, "Multiplier for extra networks", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "extra_networks_card_width": OptionInfo(0, "Card width for Extra Networks").info("in pixels"),
    "extra_networks_card_height": OptionInfo(0, "Card height for Extra Networks").info("in pixels"),
    "extra_networks_add_text_separator": OptionInfo(" ", "Extra networks separator").info("extra text to add before <...> when adding extra network to prompt"),
    "ui_extra_networks_tab_reorder": OptionInfo("", "Extra networks tab order").needs_restart(),
    "sd_hypernetwork": OptionInfo("None", "Add hypernetwork to prompt", gr.Dropdown, lambda: {"choices": ["None", *hypernetworks]}, refresh=reload_hypernetworks),
}))

options_templates.update(options_section(('ui', "User interface"), {
    "localization": OptionInfo("None", "Localization", gr.Dropdown, lambda: {"choices": ["None"] + list(localization.localizations.keys())}, refresh=lambda: localization.list_localizations(cmd_opts.localizations_dir)).needs_restart(),
    "gradio_theme": OptionInfo("Default", "Gradio theme", ui_components.DropdownEditable, lambda: {"choices": ["Default"] + gradio_hf_hub_themes}).needs_restart(),
    "img2img_editor_height": OptionInfo(720, "img2img: height of image editor", gr.Slider, {"minimum": 80, "maximum": 1600, "step": 1}).info("in pixels").needs_restart(),
    "return_grid": OptionInfo(True, "Show grid in results for web"),
    "return_mask": OptionInfo(False, "For inpainting, include the greyscale mask in results for web"),
    "return_mask_composite": OptionInfo(False, "For inpainting, include masked composite in results for web"),
    "do_not_show_images": OptionInfo(False, "Do not show any images in results for web"),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    "send_size": OptionInfo(True, "Send size when sending prompt or image to another interface"),
    "font": OptionInfo("", "Font for image grids that have text"),
    "js_modal_lightbox": OptionInfo(True, "Enable full page image viewer"),
    "js_modal_lightbox_initially_zoomed": OptionInfo(True, "Show images zoomed in by default in full page image viewer"),
    "js_modal_lightbox_gamepad": OptionInfo(False, "Navigate image viewer with gamepad"),
    "js_modal_lightbox_gamepad_repeat": OptionInfo(250, "Gamepad repeat period, in milliseconds"),
    "show_progress_in_title": OptionInfo(True, "Show generation progress in window title."),
    "samplers_in_dropdown": OptionInfo(True, "Use dropdown for sampler selection instead of radio group").needs_restart(),
    "dimensions_and_batch_together": OptionInfo(True, "Show Width/Height and Batch sliders in same row").needs_restart(),
    "keyedit_precision_attention": OptionInfo(0.1, "Ctrl+up/down precision when editing (attention:1.1)", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "keyedit_precision_extra": OptionInfo(0.05, "Ctrl+up/down precision when editing <extra networks:0.9>", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "keyedit_delimiters": OptionInfo(".,\\/!?%^*;:{}=`~()", "Ctrl+up/down word delimiters"),
    "quicksettings_list": OptionInfo([""], "Quicksettings list", ui_components.DropdownMulti, lambda: {"choices": list(opts.data_labels.keys())}).js("info", "settingsHintsShowQuicksettings").info("setting entries that appear at the top of page rather than in settings tab").needs_restart(),
    "ui_tab_order": OptionInfo([], "UI tab order", ui_components.DropdownMulti, lambda: {"choices": list(tab_names)}).needs_restart(),
    "hidden_tabs": OptionInfo([], "Hidden UI tabs", ui_components.DropdownMulti, lambda: {"choices": list(tab_names)}).needs_restart(),
    "ui_reorder": OptionInfo(", ".join(ui_reorder_categories), "txt2img/img2img UI item order").needs_restart(),
    "hires_fix_show_sampler": OptionInfo(False, "Hires fix: show hires sampler selection").needs_restart(),
    "hires_fix_show_prompts": OptionInfo(False, "Hires fix: show hires prompt and negative prompt").needs_restart(),
}))

options_templates.update(options_section(('infotext', "Infotext"), {
    "add_model_hash_to_info": OptionInfo(True, "Add model hash to generation information"),
    "add_model_name_to_info": OptionInfo(True, "Add model name to generation information"),
    "add_version_to_infotext": OptionInfo(True, "Add program version to generation information"),
    "disable_weights_auto_swap": OptionInfo(True, "When reading generation parameters from text into UI (from PNG info or pasted text), do not change the selected model/checkpoint."),
}))

options_templates.update(options_section(('ui', "Live previews"), {
    "show_progressbar": OptionInfo(True, "Show progressbar"),
    "live_previews_enable": OptionInfo(True, "Show live previews of the created image"),
    "live_previews_image_format": OptionInfo("png", "Live preview file format", gr.Radio, {"choices": ["jpeg", "png", "webp"]}),
    "show_progress_grid": OptionInfo(True, "Show previews of all images generated in a batch as a grid"),
    "show_progress_every_n_steps": OptionInfo(10, "Live preview display period", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}).info("in sampling steps - show new live preview image every N sampling steps; -1 = only show after completion of batch"),
    "show_progress_type": OptionInfo("Approx NN", "Live preview method", gr.Radio, {"choices": ["Full", "Approx NN", "Approx cheap", "TAESD"]}).info("Full = slow but pretty; Approx NN and TAESD = fast but low quality; Approx cheap = super fast but terrible otherwise"),
    "live_preview_content": OptionInfo("Prompt", "Live preview subject", gr.Radio, {"choices": ["Combined", "Prompt", "Negative prompt"]}),
    "live_preview_refresh_period": OptionInfo(1000, "Progressbar and preview update period").info("in milliseconds"),
}))

options_templates.update(options_section(('sampler-params', "Sampler parameters"), {
    "hide_samplers": OptionInfo([], "Hide samplers in user interface", gr.CheckboxGroup, lambda: {"choices": [x.name for x in list_samplers()]}).needs_restart(),
    "eta_ddim": OptionInfo(0.0, "Eta for DDIM", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}).info("noise multiplier; higher = more unperdictable results"),
    "eta_ancestral": OptionInfo(1.0, "Eta for ancestral samplers", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}).info("noise multiplier; applies to Euler a and other samplers that have a in them"),
    "ddim_discretize": OptionInfo('uniform', "img2img DDIM discretize", gr.Radio, {"choices": ['uniform', 'quad']}),
    's_churn': OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_tmin':  OptionInfo(0.0, "sigma tmin",  gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_noise': OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    'eta_noise_seed_delta': OptionInfo(0, "Eta noise seed delta", gr.Number, {"precision": 0}).info("ENSD; does not improve anything, just produces different results for ancestral samplers - only useful for reproducing images"),
    'always_discard_next_to_last_sigma': OptionInfo(False, "Always discard next-to-last sigma").link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6044"),
    'uni_pc_variant': OptionInfo("bh1", "UniPC variant", gr.Radio, {"choices": ["bh1", "bh2", "vary_coeff"]}),
    'uni_pc_skip_type': OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ["time_uniform", "time_quadratic", "logSNR"]}),
    'uni_pc_order': OptionInfo(3, "UniPC order", gr.Slider, {"minimum": 1, "maximum": 50, "step": 1}).info("must be < sampling steps"),
    'uni_pc_lower_order_final': OptionInfo(True, "UniPC lower order final"),
}))

options_templates.update(options_section(('postprocessing', "Postprocessing"), {
    'postprocessing_enable_in_main_ui': OptionInfo([], "Enable postprocessing operations in txt2img and img2img tabs", ui_components.DropdownMulti, lambda: {"choices": ['CodeFormer', 'GFPGAN', 'Upscale']}),
    'postprocessing_operation_order': OptionInfo([], "Postprocessing operation order", ui_components.DropdownMulti, lambda: {"choices": ['CodeFormer', 'GFPGAN', 'Upscale']}),
    'upscaling_max_images_in_cache': OptionInfo(5, "Maximum number of images in upscaling cache", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
}))

options_templates.update(options_section((None, "Hidden options"), {
    "disabled_extensions": OptionInfo([], "Disable these extensions"),
    "disable_all_extensions": OptionInfo("none", "Disable all extensions (preserves the list of disabled extensions)", gr.Radio, {"choices": ["none", "extra", "all"]}),
    "restore_config_state_file": OptionInfo("", "Config state file to restore from, under 'config-states/' folder"),
    "sd_checkpoint_hash": OptionInfo("", "SHA256 hash of the current checkpoint"),
}))


options_templates.update()


class Options:
    data = None
    data_labels = options_templates
    typemap = {int: float}

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value):
        if self.data is not None:
            if key in self.data or key in self.data_labels:
                assert not cmd_opts.freeze_settings, "changing settings is disabled"

                info = opts.data_labels.get(key, None)
                comp_args = info.component_args if info else None
                if isinstance(comp_args, dict) and comp_args.get('visible', True) is False:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                if cmd_opts.hide_ui_dir_config and key in restricted_opts:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                self.data[key] = value
                return

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super(Options, self).__getattribute__(item)

    def set(self, key, value):
        """sets an option and calls its onchange callback, returning True if the option changed and False otherwise"""

        oldval = self.data.get(key, None)
        if oldval == value:
            return False

        try:
            setattr(self, key, value)
        except RuntimeError:
            return False

        if self.data_labels[key].onchange is not None:
            try:
                self.data_labels[key].onchange()
            except Exception as e:
                errors.display(e, f"changing setting {key} to {value}")
                setattr(self, key, oldval)
                return False

        return True

    def get_default(self, key):
        """returns the default value for the key"""

        data_label = self.data_labels.get(key)
        if data_label is None:
            return None

        return data_label.default

    def save(self, filename):
        assert not cmd_opts.pureui and not cmd_opts.freeze_settings, "saving settings is disabled"

        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file, indent=4)

    def same_type(self, x, y):
        if x is None or y is None:
            return True

        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))

        return type_x == type_y

    def load(self, filename):
        assert not cmd_opts.pureui

        with open(filename, "r", encoding="utf8") as file:
            self.data = json.load(file)

        # 1.1.1 quicksettings list migration
        if self.data.get('quicksettings') is not None and self.data.get('quicksettings_list') is None:
            self.data['quicksettings_list'] = [i.strip() for i in self.data.get('quicksettings').split(',')]

        bad_settings = 0
        for k, v in self.data.items():
            info = self.data_labels.get(k, None)
            if info is not None and not self.same_type(info.default, v):
                print(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})", file=sys.stderr)
                bad_settings += 1

        if bad_settings > 0:
            print(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.", file=sys.stderr)

    def onchange(self, key, func, call=True):
        item = self.data_labels.get(key)
        item.onchange = func

        if call:
            func()

    def dumpjson(self):
        d = {k: self.data.get(k, v.default) for k, v in self.data_labels.items()}
        d["_comments_before"] = {k: v.comment_before for k, v in self.data_labels.items() if v.comment_before is not None}
        d["_comments_after"] = {k: v.comment_after for k, v in self.data_labels.items() if v.comment_after is not None}
        return json.dumps(d)

    def add_option(self, key, info):
        self.data_labels[key] = info

    def reorder(self):
        """reorder settings so that all items related to section always go together"""

        section_ids = {}
        settings_items = self.data_labels.items()
        for _, item in settings_items:
            if item.section not in section_ids:
                section_ids[item.section] = len(section_ids)

        self.data_labels = dict(sorted(settings_items, key=lambda x: section_ids[x[1].section]))

    def cast_value(self, key, value):
        """casts an arbitrary to the same type as this setting's value with key
        Example: cast_value("eta_noise_seed_delta", "12") -> returns 12 (an int rather than str)
        """

        if value is None:
            return None

        default_value = self.data_labels[key].default
        if default_value is None:
            default_value = getattr(self, key, None)
        if default_value is None:
            return None

        expected_type = type(default_value)
        if expected_type == bool and value == "False":
            value = False
        else:
            value = expected_type(value)

        return value


opts = Options()
if os.path.exists(config_filename):
    opts.load(config_filename)


class Shared(sys.modules[__name__].__class__):
    """
    this class is here to provide sd_model field as a property, so that it can be created and loaded on demand rather than
    at program startup.
    """

    sd_model_val = None

    @property
    def sd_model(self):
        import modules.sd_models

        return modules.sd_models.model_data.get_sd_model()

    @sd_model.setter
    def sd_model(self, value):
        import modules.sd_models

        modules.sd_models.model_data.set_sd_model(value)


sd_model: LatentDiffusion = None  # this var is here just for IDE's type checking; it cannot be accessed because the class field above will be accessed instead
sys.modules[__name__].__class__ = Shared

settings_components = None
"""assinged from ui.py, a mapping on setting names to gradio components repsponsible for those settings"""

latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

if cmd_opts.pureui and opts.localization == "None":
    opts.localization = "zh_CN"

sd_upscalers = []

clip_model = None

progress_print_out = sys.stdout

gradio_theme = gr.themes.Base()


def reload_gradio_theme(theme_name=None):
    global gradio_theme
    if not theme_name:
        theme_name = opts.gradio_theme

    default_theme_args = dict(
        font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
        font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
    )

    if theme_name == "Default":
        gradio_theme = gr.themes.Default(**default_theme_args)
    else:
        try:
            gradio_theme = gr.themes.ThemeClass.from_hub(theme_name)
        except Exception as e:
            errors.display(e, "changing gradio theme")
            gradio_theme = gr.themes.Default(**default_theme_args)



class TotalTQDM:
    def __init__(self):
        self._tqdm = None

    def reset(self):
        self._tqdm = tqdm.tqdm(
            desc="Total progress",
            total=state.job_count * state.sampling_steps,
            position=1,
            file=progress_print_out
        )

    def update(self):
        if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.update()

    def updateTotal(self, new_total):
        if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.total = new_total

    def clear(self):
        if self._tqdm is not None:
            self._tqdm.refresh()
            self._tqdm.close()
            self._tqdm = None


total_tqdm = TotalTQDM()

mem_mon = modules.memmon.MemUsageMonitor("MemMon", device, opts)
mem_mon.start()


def listfiles(dirname):
    filenames = [os.path.join(dirname, x) for x in sorted(os.listdir(dirname), key=str.lower) if not x.startswith(".")]
    return [file for file in filenames if os.path.isfile(file)]


def html_path(filename):
    return os.path.join(script_path, "html", filename)


def html(filename):
    path = html_path(filename)

    if os.path.exists(path):
        with open(path, encoding="utf8") as file:
            return file.read()

    return ""


def walk_files(path, allowed_extensions=None):
    if not os.path.exists(path):
        return

    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)

    for root, _, files in os.walk(path, followlinks=True):
        for filename in files:
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext not in allowed_extensions:
                    continue

            if not opts.list_hidden_files and ("/." in root or "\\." in root):
                continue

            yield os.path.join(root, filename)

models_s3_bucket = None
s3_folder_sd = None
s3_folder_cn = None
s3_folder_lora = None
s3_folder_vae = None
syncLock = threading.Lock()
sync_images_lock = threading.Lock()
tmp_models_dir = '/tmp/models'
tmp_cache_dir = '/tmp/model_sync_cache'

def get_cookies(request):
    # request.headers is of type Gradio.queue.Obj, can't be subscripted
    # directly, so we need to retrieve its underlying dict first.
    cookies = request.headers.__dict__['cookie'].split('; ')
    return cookies

def get_webui_username(request):
    tokens = demo.server_app.tokens
    cookies = request.headers.__dict__['cookie'].split('; ')
    access_token = None
    for cookie in cookies:
        if cookie.startswith('access-token'):
            access_token = cookie[len('access-token=') : ]
            break
    if access_token.startswith('unsecure='):
        access_token = access_token[len('unsecure=') : ]
    username = tokens[access_token] if access_token else None
    return username

huggingface_models = [
    {
        'repo_id': 'stabilityai/stable-diffusion-2-1',
        'filename': 'v2-1_768-ema-pruned.ckpt',
    },
    {
        'repo_id': 'stabilityai/stable-diffusion-2-1',
        'filename': 'v2-1_768-nonema-pruned.ckpt',
    },
    {
        'repo_id': 'stabilityai/stable-diffusion-2',
        'filename': '768-v-ema.ckpt',
    },
    {
        'repo_id': 'runwayml/stable-diffusion-v1-5',
        'filename': 'v1-5-pruned-emaonly.ckpt',          
    },
    {
        'repo_id': 'runwayml/stable-diffusion-v1-5',
        'filename': 'v1-5-pruned.ckpt',          
    },
    {
        'repo_id': 'CompVis/stable-diffusion-v-1-4-original',
        'filename': 'sd-v1-4.ckpt',          
    },
    {
        'repo_id': 'CompVis/stable-diffusion-v-1-4-original',
        'filename': 'sd-v1-4-full-ema.ckpt',          
    },
]

cache = dict()
region_name = boto3.session.Session().region_name if not cmd_opts.train else cmd_opts.region_name
s3_client = boto3.client('s3', region_name=region_name)
endpointUrl = s3_client.meta.endpoint_url
s3_client = boto3.client('s3', endpoint_url=endpointUrl, region_name=region_name)
s3_resource= boto3.resource('s3')
generated_images_s3uri = os.environ.get('generated_images_s3uri', None)

def get_default_bucket():
    region_name = boto3.Session().region_name
    account_id = boto3.Session().client('sts').get_caller_identity()['Account']
    return f"sagemaker-{region_name}-{account_id}"

def realesrgan_models_names():
    import modules.realesrgan_model
    return [x.name for x in modules.realesrgan_model.get_realesrgan_models(None)]

def list_objects(bucket, prefix='', exts=['.pt', '.pth', '.ckpt', '.safetensors','.yaml']):
    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                _, ext = os.path.splitext(obj['Key'].lstrip('/'))
                if ext in exts:
                    objects.append(obj)
        if 'NextContinuationToken' in page:
            page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix,
                                                ContinuationToken=page['NextContinuationToken'])
    return objects

class ModelsRef:
    def __init__(self):
        self.models_ref = {}

    def get_models_ref_dict(self):
        return self.models_ref
    
    def add_models_ref(self, model_name):
        if model_name in self.models_ref:
            self.models_ref[model_name] += 1
        else:
            self.models_ref[model_name] = 0

    def remove_model_ref(self,model_name):
        if self.models_ref.get(model_name):
            del self.models_ref[model_name]

    def get_models_ref(self, model_name):
        return self.models_ref.get(model_name)
    
    def get_least_ref_model(self):
        sorted_models = sorted(self.models_ref.items(), key=lambda item: item[1])
        if sorted_models:
            least_ref_model, least_counter = sorted_models[0]
            return least_ref_model,least_counter
        else:
            return None,None
    
    def pop_least_ref_model(self):
        sorted_models = sorted(self.models_ref.items(), key=lambda item: item[1])
        if sorted_models:
            least_ref_model, least_counter = sorted_models[0]
            del self.models_ref[least_ref_model]
            return least_ref_model,least_counter
        else:
            return None,None
        
sd_models_Ref = ModelsRef()
cn_models_Ref = ModelsRef()
lora_models_Ref = ModelsRef()
vae_models_Ref = ModelsRef()


def de_register_model(model_name,mode):
    models_Ref = sd_models_Ref
    if mode == 'sd' :
        models_Ref = sd_models_Ref
    elif mode == 'cn':
        models_Ref = cn_models_Ref
    elif mode == 'lora':
        models_Ref = lora_models_Ref
    elif mode == 'vae':
        models_Ref = vae_models_Ref
    models_Ref.remove_model_ref(model_name)
    print (f'---de_register_{mode}_model({model_name})---models_Ref({models_Ref.get_models_ref_dict()})----')
    if 'endpoint_name' in os.environ:
        api_endpoint = os.environ['api_endpoint']
        endpoint_name = os.environ['endpoint_name']
        data = {
            "module":mode,
            "model_name": model_name,
            "endpoint_name": endpoint_name
        }  
        response = requests.delete(url=f'{api_endpoint}/sd/models', json=data)
        # Check if the request was successful
        if response.status_code == requests.codes.ok:
            print(f"{model_name} deleted successfully!")
        else:
            print(f"Error deleting {model_name}: ", response.text)

def get_bucket_and_key(s3uri):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]
    return bucket, key

def s3_download(s3uri, path):
    global cache

    print('---path---', path)
    os.system(f'ls -l {os.path.dirname(path)}')

    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]

    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=key)
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                objects.append(obj)
        if 'NextContinuationToken' in page:
            page_iterator = paginator.paginate(Bucket=bucket, Prefix=key,
                                                ContinuationToken=page['NextContinuationToken'])

    if os.path.isfile('cache'):
        cache = json.load(open('cache', 'r'))

    for obj in objects:
        if obj['Size'] == 0:
            continue
        response = s3_client.head_object(
            Bucket = bucket,
            Key =  obj['Key']
        )
        obj_key = 's3://{0}/{1}'.format(bucket, obj['Key'])
        if obj_key not in cache or cache[obj_key] != response['ETag']:
            filename = obj['Key'][obj['Key'].rfind('/') + 1 : ]

            s3_client.download_file(bucket, obj['Key'], os.path.join(path, filename))
            cache[obj_key] = response['ETag']

    json.dump(cache, open('cache', 'w'))

def http_download(httpuri, path):
    with requests.get(httpuri, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def upload_s3files(s3uri, file_path_with_pattern):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]

    try:
        for file_path in glob.glob(file_path_with_pattern):
            file_name = os.path.basename(file_path)
            __s3file = f'{key}{file_name}'
            print(file_path, __s3file)
            s3_client.upload_file(file_path, bucket, __s3file)
    except ClientError as e:
        print(e)
        return False
    return True

def upload_s3folder(s3uri, file_path):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]

    try:
        for path, _, files in os.walk(file_path):
            for file in files:
                dest_path = path.replace(file_path,"")
                __s3file = f'{key}{dest_path}/{file}'
                __local_file = os.path.join(path, file)
                print(__local_file, __s3file)
                s3_client.upload_file(__local_file, bucket, __s3file)
    except Exception as e:
        print(e)

s3_resource = boto3.resource('s3')
s3_image_path_prefix = 'stable-diffusion-webui/generated/'

def download_images_for_ui(bucket_name):
    bucket_name = get_default_bucket()
    # Create an empty file if not exist
    cache_dir = opts.outdir_txt2img_samples.split('/')[0]
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_name = os.path.join(cache_dir,'cache_image_files_index.json')
    if os.path.isfile(cache_file_name) == False:
        with open(cache_file_name, "w") as f:
            cache_files = {}
            json.dump(cache_files, f)

    bucket = s3_resource.Bucket(bucket_name)
    caches = {}
    with open(cache_file_name, "r") as f:
        _cache = json.load(f)  
        caches = _cache.copy()

    for obj in bucket.objects.all():
        if obj.key.startswith(s3_image_path_prefix):
            new_obj_key = obj.key.replace(s3_image_path_prefix,'').split('/')
            etag = obj.e_tag.replace('"','')
            if caches.get(etag): 
                continue
            # print(f'download:{new_obj_key}')
            if len(new_obj_key) >=3 : ## {username}/{task}/{image}
                task_name = new_obj_key[1]
                if task_name == 'text-to-image':
                    dir_name = os.path.join(opts.outdir_txt2img_samples,new_obj_key[0]) 
                elif task_name == 'image-to-image':
                    dir_name = os.path.join(opts.outdir_img2img_samples,new_obj_key[0])             
                elif task_name == 'extras-single-image':
                    dir_name = os.path.join(opts.outdir_extras_samples,new_obj_key[0])            
                elif task_name == 'extras-batch-images':
                    dir_name = os.path.join(opts.outdir_extras_samples,new_obj_key[0])  
                elif task_name == 'favorites':
                    dir_name = os.path.join(opts.outdir_save,new_obj_key[0])  
                else:
                    dir_name = os.path.join(opts.outdir_txt2img_samples,new_obj_key[0]) 
                os.makedirs(dir_name, exist_ok=True)
                bucket.download_file(obj.key, os.path.join(dir_name,new_obj_key[2]))
            elif len(new_obj_key) ==2:  ## {username}/{image} default save to txt_img
                dir_name = os.path.join(opts.outdir_txt2img_samples,new_obj_key[0]) 
                os.makedirs(dir_name, exist_ok=True)
                bucket.download_file(obj.key,os.path.join(dir_name,new_obj_key[1]))
            else: ## {image} default save to txt_img
                dir_name = os.path.join(opts.outdir_txt2img_samples) 
                os.makedirs(dir_name, exist_ok=True)
                bucket.download_file(obj.key,os.path.join(dir_name,new_obj_key[0]))
            caches[etag] = 1

    with open(cache_file_name, "w") as f:
                json.dump(caches, f) 

def upload_images_to_s3(imgs,request : gr.Request):
    username = get_webui_username(request)
    timestamp = datetime.now(timezone(timedelta(hours=+8))).strftime('%Y-%m-%dT%H:%M:%S')
    bucket_name = opts.train_files_s3bucket.replace('s3://','')
    if bucket_name.endswith('/'):
        bucket_name= bucket_name[:-1]
    if bucket_name == '':
        return 'Error, please configure a S3 bucket at settings page first'
    folder_name = f"train-images/{username}/{timestamp}"
    try:
        for i, img in enumerate(imgs):
            filename = img.name.split('/')[-1]
            object_name = f"{folder_name}/{filename}"
            s3_client.upload_file(img.name, bucket_name.replace('s3://',''), object_name)
    except ClientError as e:
        print(e)
        return e

    return f"{len(imgs)} images uploaded to S3 folder:s3://{bucket_name}/{folder_name}"
