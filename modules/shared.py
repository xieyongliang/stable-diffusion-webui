import argparse
import datetime
import json
import os
import sys
import time
import threading
import gradio as gr
import tqdm
import glob

import modules.artists
import modules.interrogate
import modules.memmon
import modules.styles
import modules.devices as devices
from modules import localization, sd_vae, extensions, script_loading
from modules.paths import models_path, script_path, sd_path
import requests
import boto3
from botocore.exceptions import ClientError

demo = None
#Add by River
models_s3_bucket = None
s3_folder_sd = None
s3_folder_cn = None
s3_folder_lora = None
s3_folder_vae = None
syncLock = threading.Lock()
sync_images_lock = threading.Lock()
tmp_models_dir = '/tmp/models'
tmp_cache_dir = '/tmp/model_sync_cache'
#end 

sd_model_file = os.path.join(script_path, 'model.ckpt')
default_sd_model_file = sd_model_file
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.join(script_path, "v1-inference.yaml"), help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default=sd_model_file, help="path to checkpoint of stable diffusion model; if specified, this checkpoint will be added to the list of checkpoints and loaded",)
parser.add_argument("--ckpt-dir", type=str, default=None, help="Path to directory with stable diffusion checkpoints")
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN'))
parser.add_argument("--gfpgan-model", type=str, help="GFPGAN model file name", default=None)
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats")
parser.add_argument("--no-half-vae", action='store_true', help="do not switch the VAE model to 16-bit floats")
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware acceleration in browser)")
parser.add_argument("--max-batch-count", type=int, default=16, help="maximum batch count value for the UI")
parser.add_argument("--embeddings-dir", type=str, default=os.path.join(script_path, 'embeddings'), help="embeddings directory for textual inversion (default: embeddings)")
parser.add_argument("--hypernetwork-dir", type=str, default=os.path.join(models_path, 'hypernetworks'), help="hypernetwork directory")
parser.add_argument("--localizations-dir", type=str, default=os.path.join(script_path, 'localizations'), help="localizations directory")
parser.add_argument("--allow-code", action='store_true', help="allow custom script execution from webui")
parser.add_argument("--medvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a little speed for low VRM usage")
parser.add_argument("--lowvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage")
parser.add_argument("--lowram", action='store_true', help="load stable diffusion checkpoint weights to VRAM instead of RAM")
parser.add_argument("--always-batch-cond-uncond", action='store_true', help="disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram")
parser.add_argument("--unload-gfpgan", action='store_true', help="does not do anything.")
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
parser.add_argument("--ngrok", type=str, help="ngrok authtoken, alternative to gradio --share", default=None)
parser.add_argument("--ngrok-region", type=str, help="The region in which ngrok should start.", default="us")
parser.add_argument("--enable-insecure-extension-access", action='store_true', help="enable extensions tab regardless of other options")
parser.add_argument("--codeformer-models-path", type=str, help="Path to directory with codeformer model file(s).", default=os.path.join(models_path, 'Codeformer'))
parser.add_argument("--gfpgan-models-path", type=str, help="Path to directory with GFPGAN model file(s).", default=os.path.join(models_path, 'GFPGAN'))
parser.add_argument("--esrgan-models-path", type=str, help="Path to directory with ESRGAN model file(s).", default=os.path.join(models_path, 'ESRGAN'))
parser.add_argument("--bsrgan-models-path", type=str, help="Path to directory with BSRGAN model file(s).", default=os.path.join(models_path, 'BSRGAN'))
parser.add_argument("--realesrgan-models-path", type=str, help="Path to directory with RealESRGAN model file(s).", default=os.path.join(models_path, 'RealESRGAN'))
parser.add_argument("--clip-models-path", type=str, help="Path to directory with CLIP model file(s).", default=None)
parser.add_argument("--xformers", action='store_true', help="enable xformers for cross attention layers")
parser.add_argument("--force-enable-xformers", action='store_true', help="enable xformers for cross attention layers regardless of whether the checking code thinks you can run it; do not make bug reports if this fails to work")
parser.add_argument("--deepdanbooru", action='store_true', help="does not do anything")
parser.add_argument("--opt-split-attention", action='store_true', help="force-enables Doggettx's cross-attention layer optimization. By default, it's on for torch cuda.")
parser.add_argument("--opt-split-attention-invokeai", action='store_true', help="force-enables InvokeAI's cross-attention layer optimization. By default, it's on when cuda is unavailable.")
parser.add_argument("--opt-split-attention-v1", action='store_true', help="enable older version of split attention optimization that does not consume all the VRAM it can find")
parser.add_argument("--disable-opt-split-attention", action='store_true', help="force-disables cross-attention layer optimization")
parser.add_argument("--use-cpu", nargs='+', help="use CPU as torch device for specified modules", default=[], type=str.lower)
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
parser.add_argument("--port", type=int, help="launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available", default=None)
parser.add_argument("--show-negative-prompt", action='store_true', help="does not do anything", default=False)
parser.add_argument("--ui-config-file", type=str, help="filename to use for ui configuration", default=os.path.join(script_path, 'ui-config.json'))
parser.add_argument("--hide-ui-dir-config", action='store_true', help="hide directory configuration from webui", default=False)
parser.add_argument("--freeze-settings", action='store_true', help="disable editing settings", default=False)
parser.add_argument("--ui-settings-file", type=str, help="filename to use for ui settings", default=os.path.join(script_path, 'config.json'))
parser.add_argument("--gradio-debug",  action='store_true', help="launch gradio with --debug option")
parser.add_argument("--gradio-auth", type=str, help='set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"', default=None)
parser.add_argument("--gradio-img2img-tool", type=str, help='gradio image uploader tool: can be either editor for ctopping, or color-sketch for drawing', choices=["color-sketch", "editor"], default="editor")
parser.add_argument("--gradio-inpaint-tool", type=str, choices=["sketch", "color-sketch"], default="sketch", help="gradio inpainting editor: can be either sketch to only blur/noise the input, or color-sketch to paint over it")
parser.add_argument("--opt-channelslast", action='store_true', help="change memory type for stable diffusion to channels last")
parser.add_argument("--styles-file", type=str, help="filename to use for styles", default=os.path.join(script_path, 'styles.csv'))
parser.add_argument("--autolaunch", action='store_true', help="open the webui URL in the system's default browser upon launch", default=False)
parser.add_argument("--theme", type=str, help="launches the UI with light or dark theme", default=None)
parser.add_argument("--use-textbox-seed", action='store_true', help="use textbox for seeds in UI (no up/down, but possible to input long seeds)", default=False)
parser.add_argument("--disable-console-progressbars", action='store_true', help="do not output progressbars to console", default=False)
parser.add_argument("--enable-console-prompts", action='store_true', help="print prompts to console when generating with txt2img and img2img", default=False)
parser.add_argument('--vae-path', type=str, help='Path to Variational Autoencoders model', default=None)
parser.add_argument("--disable-safe-unpickle", action='store_true', help="disable checking pytorch models for malicious code", default=False)
parser.add_argument("--api", action='store_true', help="use api=True to launch the API together with the webui (use --nowebui instead for only the API)")
parser.add_argument("--api-auth", type=str, help='Set authentication for API like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"', default=None)
parser.add_argument("--nowebui", action='store_true', help="use api=True to launch the API instead of the webui")
parser.add_argument("--ui-debug-mode", action='store_true', help="Don't load model to quickly launch UI")
parser.add_argument("--device-id", type=str, help="Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc might be needed before)", default=None)
parser.add_argument("--administrator", action='store_true', help="Administrator rights", default=False)
parser.add_argument("--cors-allow-origins", type=str, help="Allowed CORS origin(s) in the form of a comma-separated list (no spaces)", default=None)
parser.add_argument("--cors-allow-origins-regex", type=str, help="Allowed CORS origin(s) in the form of a single regular expression", default=None)
parser.add_argument("--tls-keyfile", type=str, help="Partially enables TLS, requires --tls-certfile to fully function", default=None)
parser.add_argument("--tls-certfile", type=str, help="Partially enables TLS, requires --tls-keyfile to fully function", default=None)
parser.add_argument("--server-name", type=str, help="Sets hostname of server", default=None)
parser.add_argument("--pureui", action='store_true', help="Pure UI without local inference and progress bar", default=False)
parser.add_argument("--train", action='store_true', help="Train only on SageMaker", default=False)
parser.add_argument("--train-task", type=str, help='Train task - embedding or hypernetwork', default='embedding')
parser.add_argument("--train-args", type=str, help='Train args', default='')
parser.add_argument('--embeddings-s3uri', default='', type=str, help='Embedding S3Uri')
parser.add_argument('--hypernetwork-s3uri', default='', type=str, help='Hypernetwork S3Uri')
parser.add_argument('--sd-models-s3uri', default='', type=str, help='SD Models S3Uri')
parser.add_argument('--db-models-s3uri', default='', type=str, help='DB Models S3Uri')
parser.add_argument('--lora-models-s3uri', default='', type=str, help='Lora Models S3Uri')
parser.add_argument('--username', default='', type=str, help='Username')
parser.add_argument('--api-endpoint', default='', type=str, help='API Endpoint')
parser.add_argument('--dreambooth-config-id', default='', type=str, help='Dreambooth config ID')
parser.add_argument('--model-name', default='', type=str, help='Model name')
parser.add_argument('--region-name', default='', type=str, help='Region name')

script_loading.preload_extensions(extensions.extensions_dir, parser)
script_loading.preload_extensions(extensions.extensions_builtin_dir, parser)

cmd_opts = parser.parse_args()

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
}

cmd_opts.disable_extension_access = (cmd_opts.share or cmd_opts.listen or cmd_opts.server_name) and not cmd_opts.enable_insecure_extension_access

devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
    (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])

device = devices.device
weight_load_location = None if cmd_opts.lowram else "cpu"

batch_cond_uncond = cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram and not cmd_opts.medvram
xformers_available = False
config_filename = cmd_opts.ui_settings_file

os.makedirs(cmd_opts.hypernetwork_dir, exist_ok=True)
embeddings = {}
hypernetworks = {}
loaded_hypernetwork = None
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
    job_timestamp = '0'
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0
    textinfo = None
    time_start = None
    need_restart = False

    def skip(self):
        self.skipped = True

    def interrupt(self):
        self.interrupted = True

    def nextjob(self):
        if opts.show_progress_every_n_steps == -1:
            self.do_set_current_image()

        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0

    def dict(self):
        obj = {
            "skipped": self.skipped,
            "interrupted": self.skipped,
            "job": self.job,
            "job_count": self.job_count,
            "job_no": self.job_no,
            "sampling_step": self.sampling_step,
            "sampling_steps": self.sampling_steps,
        }

        return obj

    def begin(self):
        self.sampling_step = 0
        self.job_count = -1
        self.job_no = 0
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.current_latent = None
        self.current_image = None
        self.current_image_sampling_step = 0
        self.skipped = False
        self.interrupted = False
        self.textinfo = None
        self.time_start = time.time()

        devices.torch_gc()

    def end(self):
        self.job = ""
        self.job_count = 0

        devices.torch_gc()

    """sets self.current_image from self.current_latent if enough sampling steps have been made after the last call to this"""
    def set_current_image(self):
        if self.sampling_step - self.current_image_sampling_step >= opts.show_progress_every_n_steps and opts.show_progress_every_n_steps > 0:
            self.do_set_current_image()

    def do_set_current_image(self):
        if not parallel_processing_allowed:
            return
        if self.current_latent is None:
            return

        import modules.sd_samplers
        if opts.show_progress_grid:
            self.current_image = modules.sd_samplers.samples_to_image_grid(self.current_latent)
        else:
            self.current_image = modules.sd_samplers.sample_to_image(self.current_latent)

        self.current_image_sampling_step = self.sampling_step

state = State()

artist_db = modules.artists.ArtistsDatabase(os.path.join(script_path, 'artists.csv'))

styles_filename = cmd_opts.styles_file
prompt_styles = modules.styles.StyleDatabase(styles_filename)

interrogator = modules.interrogate.InterrogateModels("interrogate")

face_restorers = []

def get_default_sagemaker_bucket():
    region_name = boto3.Session().region_name
    account_id = boto3.Session().client('sts').get_caller_identity()['Account']
    return f"s3://sagemaker-{region_name}-{account_id}"

def realesrgan_models_names():
    import modules.realesrgan_model
    return [x.name for x in modules.realesrgan_model.get_realesrgan_models(None)]

#add by River
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
#end by River

class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.refresh = refresh

def options_section(section_identifier, options_dict):
    for k, v in options_dict.items():
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

options_templates.update(options_section(('sd', "Stable Diffusion"), {
    "sagemaker_endpoint": OptionInfo(None, "SaegMaker endpoint", gr.Dropdown, lambda: {"choices": list_sagemaker_endpoints()}, refresh=refresh_sagemaker_endpoints),
    "sd_model_checkpoint": OptionInfo(None, "Stable Diffusion checkpoint", gr.Dropdown, lambda: {"choices": list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "sd_checkpoint_cache": OptionInfo(0, "Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_vae": OptionInfo("auto", "SD VAE", gr.Dropdown, lambda: {"choices": sd_vae.vae_list}, refresh=sd_vae.refresh_vae_list),
    "sd_vae_as_default": OptionInfo(False, "Ignore selected VAE for stable diffusion checkpoints that have their own .vae.pt next to them"),
    "sd_hypernetwork": OptionInfo("None", "Hypernetwork", gr.Dropdown, lambda: {"choices": ["None"] + [x for x in hypernetworks.keys()]}, refresh=reload_hypernetworks),
    "sd_hypernetwork_strength": OptionInfo(1.0, "Hypernetwork strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.001}),
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors."),
    "img2img_fix_steps": OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies (normally you'd do less with less denoising)."),
    "enable_quantization": OptionInfo(False, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds. Requires restart to apply."),
    "enable_emphasis": OptionInfo(True, "Emphasis: use (text) to make model pay more attention to text and [text] to make it pay less attention"),
    "use_old_emphasis_implementation": OptionInfo(False, "Use old emphasis implementation. Can be useful to reproduce old seeds."),
    "enable_batch_seeds": OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image"),
    "comma_padding_backtrack": OptionInfo(20, "Increase coherency by padding from the last comma within n tokens when using more than 75 tokens", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1 }),
    "filter_nsfw": OptionInfo(False, "Filter NSFW content"),
    'CLIP_stop_at_last_layers': OptionInfo(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}),
    "random_artist_categories": OptionInfo([], "Allowed categories for random artists selection when using the Roll button", gr.CheckboxGroup, {"choices": artist_db.categories()}),
}))

options_templates.update(options_section(('saving-images', "Saving images/grids"), {
    "samples_save": OptionInfo(True, "Always save all generated images"),
    "samples_format": OptionInfo('png', 'File format for images'),
    "samples_filename_pattern": OptionInfo("", "Images filename pattern", component_args=hide_dirs),
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
    "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "export_for_4chan": OptionInfo(True, "If PNG image is larger than 4MB or any dimension is larger than 4000, downscale and save copy as JPG"),

    "use_original_name_batch": OptionInfo(False, "Use original name for output filename during batch process in extras tab"),
    "save_selected_only": OptionInfo(True, "When using 'Save' button, only save a single selected image"),
    "do_not_add_watermark": OptionInfo(False, "Do not add watermark to images"),

    "temp_dir":  OptionInfo("", "Directory for temporary images; leave empty for default"),
    "clean_temp_dir_at_start": OptionInfo(False, "Cleanup non-default temporary directory when starting webui"),

}))

options_templates.update(options_section(('saving-paths', "Paths for saving"), {
    "train_files_s3bucket":OptionInfo(get_default_sagemaker_bucket(),"S3 bucket name for uploading/downloading images",component_args=hide_dirs),
    "outdir_samples": OptionInfo("", "Output directory for images; if empty, defaults to three directories below", component_args=hide_dirs),
    "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'Output directory for txt2img images', component_args=hide_dirs),
    "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'Output directory for img2img images', component_args=hide_dirs),
    "outdir_extras_samples": OptionInfo("outputs/extras-images", 'Output directory for images from extras tab', component_args=hide_dirs),
    "outdir_grids": OptionInfo("", "Output directory for grids; if empty, defaults to two directories below", component_args=hide_dirs),
    "outdir_txt2img_grids": OptionInfo("outputs/txt2img-grids", 'Output directory for txt2img grids', component_args=hide_dirs),
    "outdir_img2img_grids": OptionInfo("outputs/img2img-grids", 'Output directory for img2img grids', component_args=hide_dirs),
    "outdir_save": OptionInfo("log/images", "Directory for saving images using the Save button", component_args=hide_dirs),
}))

options_templates.update(options_section(('saving-to-dirs', "Saving to a directory"), {
    "save_to_dirs": OptionInfo(False, "Save images to a subdirectory"),
    "grid_save_to_dirs": OptionInfo(False, "Save grids to a subdirectory"),
    "use_save_to_dirs_for_ui": OptionInfo(False, "When using \"Save\" button, save images to a subdirectory"),
    "directories_filename_pattern": OptionInfo("", "Directory name pattern", component_args=hide_dirs),
    "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1, **hide_dirs}),
}))

options_templates.update(options_section(('upscaling', "Upscaling"), {
    "ESRGAN_tile": OptionInfo(192, "Tile size for ESRGAN upscalers. 0 = no tiling.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for ESRGAN upscalers. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"], "Select which Real-ESRGAN models to show in the web UI. (Requires restart)", gr.CheckboxGroup, lambda: {"choices": realesrgan_models_names()}),
    "upscaler_for_img2img": OptionInfo(None, "Upscaler for img2img", gr.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers]}),
    "use_scale_latent_for_hires_fix": OptionInfo(False, "Upscale latent space image when doing hires. fix"),
}))

options_templates.update(options_section(('face-restoration', "Face restoration"), {
    "face_restoration_model": OptionInfo(None, "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
    "code_former_weight": OptionInfo(0.5, "CodeFormer weight parameter; 0 = maximum effect; 1 = minimum effect", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
}))

options_templates.update(options_section(('system', "System"), {
    "memmon_poll_rate": OptionInfo(8, "VRAM usage polls per second during generation. Set to 0 to disable.", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}),
    "samples_log_stdout": OptionInfo(False, "Always print all generation info to standard output"),
    "multiple_tqdm": OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job."),
}))

options_templates.update(options_section(('training', "Training"), {
    "unload_models_when_training": OptionInfo(False, "Move VAE and CLIP to RAM when training if possible. Saves VRAM."),
    "pin_memory": OptionInfo(False, "Turn on pin_memory for DataLoader. Makes training slightly faster but can increase memory usage."),
    "save_optimizer_state": OptionInfo(False, "Saves Optimizer state as separate *.optim file. Training can be resumed with HN itself and matching optim file."),
    "dataset_filename_word_regex": OptionInfo("", "Filename word regex"),
    "dataset_filename_join_string": OptionInfo(" ", "Filename join string"),
    "training_image_repeats_per_epoch": OptionInfo(1, "Number of repeats for a single input image per epoch; used only for displaying epoch number", gr.Number, {"precision": 0}),
    "training_write_csv_every": OptionInfo(500, "Save an csv containing the loss to log directory every N steps, 0 to disable"),
    "training_xattention_optimizations": OptionInfo(False, "Use cross attention optimizations while training"),
}))

options_templates.update(options_section(('interrogate', "Interrogate Options"), {
    "interrogate_keep_models_in_memory": OptionInfo(False, "Interrogate: keep models in VRAM"),
    "interrogate_use_builtin_artists": OptionInfo(True, "Interrogate: use artists from artists.csv"),
    "interrogate_return_ranks": OptionInfo(False, "Interrogate: include ranks of model tags matches in results (Has no effect on caption-based interrogators)."),
    "interrogate_clip_num_beams": OptionInfo(1, "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length": OptionInfo(24, "Interrogate: minimum description length (excluding artists, etc..)", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length": OptionInfo(48, "Interrogate: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_dict_limit": OptionInfo(1500, "CLIP: maximum number of lines in text file (0 = No limit)"),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.5, "Interrogate: deepbooru score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(True, "Interrogate: deepbooru sort alphabetically"),
    "deepbooru_use_spaces": OptionInfo(False, "use spaces for tags in deepbooru"),
    "deepbooru_escape": OptionInfo(True, "escape (\\) brackets in deepbooru (so they are used as literal brackets and not for emphasis)"),
}))

options_templates.update(options_section(('extra_networks', "Extra Networks"), {
    "extra_networks_default_view": OptionInfo("cards", "Default view for Extra Networks", gr.Dropdown, {"choices": ["cards", "thumbs"]}),
    "extra_networks_default_multiplier": OptionInfo(1.0, "Multiplier for extra networks", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "extra_networks_card_width": OptionInfo(0, "Card width for Extra Networks (px)"),
    "extra_networks_card_height": OptionInfo(0, "Card height for Extra Networks (px)"),
    "extra_networks_add_text_separator": OptionInfo(" ", "Extra text to add before <...> when adding extra network to prompt"),
    "sd_hypernetwork": OptionInfo("None", "Add hypernetwork to prompt", gr.Dropdown, lambda: {"choices": [""] + [x for x in hypernetworks.keys()]}, refresh=reload_hypernetworks),
}))

options_templates.update(options_section(('ui', "User interface"), {
    "show_progressbar": OptionInfo(True, "Show progressbar"),
    "show_progress_every_n_steps": OptionInfo(0, "Show image creation progress every N sampling steps. Set to 0 to disable. Set to -1 to show after completion of batch.", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}),
    "show_progress_grid": OptionInfo(True, "Show previews of all images generated in a batch as a grid"),
    "return_grid": OptionInfo(True, "Show grid in results for web"),
    "do_not_show_images": OptionInfo(False, "Do not show any images in results for web"),
    "add_model_hash_to_info": OptionInfo(True, "Add model hash to generation information"),
    "add_model_name_to_info": OptionInfo(False, "Add model name to generation information"),
    "disable_weights_auto_swap": OptionInfo(False, "When reading generation parameters from text into UI (from PNG info or pasted text), do not change the selected model/checkpoint."),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    "font": OptionInfo("", "Font for image grids that have text"),
    "js_modal_lightbox": OptionInfo(True, "Enable full page image viewer"),
    "js_modal_lightbox_initially_zoomed": OptionInfo(True, "Show images zoomed in by default in full page image viewer"),
    "show_progress_in_title": OptionInfo(True, "Show generation progress in window title."),
    'quicksettings': OptionInfo("", "Quicksettings list"),
    'localization': OptionInfo("None", "Localization (requires restart)", gr.Dropdown, lambda: {"choices": ["None"] + list(localization.localizations.keys())}, refresh=lambda: localization.list_localizations(cmd_opts.localizations_dir)),
    "ui_extra_networks_tab_reorder": OptionInfo("", "Extra networks tab order"),
}))

options_templates.update(options_section(('sampler-params', "Sampler parameters"), {
    "hide_samplers": OptionInfo([], "Hide samplers in user interface (requires restart)", gr.CheckboxGroup, lambda: {"choices": [x.name for x in list_samplers()]}),
    "eta_ddim": OptionInfo(0.0, "eta (noise multiplier) for DDIM", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "eta_ancestral": OptionInfo(1.0, "eta (noise multiplier) for ancestral samplers", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "ddim_discretize": OptionInfo('uniform', "img2img DDIM discretize", gr.Radio, {"choices": ['uniform', 'quad']}),
    's_churn': OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_tmin':  OptionInfo(0.0, "sigma tmin",  gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_noise': OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    'eta_noise_seed_delta': OptionInfo(0, "Eta noise seed delta", gr.Number, {"precision": 0}),
}))

options_templates.update(options_section((None, "Hidden options"), {
    "disabled_extensions": OptionInfo([], "Disable those extensions"),
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
            self.data_labels[key].onchange()

        return True

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
        d = {k: self.data.get(k, self.data_labels.get(k).default) for k in self.data_labels.keys()}
        return json.dumps(d)

    def add_option(self, key, info):
        self.data_labels[key] = info

    def reorder(self):
        """reorder settings so that all items related to section always go together"""

        section_ids = {}
        settings_items = self.data_labels.items()
        for k, item in settings_items:
            if item.section not in section_ids:
                section_ids[item.section] = len(section_ids)

        self.data_labels = {k: v for k, v in sorted(settings_items, key=lambda x: section_ids[x[1].section])}


opts = Options()
if os.path.exists(config_filename):
    opts.load(config_filename)

if cmd_opts.pureui and opts.localization == "None":
    opts.localization = "zh_CN"


sd_upscalers = []

sd_model = None

clip_model = None

progress_print_out = sys.stdout


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
        self._tqdm.total=new_total

    def clear(self):
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None


total_tqdm = TotalTQDM()

mem_mon = modules.memmon.MemUsageMonitor("MemMon", device, opts)
mem_mon.start()


def listfiles(dirname):
    filenames = [os.path.join(dirname, x) for x in sorted(os.listdir(dirname)) if not x.startswith(".")]
    return [file for file in filenames if os.path.isfile(file)]

def html_path(filename):
    return os.path.join(script_path, "html", filename)


def html(filename):
    path = html_path(filename)

    if os.path.exists(path):
        with open(path, encoding="utf8") as file:
            return file.read()

    return ""

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
    bucket_name = get_default_sagemaker_bucket().replace('s3://','')
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
