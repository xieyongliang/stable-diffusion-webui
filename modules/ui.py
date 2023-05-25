import html
import json
import math
import mimetypes
import os
import platform
import random
import subprocess as sp
import sys
import tempfile
import time
import traceback
from functools import partial, reduce
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timedelta, timezone
import gradio as gr
import gradio.routes
import gradio.utils
import numpy as np
from PIL import Image, PngImagePlugin
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call

from modules import sd_hijack, sd_models, localization, script_callbacks, ui_extensions, deepbooru
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML
from modules.paths import script_path

from modules.shared import opts, cmd_opts, restricted_opts,get_default_sagemaker_bucket
import modules.codeformer_model
import modules.generation_parameters_copypaste as parameters_copypaste
import modules.gfpgan_model
import modules.hypernetworks.ui
import modules.scripts
import modules.shared as shared
import modules.styles
import modules.textual_inversion.ui
import modules.model_merger
from modules import prompt_parser
from modules.images import save_image
from modules.sd_hijack import model_hijack
from modules.sd_samplers import samplers, samplers_for_img2img
import modules.textual_inversion.ui
import modules.hypernetworks.ui
from modules.generation_parameters_copypaste import image_from_url_text
from modules.sd_models import get_sd_model_checkpoint_from_title
import requests

region_name = boto3.session.Session().region_name
s3_client = boto3.client('s3', region_name=region_name)
endpointUrl = s3_client.meta.endpoint_url
s3_client = boto3.client('s3', endpoint_url=endpointUrl, region_name=region_name)

training_instance_types = [
    'ml.p2.xlarge',
    'ml.p2.8xlarge',
    'ml.p2.16xlarge',
    'ml.p3.2xlarge',
    'ml.p3.8xlarge',
    'ml.p3.16xlarge',
    'ml.g4dn.xlarge',
    'ml.g4dn.2xlarge',
    'ml.g4dn.4xlarge',
    'ml.g4dn.8xlarge',
    'ml.g4dn.12xlarge',
    'ml.g4dn.16xlarge',
    'ml.g5.xlarge',
    'ml.g5.2xlarge',
    'ml.g5.4xlarge',
    'ml.g5.8xlarge',
    'ml.g5.12xlarge',
    'ml.g5.16xlarge',
    'ml.g5.24xlarge',
    'ml.g5.48xlarge',
    'ml.p4d.24xlarge'
]
component_dict = {}

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the browser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'

if cmd_opts.ngrok != None:
    import modules.ngrok as ngrok
    print('ngrok authtoken detected, trying to connect...')
    ngrok.connect(cmd_opts.ngrok, cmd_opts.port if cmd_opts.port != None else 7860, cmd_opts.ngrok_region)


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

## Begin output images uploaded to s3 by River
s3_resource = boto3.resource('s3')

def save_images_to_s3(full_fillnames,timestamp,username):
    sagemaker_endpoint = shared.opts.sagemaker_endpoint
    bucket_name = opts.train_files_s3bucket.replace('s3://','')
    if bucket_name.endswith('/'):
        bucket_name= bucket_name[:-1]
    if bucket_name == '':
        return 'Error, please configure a S3 bucket at settings page first'
    folder_name = f"output-images/{username}/{sagemaker_endpoint}/{timestamp}"
    try:
        for i, fname in enumerate(full_fillnames):
            filename = fname.split('/')[-1]
            object_name = f"{folder_name}/{filename}"
            s3_client.upload_file(fname, bucket_name, object_name)
            print (f'upload file [{i}]:{filename} to s3://{bucket_name}/{object_name}')
    except ClientError as e:
        print(e)
        return e
    return f"s3://{bucket_name}/{folder_name}"
## End output images uploaded to s3 by River


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.wrap .z-20 svg { display:none!important; }
.wrap .z-20::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
.meta-text-center { display:none!important; }
"""

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
art_symbol = '\U0001f3a8'  # 🎨
paste_symbol = '\u2199\ufe0f'  # ↙
folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
extra_networks_symbol = '\U0001F3B4'  # 🎴

def text_to_hyperlink_html(url):
    text= f'<p><a target="_blank" href="{url}">{url}</a></p>'
    return text

def plaintext_to_html(text):
    text = "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split('\n')]) + "</p>"
    return text

def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return image_from_url_text(x[0])

def save_files(username,js_data, images, do_make_zip, index):
    import csv
    filenames = []
    fullfns = []

    #quick dictionary to class object conversion. Its necessary due apply_filename_pattern requiring it
    class MyObject:
        def __init__(self, d=None):
            if d is not None:
                for key, value in d.items():
                    setattr(self, key, value)

    data = json.loads(js_data)

    p = MyObject(data)
    path = opts.outdir_save +'/'+username
    save_to_dirs = opts.use_save_to_dirs_for_ui
    extension: str = opts.samples_format
    start_index = 0

    if index > -1 and opts.save_selected_only and (index >= data["index_of_first_image"]):  # ensures we are looking at a specific non-grid picture, and we have save_selected_only

        images = [images[index]]
        start_index = index

    os.makedirs(opts.outdir_save+'/'+username, exist_ok=True)

    with open(os.path.join(opts.outdir_save+'/'+username, "log.csv"), "w", encoding="utf8", newline='') as file:
        at_start = file.tell() == 0
        writer = csv.writer(file)
        if at_start:
            writer.writerow(["prompt", "seed", "width", "height", "sampler", "cfgs", "steps", "filename", "negative_prompt"])

        for image_index, filedata in enumerate(images, start_index):
            image = image_from_url_text(filedata)

            is_grid = image_index < p.index_of_first_image
            i = 0 if is_grid else (image_index - p.index_of_first_image)

            if image_index >= len(p.infotexts) or i >= len(p.all_seeds) or i >= len(p.all_prompts):
                break

            fullfn, txt_fullfn = save_image(image, path, "", seed=p.all_seeds[i], prompt=p.all_prompts[i], extension=extension, info=p.infotexts[image_index], grid=is_grid, p=p, save_to_dirs=save_to_dirs)
            filename = os.path.relpath(fullfn, path)
            print(f'fullfn:{fullfn},\n txt_fullfn:{txt_fullfn} \nfilename:{filename}')
            filenames.append(filename)
            fullfns.append(fullfn)
            if txt_fullfn:
                filenames.append(os.path.basename(txt_fullfn))
                fullfns.append(txt_fullfn)
        writer.writerow([data["prompt"], data["seed"], data["width"], data["height"], data["sampler_name"], data["cfg_scale"], data["steps"], filenames[0], data["negative_prompt"]])
    
    timestamp = datetime.now(timezone(timedelta(hours=+8))).strftime('%Y-%m-%dT%H:%M:%S')
    logfile = os.path.join(opts.outdir_save+'/'+username, "log.csv")
    s3folder = save_images_to_s3(fullfns+[logfile],timestamp,username)
    # Make Zip
    if do_make_zip:
        zip_filepath = os.path.join(path, "images.zip")

        from zipfile import ZipFile
        with ZipFile(zip_filepath, "w") as zip_file:
            for i in range(len(fullfns)):
                with open(fullfns[i], mode="rb") as f:
                    zip_file.writestr(filenames[i], f.read())
        fullfns.insert(0, zip_filepath)

    return gr.File.update(value=fullfns, visible=True), '', '', plaintext_to_html(f"Saved: {filenames[0]}"),text_to_hyperlink_html(s3folder)




def calc_time_left(progress, threshold, label, force_display):
    if progress == 0:
        return ""
    else:
        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start
        if (eta_relative > threshold and progress > 0.02) or force_display:
            if eta_relative > 3600:
                return label + time.strftime('%H:%M:%S', time.gmtime(eta_relative))
            elif eta_relative > 60:
                return label + time.strftime('%M:%S',  time.gmtime(eta_relative))
            else:
                return label + time.strftime('%Ss',  time.gmtime(eta_relative))
        else:
            return ""


def check_progress_call(id_part):
    if shared.state.job_count == 0:
        return "", gr_show(False), gr_show(False), gr_show(False)

    progress = 0

    if shared.state.job_count > 0:
        progress += shared.state.job_no / shared.state.job_count
    if shared.state.sampling_steps > 0:
        progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

    time_left = calc_time_left( progress, 1, " ETA: ", shared.state.time_left_force_display )
    if time_left != "":
        shared.state.time_left_force_display = True

    progress = min(progress, 1)

    progressbar = ""
    if opts.show_progressbar:
        progressbar = f"""<div class='progressDiv'><div class='progress' style="overflow:visible;width:{progress * 100}%;white-space:nowrap;">{"&nbsp;" * 2 + str(int(progress*100))+"%" + time_left if progress > 0.01 else ""}</div></div>"""

    image = gr_show(False)
    preview_visibility = gr_show(False)

    if opts.show_progress_every_n_steps != 0:
        shared.state.set_current_image()
        image = shared.state.current_image

        if image is None:
            image = gr.update(value=None)
        else:
            preview_visibility = gr_show(True)

    if shared.state.textinfo is not None:
        textinfo_result = gr.HTML.update(value=shared.state.textinfo, visible=True)
    else:
        textinfo_result = gr_show(False)

    if progress == 1:
        return "", preview_visibility, image, textinfo_result
    else:
        return f"<span id='{id_part}_progress_span' style='display: none'>{time.time()}</span><p>{progressbar}</p>", preview_visibility, image, textinfo_result


def check_progress_call_initial(id_part):
    shared.state.job_count = -1
    shared.state.current_latent = None
    shared.state.current_image = None
    shared.state.textinfo = None
    shared.state.time_start = time.time()
    shared.state.time_left_force_display = False

    return check_progress_call(id_part)


def roll_artist(prompt):
    allowed_cats = set([x for x in shared.artist_db.categories() if len(opts.random_artist_categories)==0 or x in opts.random_artist_categories])
    artist = random.choice([x for x in shared.artist_db.artists if x.category in allowed_cats])

    return prompt + ", " + artist.name if prompt != '' else artist.name


def visit(x, func, path=""):
    if hasattr(x, 'children'):
        for c in x.children:
            visit(c, func, path)
    elif x.label is not None:
        func(path + "/" + str(x.label), x)


def add_style(name: str, prompt: str, negative_prompt: str):
    if name is None:
        return [gr_show() for x in range(4)]

    style = modules.styles.PromptStyle(name, prompt, negative_prompt)
    shared.prompt_styles.styles[style.name] = style
    # Save all loaded prompt styles: this allows us to update the storage format in the future more easily, because we
    # reserialize all styles every time we save them
    shared.prompt_styles.save_styles(shared.styles_filename)

    return [gr.Dropdown.update(visible=True, choices=list(shared.prompt_styles.styles)) for _ in range(4)]


def apply_styles(prompt, prompt_neg, style1_name, style2_name):
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, [style1_name, style2_name])
    prompt_neg = shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, [style1_name, style2_name])

    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value="None"), gr.Dropdown.update(value="None")]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image)

    return gr_show(True) if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = deepbooru.model.tag(image)
    return gr_show(True) if prompt is None else prompt


def create_seed_inputs():
    with gr.Row():
        with gr.Box():
            with gr.Row(elem_id='seed_row'):
                seed = (gr.Textbox if cmd_opts.use_textbox_seed else gr.Number)(label='Seed', value=-1)
                seed.style(container=False)
                random_seed = gr.Button(random_symbol, elem_id='random_seed')
                reuse_seed = gr.Button(reuse_symbol, elem_id='reuse_seed')

        with gr.Box(elem_id='subseed_show_box'):
            seed_checkbox = gr.Checkbox(label='Extra', elem_id='subseed_show', value=False)

    # Components to show/hide based on the 'Extra' checkbox
    seed_extras = []

    with gr.Row(visible=False) as seed_extra_row_1:
        seed_extras.append(seed_extra_row_1)
        with gr.Box():
            with gr.Row(elem_id='subseed_row'):
                subseed = gr.Number(label='Variation seed', value=-1)
                subseed.style(container=False)
                random_subseed = gr.Button(random_symbol, elem_id='random_subseed')
                reuse_subseed = gr.Button(reuse_symbol, elem_id='reuse_subseed')
        subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01)

    with gr.Row(visible=False) as seed_extra_row_2:
        seed_extras.append(seed_extra_row_2)
        seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from width", value=0)
        seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from height", value=0)

    random_seed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[seed])
    random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])

    def change_visibility(show):
        return {comp: gr_show(show) for comp in seed_extras}

    seed_checkbox.change(change_visibility, show_progress=False, inputs=[seed_checkbox], outputs=seed_extras)

    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox


def connect_reuse_seed(seed: gr.Number, reuse_seed: gr.Button, generation_info: gr.Textbox, dummy_component, is_subseed):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""
    def copy_seed(gen_info_string: str, index):
        res = -1

        try:
            gen_info = json.loads(gen_info_string)
            index -= gen_info.get('index_of_first_image', 0)

            if is_subseed and gen_info.get('subseed_strength', 0) > 0:
                all_subseeds = gen_info.get('all_subseeds', [-1])
                res = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
            else:
                all_seeds = gen_info.get('all_seeds', [-1])
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]

        except json.decoder.JSONDecodeError as e:
            if gen_info_string != '':
                print("Error parsing JSON generation info:", file=sys.stderr)
                print(gen_info_string, file=sys.stderr)

        return [res, gr_show(False)]

    reuse_seed.click(
        fn=copy_seed,
        _js="(x, y) => [x, selected_gallery_index()]",
        show_progress=False,
        inputs=[generation_info, dummy_component],
        outputs=[seed, dummy_component]
    )


def update_token_counter(text, steps):
    if not cmd_opts.pureui:
        try:
            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
            prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

        except Exception:
            # a parsing error can happen here during typing, and we don't want to bother the user with
            # messages related to it in console
            prompt_schedules = [[[steps, text]]]

        flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
        prompts = [prompt_text for step, prompt_text in flat_prompts]
        tokens, token_count, max_length = max([model_hijack.tokenize(prompt) for prompt in prompts], key=lambda args: args[1])
        style_class = ' class="red"' if (token_count > max_length) else ""
        return f"<span {style_class}>{token_count}/{max_length}</span>"
    else:
        return f"<span>N/A</span>"

def create_toprow(is_img2img):
    id_part = "img2img" if is_img2img else "txt2img"

    with gr.Row(elem_id="toprow"):
        with gr.Column(scale=6):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=2,
                            placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)"
                        )

            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt", show_label=False, lines=2,
                            placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)"
                        )

        with gr.Column(scale=1, elem_id="roll_col"):
            roll = ToolButton(value=art_symbol, elem_id="roll", visible=len(shared.artist_db.artists) > 0)
            paste = ToolButton(value=paste_symbol, elem_id="paste")
            save_style = ToolButton(value=save_style_symbol, elem_id="style_create", visible=False)
            prompt_style_apply = ToolButton(value=apply_style_symbol, elem_id="style_apply", visible=False)
            extra_networks_button = ToolButton(value=extra_networks_symbol, elem_id=f"{id_part}_extra_networks")

            token_counter = gr.HTML(value="<span></span>", elem_id=f"{id_part}_token_counter")
            token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")

        button_interrogate = None
        button_deepbooru = None
        if is_img2img:
            with gr.Column(scale=1, elem_id="interrogate_col"):
                button_interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru", visible=False)

        with gr.Column(scale=1):
            with gr.Row():
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip")
                interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt")
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                skip.click(
                    fn=lambda: shared.state.skip(),
                    inputs=[],
                    outputs=[],
                )

                interrupt.click(
                    fn=lambda: shared.state.interrupt(),
                    inputs=[],
                    outputs=[],
                )

            with gr.Row():
                with gr.Column(scale=1, elem_id="style_pos_col"):
                    prompt_style = gr.Dropdown(label="Style 1", elem_id=f"{id_part}_style_index", choices=[k for k, v in shared.prompt_styles.styles.items()], value=next(iter(shared.prompt_styles.styles.keys())))
                    prompt_style.save_to_config = True
                    prompt_style.style(container=False)

                with gr.Column(scale=1, elem_id="style_neg_col"):
                    prompt_style2 = gr.Dropdown(label="Style 2", elem_id=f"{id_part}_style2_index", choices=[k for k, v in shared.prompt_styles.styles.items()], value=next(iter(shared.prompt_styles.styles.keys())))
                    prompt_style2.save_to_config = True
                    prompt_style2.style(container=False)

    return prompt, roll, prompt_style, negative_prompt, prompt_style2, submit, button_interrogate, button_deepbooru, prompt_style_apply, save_style, paste, extra_networks_button, token_counter, token_button


def setup_progressbar(progressbar, preview, id_part, textinfo=None):
    if textinfo is None:
        textinfo = gr.HTML(visible=False)

    check_progress = gr.Button('Check progress', elem_id=f"{id_part}_check_progress", visible=False)
    check_progress.click(
        fn=lambda: check_progress_call(id_part),
        show_progress=False,
        inputs=[],
        outputs=[progressbar, preview, preview, textinfo],
    )

    check_progress_initial = gr.Button('Check progress (first)', elem_id=f"{id_part}_check_progress_initial", visible=False)
    check_progress_initial.click(
        fn=lambda: check_progress_call_initial(id_part),
        show_progress=False,
        inputs=[],
        outputs=[progressbar, preview, preview, textinfo],
    )


def apply_setting(key, value):
    if value is None:
        return gr.update()

    if shared.cmd_opts.freeze_settings:
        return gr.update()

    # dont allow model to be swapped when model hash exists in prompt
    if key == "sd_model_checkpoint" and opts.disable_weights_auto_swap:
        return gr.update()

    if key == "sd_model_checkpoint":
        return gr.update()

    comp_args = opts.data_labels[key].component_args
    if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
        return

    valtype = type(opts.data_labels[key].default)
    oldval = opts.data[key]
    opts.data[key] = valtype(value) if valtype != type(None) else value
    if oldval != value and opts.data_labels[key].onchange is not None:
        opts.data_labels[key].onchange()

    if not cmd_opts.pureui:
        opts.save(shared.config_filename)

    return value


def update_generation_info(generation_info, html_info, img_index):
    try:
        generation_info = json.loads(generation_info)
        if img_index < 0 or img_index >= len(generation_info["infotexts"]):
            return html_info
        return plaintext_to_html(generation_info["infotexts"][img_index])
    except Exception:
        pass
    # if the json parse or anything else fails, just return the old html_info
    return html_info

def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    def refresh_sagemaker_endpoints(request : gr.Request):
        username = shared.get_webui_username(request)
        refresh_method(username)
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    def refresh_sd_models(request: gr.Request):
        username = shared.get_webui_username(request)
        refresh_method(username)
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    def refresh_lora_models(sagemaker_endpoint,request:gr.Request):
        username = shared.get_webui_username(request)
        refresh_method(sagemaker_endpoint,username)
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    def refresh_checkpoints(sagemaker_endpoint,request:gr.Request):
        username = shared.get_webui_username(request)
        refresh_method(sagemaker_endpoint,username)
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = gr.Button(value=refresh_symbol, elem_id=elem_id)
    if elem_id == 'refresh_sagemaker_endpoint':
        refresh_button.click(
            fn=refresh_sagemaker_endpoints,
            inputs=[],
            outputs=[refresh_component]
        )
    elif elem_id == 'refresh_sd_models':
        refresh_button.click(
            fn=refresh_sd_models,
            inputs=[],
            outputs=[refresh_component]
        )
    elif elem_id == 'refresh_sd_model_checkpoint':
        refresh_button.click(
            fn=refresh_checkpoints,
            inputs=[shared.sagemaker_endpoint_component],
            outputs=[refresh_component]
        )
    elif elem_id == 'refresh_sd_lora':
        refresh_button.click(
            fn=refresh_lora_models,
            inputs=[shared.sagemaker_endpoint_component],
            outputs=[refresh_component]
        )
    else:
        refresh_button.click(
            fn=refresh,
            inputs=[],
            outputs=[refresh_component]
        )
    return refresh_button


def create_output_panel(tabname, outdir):
    def open_folder(f):
        if not os.path.exists(f):
            print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
            return
        elif not os.path.isdir(f):
            print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
            return

        if not shared.cmd_opts.hide_ui_dir_config:
            path = os.path.normpath(f)
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                sp.Popen(["open", path])
            else:
                sp.Popen(["xdg-open", path])

    with gr.Column(variant='panel'):
            with gr.Group():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"{tabname}_gallery", elem_classes="gradio-gallery").style(grid=4)

            generation_info = None
            with gr.Column():
                with gr.Row():
                    if tabname != "extras":
                        save = gr.Button('Save', elem_id=f'save_{tabname}')

                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])
                    button_id = "hidden_element" if shared.cmd_opts.hide_ui_dir_config else 'open_folder'
                    open_folder_button = gr.Button(folder_symbol, elem_id=button_id, visible=False)

                open_folder_button.click(
                    fn=lambda: open_folder(opts.outdir_samples or outdir),
                    inputs=[],
                    outputs=[],
                )

                if tabname != "extras":
                    with gr.Row():
                        do_make_zip = gr.Checkbox(label="Make Zip when Save?", value=False, visible=False)

                    with gr.Row():
                        download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False)

                    with gr.Group():
                        html_info = gr.HTML()
                        generation_info = gr.Textbox(visible=False)
                        if tabname == 'txt2img' or tabname == 'img2img':
                            generation_info_button = gr.Button(visible=False, elem_id=f"{tabname}_generation_info_button")
                            generation_info_button.click(
                                fn=update_generation_info,
                                _js="(x, y) => [x, y, selected_gallery_index()]",
                                inputs=[generation_info, html_info],
                                outputs=[html_info],
                                preprocess=False
                            )

                        save.click(
                            fn=wrap_gradio_call(save_files),
                            _js="(x, y, z, w) => [x, y, z, selected_gallery_index()]",
                            inputs=[
                                generation_info,
                                result_gallery,
                                do_make_zip,
                                html_info
                            ],
                            outputs=[
                                download_files,
                                html_info,
                                html_info,
                                html_info,
                                html_info
                            ]
                        )
                else:
                    html_info_x = gr.HTML()
                    html_info = gr.HTML()
                parameters_copypaste.bind_buttons(buttons, result_gallery, "txt2img" if tabname == "txt2img" else None)
                return result_gallery, generation_info if tabname != "extras" else html_info_x, html_info

def create_ui():
    import modules.img2img
    import modules.txt2img

    reload_javascript()

    parameters_copypaste.reset()

    modules.scripts.scripts_current = modules.scripts.scripts_txt2img
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)

    # print(modules.scripts.scripts_data)

    interfaces = []

    ##add River
    def list_objects(bucket,prefix=''):
        response = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
        objects = response['Contents'] if response.get('Contents') else []
        return [obj['Key'] for obj in objects]

    def image_viewer(path,cols_width,current_only,request:gr.Request):
        if current_only:
            username = shared.get_webui_username(request)
            path = path+'/'+username
        dirs = path.replace('s3://','').split('/')
        prefix = '/'.join(dirs[1:])
        bucket = dirs[0]
        objects = list_objects(bucket,prefix)
        image_url = []
        for object_key in objects:
            if object_key.endswith('.jpg') or object_key.endswith('.jpeg') or object_key.endswith('.png'):
                image_url.append([s3_client.generate_presigned_url('get_object', Params={
                    'Bucket': bucket, 'Key': object_key}, ExpiresIn=3600),object_key.split('/')[-1]])
        image_tags = ""
        for image,key in image_url:
            image_tags += f"<div style='padding: 5px;';widget><a href='{image}' target='_blank'><img src='{image}'></a><div style='color:#7d8998'>{key}</div></div>"
        div = f"<div style='display: grid; grid-template-columns: repeat({cols_width}, 1fr); grid-gap: 10px;border: 1px solid #e9ebed; border-radius:10px;padding: 10px;'>{image_tags}</div>"
        return div

    
    with gr.Blocks(analytics_enabled=False) as imagesviewer_interface:
        with gr.Row():
            with gr.Column(scale=3):
                images_s3_path = gr.Textbox(label="Input S3 path of images",visible=True, value = get_default_sagemaker_bucket()+'/stable-diffusion-webui/generated')
                dummy_images_s3_path = gr.Textbox(label="Input S3 path of images",visible=False, interactive=False,
                                                  value = get_default_sagemaker_bucket()+'/stable-diffusion-webui/generated/{username}')

            with gr.Column(scale=1):
                show_user_only = gr.Checkbox(label="Show current user's images only", value=True,visible=True,interactive=True)
            with gr.Column(scale=1):
                cols_width = gr.Slider(minimum=4, maximum=20, step=1, label="columns width", value=8)
            with gr.Column(scale=1):
                images_s3_path_btn = gr.Button(value="Submit",variant='primary')
        with gr.Row():
            result = gr.HTML("<div style='height:300px;border:1px solid #e9ebed;border-radius:10px;'></div>")
        images_s3_path_btn.click(fn=image_viewer, inputs=[images_s3_path,cols_width,show_user_only], outputs=[result])


    ## end
    with gr.Blocks(analytics_enabled=False) as pnginfo_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                image = gr.Image(elem_id="pnginfo_image", label="Source", source="upload", interactive=True, type="pil")

            with gr.Column(variant='panel'):
                html = gr.HTML()
                generation_info = gr.Textbox(visible=False)
                html2 = gr.HTML()
                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])
                parameters_copypaste.bind_buttons(buttons, image, generation_info)

        image.change(
            fn=wrap_gradio_call(modules.extras.run_pnginfo),
            inputs=[image],
            outputs=[html, generation_info, html2],
        )
    
    script_callbacks.ui_settings_callback()

    ui_tabs = script_callbacks.ui_tabs_callback()
    
    dreambooth_tab = None
    images_history_ui_tab = None
    # for ui_tab in ui_tabs: 
    #     if ui_tab[2] != 'dreambooth_interface' :
    #         interfaces += [ui_tab]
    #     else:
    #         dreambooth_tab = ui_tab[0]
    for ui_tab in ui_tabs: 
        if ui_tab[2] == 'dreambooth_interface':
            dreambooth_tab = ui_tab[0]
        elif ui_tab[2] == 'images_history':
            images_history_ui_tab = ui_tab
        else:
            interfaces += [ui_tab]


    def create_setting_component(key, is_quicksettings=False):
        def fun():
            return opts.data[key] if key in opts.data else opts.data_labels[key].default

        info = opts.data_labels[key]
        t = type(info.default)

        args = info.component_args() if callable(info.component_args) else info.component_args

        if info.component is not None:
            comp = info.component
        elif t == str:
            comp = gr.Textbox
        elif t == int:
            comp = gr.Number
        elif t == bool:
            comp = gr.Checkbox
        else:
            raise Exception(f'bad options item type: {str(t)} for key {key}')

        elem_id = "setting_"+key

        if info.refresh is not None:
            if is_quicksettings:
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
                create_refresh_button(res, info.refresh, info.component_args, "refresh_" + key)
            else:
                with gr.Row(variant="compact"):
                    res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
                    create_refresh_button(res, info.refresh, info.component_args, "refresh_" + key)
        else:
            res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))

        if key == 'sagemaker_endpoint':
            shared.sagemaker_endpoint_component = res

        if key == 'sd_model_checkpoint':
            shared.sd_model_checkpoint_component = res

        if key == 'sd_hypernetwork':
            shared.sd_hypernetwork_component = res
        return res

    components = []
    global component_dict

   
    opts.reorder()

    def run_settings(username, *args):
        changed = []

        if not username or username == '':
            return opts.dumpjson(), f'{len(changed)} settings changed: {", ".join(changed)}.'

        for key, value, comp in zip(opts.data_labels.keys(), args, components):
            assert comp == dummy_component or opts.same_type(value, opts.data_labels[key].default), f"Bad value for setting {key}: {value}; expecting {type(opts.data_labels[key].default).__name__}"

        for key, value, comp in zip(opts.data_labels.keys(), args, components):
            if comp == dummy_component:
                continue

            if opts.set(key, value):
                changed.append(key)

        try:
            inputs = {
                'action': 'put',
                'username': username,
                'options': opts.dumpjson()
            }

            response = requests.post(url=f'{shared.api_endpoint}/sd/user', json=inputs)
            if response.status_code != 200:
                raise RuntimeError("Settings saved failed")
        except RuntimeError:
            return opts.dumpjson(), 'Settings changed without save'
        return opts.dumpjson(), 'Settings changed and saved'

    def run_settings_single(value, key, request : gr.Request):
        username = shared.get_webui_username(request)

        if username and username != '':
            if not opts.same_type(value, opts.data_labels[key].default):
                return gr.update(visible=True), opts.dumpjson()

            if not opts.set(key, value):
                return gr.update(value=getattr(opts, key)), opts.dumpjson()

            try:
                if username and username != '':
                    inputs = {
                        'action': 'edit',
                        'username': username,
                        'options': opts.dumpjson()
                    }

                    response = requests.post(url=f'{shared.api_endpoint}/sd/user', json=inputs)
                    if response.status_code != 200:
                        raise RuntimeError("Settings saved failed")
            except RuntimeError:
                return gr.update(visible=True), opts.dumpjson()

            return gr.update(value=value), opts.dumpjson()
        else:
            if not opts.same_type(value, opts.data_labels[key].default):
                return gr.update(visible=True), opts.dumpjson()

            if not opts.set(key, value):
                return gr.update(value=getattr(opts, key)), opts.dumpjson()

            return gr.update(value=value), opts.dumpjson()

    default_sagemaker_s3 = get_default_sagemaker_bucket()
    default_s3_path =  f"{default_sagemaker_s3}/stable-diffusion-webui/models/"
    with gr.Blocks(analytics_enabled=False) as settings_interface:
        dummy_component = gr.Label(visible=False)
        with gr.Row():
            settings_submit = gr.Button(value="Apply settings", variant='primary')
        with gr.Row():
            with gr.Column(scale=4):
                models_s3bucket = gr.Textbox(label="S3 path for downloading model files (E.g, s3://bucket-name/models/)",
                                            value=default_s3_path,visible=True)
            with gr.Column(scale=1):
                set_models_s3bucket_btn = gr.Button(value="Update model files path",elem_id='id_set_models_s3bucket',visible=True)
            with gr.Column(scale=1):
                reload_models_btn = gr.Button(value='Reload all models', elem_id='id_reload_all_models')

        
        
        result = gr.HTML()

        settings_cols = 3
        items_per_col = int(len(opts.data_labels) * 0.9 / settings_cols)

        quicksettings_names = [x.strip() for x in opts.quicksettings.split(",")]
        quicksettings_names = set(x for x in quicksettings_names if x != 'quicksettings')

        quicksettings_list = []

        cols_displayed = 0
        items_displayed = 0
        previous_section = None
        column = None

        with gr.Row(elem_id="settings").style(equal_height=False):
            for i, (k, item) in enumerate(opts.data_labels.items()):
                section_must_be_skipped = item.section[0] is None

                if previous_section != item.section and not section_must_be_skipped:
                    if cols_displayed < settings_cols and (items_displayed >= items_per_col or previous_section is None):
                        if column is not None:
                            column.__exit__()

                        column = gr.Column(variant='panel')
                        column.__enter__()

                        items_displayed = 0
                        cols_displayed += 1

                    previous_section = item.section

                    elem_id, text = item.section
                    gr.HTML(elem_id="settings_header_text_{}".format(elem_id), value='<h1 class="gr-button-lg">{}</h1>'.format(text))

                if k in quicksettings_names and not shared.cmd_opts.freeze_settings:
                    quicksettings_list.append((i, k, item))
                    components.append(dummy_component)
                elif section_must_be_skipped:
                    components.append(dummy_component)
                else:
                    component = create_setting_component(k)
                    component_dict[k] = component
                    components.append(component)
                    items_displayed += 1

        with gr.Row():
            request_notifications = gr.Button(value='Request browser notifications', elem_id="request_notifications")
            download_localization = gr.Button(value='Download localization template', elem_id="download_localization")

        with gr.Row():
            reload_script_bodies = gr.Button(value='Reload custom script bodies (No ui updates, No restart)', variant='secondary')
            restart_gradio = gr.Button(value='Restart Gradio and Refresh components (Custom Scripts, ui.py, js and css only)', variant='primary')

        request_notifications.click(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            _js='function(){}'
        )

        download_localization.click(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            _js='download_localization'
        )

        def reload_scripts():
            modules.scripts.reload_script_body_only()
            reload_javascript()  # need to refresh the html page

        reload_script_bodies.click(
            fn=reload_scripts,
            inputs=[],
            outputs=[]
        )

        def request_restart():
            shared.state.interrupt()
            shared.state.need_restart = True

        restart_gradio.click(
            fn=request_restart,
            _js='restart_reload',
            inputs=[],
            outputs=[],
        )

        def reload_all_models():
            sagemaker_endpoint=shared.opts.sagemaker_endpoint
            print(f'reload_all_models from:{sagemaker_endpoint}')
            inputs = {'task': 'reload-all-models'}
            params = {'endpoint_name': sagemaker_endpoint}
            response = requests.post(url=f'{shared.api_endpoint}/inference', params=params, json=inputs)
            if response.status_code == 200:
                return f'[{sagemaker_endpoint}] reload_all_models success'
            else:
                print(response.status_code )
                return f'[{sagemaker_endpoint}] reload_all_models failed'
        
        reload_models_btn.click(
            fn=reload_all_models,
            inputs=[],
            outputs=[result]
        )
        
        # River
        def set_models_s3bucket(bucket_name):
            if bucket_name == '':
                return 'Error, please configure a S3 bucket for downloading model files'
            sagemaker_endpoint=shared.opts.sagemaker_endpoint
            print(f'set_models_s3bucket to:{sagemaker_endpoint}')
            inputs = {'task': 'set-models-bucket',
                        'models_bucket':bucket_name}
            params = {'endpoint_name': 
                        sagemaker_endpoint}
            response = requests.post(url=f'{shared.api_endpoint}/inference', params=params, json=inputs)
            if response.status_code == 200:
                return f'[{sagemaker_endpoint}] set bucket succeess'
            else:
                print(response.status_code )
                return f'[{sagemaker_endpoint}] set bucket failed'

        
        set_models_s3bucket_btn.click(
            fn=set_models_s3bucket,
            inputs=[models_s3bucket],
            outputs=[result]

        )

        if column is not None:
            column.__exit__()

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        txt2img_prompt, roll, txt2img_prompt_style, txt2img_negative_prompt, txt2img_prompt_style2, submit, _, _, txt2img_prompt_style_apply, txt2img_save_style, txt2img_paste, extra_networks_button, token_counter, token_button = create_toprow(is_img2img=False)
        txt_prompt_img = gr.File(label="", elem_id="txt2img_prompt_image", file_count="single", type="bytes", visible=False)

        with FormRow(variant='compact', elem_id="txt2img_extra_networks", visible=False) as extra_networks:
            from modules import ui_extra_networks
            extra_networks_ui = ui_extra_networks.create_ui(extra_networks, extra_networks_button, 'txt2img')

        with gr.Row(elem_id='txt2img_progress_row'):
            with gr.Column(scale=1):
                pass

            with gr.Column(scale=1):
                progressbar = gr.HTML(elem_id="txt2img_progressbar")
                txt2img_preview = gr.Image(elem_id='txt2img_preview', visible=False)
                setup_progressbar(progressbar, txt2img_preview, 'txt2img')

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20)
                sampler_index = gr.Radio(label='Sampling method', elem_id="txt2img_sampling", choices=[x.name for x in samplers], value=samplers[0].name, type="index")

                with gr.Group():
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512, elem_id='txt2img_width')
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512, elem_id='txt2img_height')

                with gr.Row():
                    restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1)
                    tiling = gr.Checkbox(label='Tiling', value=False)
                    enable_hr = gr.Checkbox(label='Highres. fix', value=False)

                with gr.Row(visible=False) as hr_options:
                    firstphase_width = gr.Slider(minimum=0, maximum=1024, step=64, label="Firstpass width", value=0)
                    firstphase_height = gr.Slider(minimum=0, maximum=1024, step=64, label="Firstpass height", value=0)
                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7)

                with gr.Row(equal_height=True):
                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1)
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1)

                cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0)

                seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs()

                with gr.Group():
                    custom_inputs = modules.scripts.scripts_txt2img.setup_ui()

            txt2img_gallery, generation_info, html_info = create_output_panel("txt2img", opts.outdir_txt2img_samples)
            parameters_copypaste.bind_buttons({"txt2img": txt2img_paste}, None, txt2img_prompt)

            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img),
                _js="submit",
                inputs=[
                    txt2img_prompt,
                    txt2img_negative_prompt,
                    txt2img_prompt_style,
                    txt2img_prompt_style2,
                    steps,
                    sampler_index,
                    restore_faces,
                    tiling,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    seed,
                    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
                    height,
                    width,
                    enable_hr,
                    denoising_strength,
                    firstphase_width,
                    firstphase_height,
                ] + custom_inputs + [shared.sagemaker_endpoint_component],

                outputs=[
                    txt2img_gallery,
                    generation_info,
                    html_info
                ],
                show_progress=False,
            )

            txt2img_prompt.submit(**txt2img_args)
            submit.click(**txt2img_args)

            txt_prompt_img.change(
                fn=modules.images.image_data,
                inputs=[
                    txt_prompt_img
                ],
                outputs=[
                    txt2img_prompt,
                    txt_prompt_img
                ]
            )

            enable_hr.change(
                fn=lambda x: gr_show(x),
                inputs=[enable_hr],
                outputs=[hr_options],
            )

            roll.click(
                fn=roll_artist,
                _js="update_txt2img_tokens",
                inputs=[
                    txt2img_prompt,
                ],
                outputs=[
                    txt2img_prompt,
                ]
            )

            txt2img_paste_fields = [
                (txt2img_prompt, "Prompt"),
                (txt2img_negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_index, "Sampler"),
                (restore_faces, "Face restoration"),
                (cfg_scale, "CFG scale"),
                (seed, "Seed"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation seed strength"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (denoising_strength, "Denoising strength"),
                (enable_hr, lambda d: "Denoising strength" in d),
                (hr_options, lambda d: gr.Row.update(visible="Denoising strength" in d)),
                (firstphase_width, "First pass size-1"),
                (firstphase_height, "First pass size-2"),
                *modules.scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields)

            txt2img_preview_params = [
                txt2img_prompt,
                txt2img_negative_prompt,
                steps,
                sampler_index,
                cfg_scale,
                seed,
                width,
                height,
            ]

            token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_prompt, steps], outputs=[token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui, txt2img_gallery)

    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)

    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        img2img_prompt, roll, img2img_prompt_style, img2img_negative_prompt, img2img_prompt_style2, submit, img2img_interrogate, img2img_deepbooru, img2img_prompt_style_apply, img2img_save_style, img2img_paste, extra_networks_button, token_counter, token_button = create_toprow(is_img2img=True)

        with gr.Row(elem_id='img2img_progress_row'):
            img2img_prompt_img = gr.File(label="", elem_id="img2img_prompt_image", file_count="single", type="bytes", visible=False)

            with gr.Column(scale=1):
                pass

            with gr.Column(scale=1):
                progressbar = gr.HTML(elem_id="img2img_progressbar")
                img2img_preview = gr.Image(elem_id='img2img_preview', visible=False)
                setup_progressbar(progressbar, img2img_preview, 'img2img')

        with FormRow(variant='compact', elem_id="img2img_extra_networks", visible=False) as extra_networks:
            from modules import ui_extra_networks
            extra_networks_ui_img2img = ui_extra_networks.create_ui(extra_networks, extra_networks_button, 'img2img')

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):

                with gr.Tabs(elem_id="mode_img2img") as tabs_img2img_mode:
                    with gr.TabItem('img2img', id='img2img'):
                        init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", tool=cmd_opts.gradio_img2img_tool).style(height=480)

                    with gr.TabItem('Inpaint', id='inpaint'):
                        init_img_with_mask = gr.Image(label="Image for inpainting with mask", show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool=cmd_opts.gradio_inpaint_tool, image_mode="RGBA").style(height=480)
                        init_img_with_mask_orig = gr.State(None)

                        use_color_sketch = cmd_opts.gradio_inpaint_tool == "color-sketch"
                        if use_color_sketch:
                            def update_orig(image, state):
                                if image is not None:
                                    same_size = state is not None and state.size == image.size
                                    has_exact_match = np.any(np.all(np.array(image) == np.array(state), axis=-1))
                                    edited = same_size and has_exact_match
                                    return image if not edited or state is None else state

                            init_img_with_mask.change(update_orig, [init_img_with_mask, init_img_with_mask_orig], init_img_with_mask_orig)

                        init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", visible=False, elem_id="img_inpaint_base")
                        init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", visible=False, elem_id="img_inpaint_mask")

                        with gr.Row():
                            mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4)
                            mask_alpha = gr.Slider(label="Mask transparency", interactive=use_color_sketch, visible=use_color_sketch)

                        with gr.Row():
                            mask_mode = gr.Radio(label="Mask mode", show_label=False, choices=["Draw mask", "Upload mask"], type="index", value="Draw mask", elem_id="mask_mode")
                            inpainting_mask_invert = gr.Radio(label='Masking mode', show_label=False, choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index")

                        inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='original', type="index")

                        with gr.Row():
                            inpaint_full_res = gr.Checkbox(label='Inpaint at full resolution', value=False)
                            inpaint_full_res_padding = gr.Slider(label='Inpaint at full resolution padding, pixels', minimum=0, maximum=256, step=4, value=32)

                    with gr.TabItem('Batch img2img', id='batch'):
                        hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                        gr.HTML(f"<p class=\"text-gray-500\">Process images in a directory on the same machine where the server is running.<br>Use an empty output directory to save pictures normally instead of writing to the output directory.{hidden}</p>")
                        img2img_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs)
                        img2img_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs)

                    if not cmd_opts.pureui:
                        with gr.TabItem('Batch img2img', id='batch'):
                            hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                            gr.HTML(f"<p class=\"text-gray-500\">Process images in a directory on the same machine where the server is running.<br>Use an empty output directory to save pictures normally instead of writing to the output directory.{hidden}</p>")
                            img2img_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs)
                            img2img_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs)

                with gr.Row():
                    resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", show_label=False, choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value="Just resize")

                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20)
                sampler_index = gr.Radio(label='Sampling method', choices=[x.name for x in samplers_for_img2img], value=samplers_for_img2img[0].name, type="index")

                with gr.Group():
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512, elem_id="img2img_width")
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512, elem_id="img2img_height")

                with gr.Row():
                    restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1)
                    tiling = gr.Checkbox(label='Tiling', value=False)

                with gr.Row():
                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1)
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1)

                with gr.Group():
                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0)
                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.75)

                seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs()

                with gr.Group():
                    custom_inputs = modules.scripts.scripts_img2img.setup_ui()

            img2img_gallery, generation_info, html_info = create_output_panel("img2img", opts.outdir_img2img_samples)
            parameters_copypaste.bind_buttons({"img2img": img2img_paste}, None, img2img_prompt)

            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            img2img_prompt_img.change(
                fn=modules.images.image_data,
                inputs=[
                    img2img_prompt_img
                ],
                outputs=[
                    img2img_prompt,
                    img2img_prompt_img
                ]
            )

            mask_mode.change(
                lambda mode, img: {
                    init_img_with_mask: gr_show(mode == 0),
                    init_img_inpaint: gr_show(mode == 1),
                    init_mask_inpaint: gr_show(mode == 1),
                },
                inputs=[mask_mode, init_img_with_mask],
                outputs=[
                    init_img_with_mask,
                    init_img_inpaint,
                    init_mask_inpaint,
                ],
            )

            img2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.img2img.img2img),
                _js="submit_img2img",
                inputs=[
                    dummy_component,
                    img2img_prompt,
                    img2img_negative_prompt,
                    img2img_prompt_style,
                    img2img_prompt_style2,
                    init_img,
                    init_img_with_mask,
                    init_img_with_mask_orig,
                    init_img_inpaint,
                    init_mask_inpaint,
                    mask_mode,
                    steps,
                    sampler_index,
                    mask_blur,
                    mask_alpha,
                    inpainting_fill,
                    restore_faces,
                    tiling,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    denoising_strength,
                    seed,
                    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
                    height,
                    width,
                    resize_mode,
                    inpaint_full_res,
                    inpaint_full_res_padding,
                    inpainting_mask_invert,
                    img2img_batch_input_dir if not cmd_opts.pureui else dummy_component,
                    img2img_batch_output_dir if not cmd_opts.pureui else dummy_component,
                ] + custom_inputs + [shared.sagemaker_endpoint_component],
                outputs=[
                    img2img_gallery,
                    generation_info,
                    html_info
                ],
                show_progress=False,
            )

            img2img_prompt.submit(**img2img_args)
            submit.click(**img2img_args)

            img2img_interrogate.click(
                fn=interrogate,
                inputs=[init_img],
                outputs=[img2img_prompt],
            )

            img2img_deepbooru.click(
                fn=interrogate_deepbooru,
                inputs=[init_img],
                outputs=[img2img_prompt],
            )


            roll.click(
                fn=roll_artist,
                _js="update_img2img_tokens",
                inputs=[
                    img2img_prompt,
                ],
                outputs=[
                    img2img_prompt,
                ]
            )

            prompts = [(txt2img_prompt, txt2img_negative_prompt), (img2img_prompt, img2img_negative_prompt)]
            style_dropdowns = [(txt2img_prompt_style, txt2img_prompt_style2), (img2img_prompt_style, img2img_prompt_style2)]
            style_js_funcs = ["update_txt2img_tokens", "update_img2img_tokens"]

            for button, (prompt, negative_prompt) in zip([txt2img_save_style, img2img_save_style], prompts):
                button.click(
                    fn=add_style,
                    _js="ask_for_style_name",
                    # Have to pass empty dummy component here, because the JavaScript and Python function have to accept
                    # the same number of parameters, but we only know the style-name after the JavaScript prompt
                    inputs=[dummy_component, prompt, negative_prompt],
                    outputs=[txt2img_prompt_style, img2img_prompt_style, txt2img_prompt_style2, img2img_prompt_style2],
                )

            for button, (prompt, negative_prompt), (style1, style2), js_func in zip([txt2img_prompt_style_apply, img2img_prompt_style_apply], prompts, style_dropdowns, style_js_funcs):
                button.click(
                    fn=apply_styles,
                    _js=js_func,
                    inputs=[prompt, negative_prompt, style1, style2],
                    outputs=[prompt, negative_prompt, style1, style2],
                )

            token_button.click(fn=update_token_counter, inputs=[img2img_prompt, steps], outputs=[token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui_img2img, img2img_gallery)

            img2img_paste_fields = [
                (img2img_prompt, "Prompt"),
                (img2img_negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_index, "Sampler"),
                (restore_faces, "Face restoration"),
                (cfg_scale, "CFG scale"),
                (seed, "Seed"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation seed strength"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (denoising_strength, "Denoising strength"),
                (mask_blur, "Mask blur"),
                *modules.scripts.scripts_img2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("img2img", init_img, img2img_paste_fields)
            parameters_copypaste.add_paste_fields("inpaint", init_img_with_mask, img2img_paste_fields)

    modules.scripts.scripts_current = None

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="mode_extras"):
                    with gr.TabItem('Single Image'):
                        extras_image = gr.Image(label="Source", source="upload", interactive=True, type="pil")

                    with gr.TabItem('Batch Process'):
                        image_batch = gr.File(label="Batch Process", file_count="multiple", interactive=True, type="file")
                        if not cmd_opts.pureui:
                            with gr.TabItem('Batch from Directory', visible=False):
                                extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.")
                                extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.")
                                show_extras_results = gr.Checkbox(label='Show result images', value=True)

                    if not cmd_opts.pureui:
                        with gr.TabItem('Batch from Directory'):
                            extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.")
                            extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.")
                            show_extras_results = gr.Checkbox(label='Show result images', value=True)

                submit = gr.Button('Generate', elem_id="extras_generate", variant='primary')

                with gr.Tabs(elem_id="extras_resize_mode"):
                    with gr.TabItem('Scale by'):
                        upscaling_resize = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label="Resize", value=4)
                    with gr.TabItem('Scale to'):
                        with gr.Group():
                            with gr.Row():
                                upscaling_resize_w = gr.Number(label="Width", value=512, precision=0)
                                upscaling_resize_h = gr.Number(label="Height", value=512, precision=0)
                            upscaling_crop = gr.Checkbox(label='Crop to fit', value=True)

                with gr.Group():
                    extras_upscaler_1 = gr.Radio(label='Upscaler 1', elem_id="extras_upscaler_1", choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index")

                with gr.Group():
                    extras_upscaler_2 = gr.Radio(label='Upscaler 2', elem_id="extras_upscaler_2", choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index")
                    extras_upscaler_2_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Upscaler 2 visibility", value=1)

                with gr.Group():
                    gfpgan_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="GFPGAN visibility", value=0, interactive=modules.gfpgan_model.have_gfpgan)

                with gr.Group():
                    codeformer_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="CodeFormer visibility", value=0, interactive=modules.codeformer_model.have_codeformer)
                    codeformer_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="CodeFormer weight (0 = maximum effect, 1 = minimum effect)", value=0, interactive=modules.codeformer_model.have_codeformer)

                with gr.Group():
                    upscale_before_face_fix = gr.Checkbox(label='Upscale Before Restoring Faces', value=False)

            result_images, html_info_x, html_info = create_output_panel("extras", opts.outdir_extras_samples)

        submit.click(
            fn=wrap_gradio_gpu_call(modules.extras.run_extras),
            _js="get_extras_tab_index",
            inputs=[
                dummy_component,
                dummy_component,
                extras_image,
                image_batch,
                extras_batch_input_dir if not cmd_opts.pureui else dummy_component,
                extras_batch_output_dir if not cmd_opts.pureui else dummy_component,
                show_extras_results if not cmd_opts.pureui else dummy_component,
                gfpgan_visibility,
                codeformer_visibility,
                codeformer_weight,
                upscaling_resize,
                upscaling_resize_w,
                upscaling_resize_h,
                upscaling_crop,
                extras_upscaler_1,
                extras_upscaler_2,
                extras_upscaler_2_visibility,
                upscale_before_face_fix,
            ] + [shared.sagemaker_endpoint_component],
            outputs=[
                result_images,
                html_info_x,
                html_info,
            ]
        )
        parameters_copypaste.add_paste_fields("extras", extras_image, None)

        extras_image.change(
            fn=modules.extras.clear_cache,
            inputs=[], outputs=[]
        )

    def load_checkpoints_from_s3_uri(model_s3url,load_all_user,request:gr.Request):
        username = shared.get_webui_username(request)
        print(username)
        return modules.model_merger.load_checkpoints_from_s3_uri(model_s3url,load_all_user,username)

    with gr.Blocks(analytics_enabled=False) as modelmerger_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column():
                gr.HTML(value="<p>Merged checkpoints will be put in the specified output S3 location</p>")
                default_ckpt_s3 = get_default_sagemaker_bucket()+'/stable-diffusion-webui/models/Stable-diffusion/'
                # default_merge_output_s3 =  default_ckpt_s3
                with gr.Row():
                        dummy_s3uri = gr.Textbox(label="Checkpoint S3 URI", elem_id="dummy_chkpt_s3uri",
                                                    value='模型存放位置:'+default_ckpt_s3+'{用户名}',
                                                    lines=2, visible=False,interactive=False)
                        chkpt_s3uri = gr.Textbox(label="Checkpoint S3 URI", elem_id="chkpt_s3uri", value= default_ckpt_s3,lines=2,visible=True)
                        merge_output_s3uri = gr.Textbox(label="Merge Result S3 URI",lines=2, visible=True,placeholder='（选填），默认输出位置:'+default_ckpt_s3+'{用户名}')
                with gr.Row():
                    with gr.Column(scale=1):
                        load_all_user = gr.Checkbox(label="Don't load other user's models",interactive=True, value=True,visible=True)
                    with gr.Column(scale=2):
                        chkpt_s3uri_button = gr.Button(value="Load Checkpoints", variant='primary')
                    
               

                with gr.Row():
                    primary_model_name = gr.Dropdown(modules.model_merger.get_checkpoints_to_merge(), elem_id="modelmerger_primary_model_name", label="Primary model (A)")
                    secondary_model_name = gr.Dropdown(modules.model_merger.get_checkpoints_to_merge(), elem_id="modelmerger_secondary_model_name", label="Secondary model (B)")
                    tertiary_model_name = gr.Dropdown(modules.model_merger.get_checkpoints_to_merge(), elem_id="modelmerger_tertiary_model_name", label="Tertiary model (C)")
                custom_name = gr.Textbox(label="Custom Name (Optional)")
                interp_amount = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Multiplier (M) - set to 0 to get model A', value=0.3)
                interp_method = gr.Radio(choices=["Weighted sum", "Add difference"], value="Weighted sum", label="Interpolation Method")


                with gr.Row():
                    checkpoint_format = gr.Radio(choices=["ckpt", "safetensors"], value="ckpt", label="Checkpoint format")
                    save_as_half = gr.Checkbox(value=False, label="Save as float16")

                modelmerger_merge = gr.Button(elem_id="modelmerger_merge", value="Merge", variant='primary')

            with gr.Column():
                submit_result = gr.Textbox(elem_id="modelmerger_result", show_label=False)
        chkpt_s3uri_button.click(
                        fn=load_checkpoints_from_s3_uri,
                        inputs=[chkpt_s3uri,load_all_user],
                        outputs=[primary_model_name, secondary_model_name, tertiary_model_name])

        # A periodic function to check the submit output
        modelmerger_interface.load(modules.model_merger.get_processing_job_status,
                                   inputs=None, outputs=submit_result,
                                   every=10, queue=True)

    with gr.Blocks(analytics_enabled=False) as train_interface:
        with gr.Row().style(equal_height=False):
            gr.HTML(value="<p style='margin-bottom: 0.7em'>See <b><a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\">wiki</a></b> for detailed explanation.</p>")

        with gr.Row().style(equal_height=False):
            with gr.Tabs(elem_id="train_tabs"):
                ## Begin add s3 images upload interface by River
                def upload_to_s3(imgs,request : gr.Request):
                    username = shared.get_webui_username(request)
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
                
                with gr.Tab(label="Upload Train Images to S3"):
                    upload_files = gr.Files(label="Files")
                    url_output = gr.Textbox(label="Output S3 folder")
                    sub_btn = gr.Button(value="Upload Images",elem_id='id_upload_train_images',variant='primary')
                    sub_btn.click(fn=upload_to_s3, inputs=upload_files, outputs=url_output)
                ## End add s3 images upload interface by River
                with gr.Tab(label="Train Embedding"):
                    gr.HTML(value="<p style='margin-bottom: 0.7em'>Train an embedding; you must specify a directory with a set of 1:1 ratio images <a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\" style=\"font-weight:bold;\">[wiki]</a></p>")

                    with gr.Box():
                        gr.HTML(value="<p style='margin-bottom: 1.5em'><b>Embedding settings</b></p>")

                        new_embedding_name = gr.Textbox(label="Name")
                        initialization_text = gr.Textbox(label="Initialization text", value="*")
                        nvpt = gr.Slider(label="Number of vectors per token", minimum=1, maximum=75, step=1, value=1)
                        overwrite_old_embedding = gr.Checkbox(value=False, label="Overwrite Old Embedding")

                    with gr.Box():
                        gr.HTML(value="<p style='margin-bottom: 1.5em'><b>Image preprocess settings</b></p>")

                        embedding_images_s3uri = gr.Textbox(label='Images S3 URI')
                        embedding_models_s3uri = gr.Textbox(label='Models S3 URI')
                        embedding_process_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                        embedding_process_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                        embedding_preprocess_txt_action = gr.Dropdown(label='Existing Caption txt Action', value="ignore", choices=["ignore", "copy", "prepend", "append"])

                        with gr.Row():
                            embedding_process_flip = gr.Checkbox(label='Create flipped copies')
                            embedding_process_split = gr.Checkbox(label='Split oversized images')
                            embedding_process_focal_crop = gr.Checkbox(label='Auto focal point crop')
                            embedding_process_caption = gr.Checkbox(label='Use BLIP for caption')
                            embedding_process_caption_deepbooru = gr.Checkbox(label='Use deepbooru for caption', visible=True)

                        with gr.Row(visible=False) as embedding_process_split_extra_row:
                            embedding_process_split_threshold = gr.Slider(label='Split image threshold', value=0.5, minimum=0.0, maximum=1.0, step=0.05)
                            embedding_process_overlap_ratio = gr.Slider(label='Split image overlap ratio', value=0.2, minimum=0.0, maximum=0.9, step=0.05)

                        with gr.Row(visible=False) as embedding_process_focal_crop_row:
                            embedding_process_focal_crop_face_weight = gr.Slider(label='Focal point face weight', value=0.9, minimum=0.0, maximum=1.0, step=0.05)
                            embedding_process_focal_crop_entropy_weight = gr.Slider(label='Focal point entropy weight', value=0.15, minimum=0.0, maximum=1.0, step=0.05)
                            embedding_process_focal_crop_edges_weight = gr.Slider(label='Focal point edges weight', value=0.5, minimum=0.0, maximum=1.0, step=0.05)
                            embedding_process_focal_crop_debug = gr.Checkbox(label='Create debug image')

                        embedding_process_split.change(
                            fn=lambda show: gr_show(show),
                            inputs=[embedding_process_split],
                            outputs=[embedding_process_split_extra_row],
                        )

                        embedding_process_focal_crop.change(
                            fn=lambda show: gr_show(show),
                            inputs=[embedding_process_focal_crop],
                            outputs=[embedding_process_focal_crop_row],
                        )

                    with gr.Box():
                        gr.HTML(value="<p style='margin-bottom: 1.5em'><b>Train settings</b></p>")

                        with gr.Row():
                            with gr.Column():
                                embedding_training_instance_type = gr.Dropdown(label='Instance type', value="ml.g4dn.xlarge", choices=training_instance_types)
                            with gr.Column():
                                embedding_training_instance_count = gr.Number(label='Instance count', value=1, precision=0)

                        with gr.Row():
                            embedding_learn_rate = gr.Textbox(label='Embedding Learning rate', placeholder="Embedding Learning rate", value="0.005")

                        embedding_batch_size = gr.Number(label='Batch size', value=1, precision=0)
                        embedding_gradient_step = gr.Number(label='Gradient accumulation steps', value=1, precision=0)
                        embedding_training_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                        embedding_training_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                        embedding_steps = gr.Number(label='Max steps', value=100000, precision=0)
                        embedding_create_image_every = gr.Number(label='Save an image to log directory every N steps, 0 to disable', value=500, precision=0)
                        embedding_save_embedding_every = gr.Number(label='Save a copy of embedding to log directory every N steps, 0 to disable', value=500, precision=0)
                        embedding_save_image_with_stored_embedding = gr.Checkbox(label='Save images with embedding in PNG chunks', value=True)
                        embedding_preview_from_txt2img = gr.Checkbox(label='Read parameters (prompt, etc...) from txt2img tab when making previews', value=False)
                        with gr.Row():
                            embedding_shuffle_tags = gr.Checkbox(label="Shuffle tags by ',' when creating prompts.", value=False)
                            embedding_tag_drop_out = gr.Slider(minimum=0, maximum=1, step=0.1, label="Drop out tags when creating prompts.", value=0)
                        with gr.Row():
                            embedding_latent_sampling_method = gr.Radio(label='Choose latent sampling method', value="once", choices=['once', 'deterministic', 'random'])

                    with gr.Row():
                        with gr.Column(scale=3):
                            embedding_output = gr.Label(label='Output')
                             ##begin add train job info by River
                            embedding_training_job = gr.Markdown('Job detail')
                            ##end add train job info by River

                        with gr.Column():
                            create_train_embedding = gr.Button(value="Train Embedding", variant='primary')

                with gr.Tab(label="Train Hypernetwork"):
                    gr.HTML(value="<p style='margin-bottom: 0.7em'>Train an hypernetwork; you must specify a directory with a set of 1:1 ratio images <a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\" style=\"font-weight:bold;\">[wiki]</a></p>")

                    with gr.Box():
                        gr.HTML(value="<p style='margin-bottom: 1.5em'><b>Hypernetwork settings</b></p>")

                        new_hypernetwork_name = gr.Textbox(label="Name")
                        new_hypernetwork_sizes = gr.CheckboxGroup(label="Modules", value=["768", "320", "640", "1280"], choices=["768", "320", "640", "1280"])
                        new_hypernetwork_layer_structure = gr.Textbox("1, 2, 1", label="Enter hypernetwork layer structure", placeholder="1st and last digit must be 1. ex:'1, 2, 1'")
                        new_hypernetwork_activation_func = gr.Dropdown(value="linear", label="Select activation function of hypernetwork. Recommended : Swish / Linear(none)", choices=modules.hypernetworks.ui.keys)
                        new_hypernetwork_initialization_option = gr.Dropdown(value = "Normal", label="Select Layer weights initialization. Recommended: Kaiming for relu-like, Xavier for sigmoid-like, Normal otherwise", choices=["Normal", "KaimingUniform", "KaimingNormal", "XavierUniform", "XavierNormal"])
                        new_hypernetwork_add_layer_norm = gr.Checkbox(label="Add layer normalization")
                        new_hypernetwork_use_dropout = gr.Checkbox(label="Use dropout")
                        overwrite_old_hypernetwork = gr.Checkbox(value=False, label="Overwrite Old Hypernetwork")

                    with gr.Box():
                        gr.HTML(value="<p style='margin-bottom: 1.5em'><b>Image preprocess settings</b></p>")

                        hypernetwork_images_s3uir = gr.Textbox(label='Images S3 URI')
                        hypernetwork_models_s3uri = gr.Textbox(label='Models S3 URI')
                        hypernetwork_process_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                        hypernetwork_process_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                        hypernetwork_preprocess_txt_action = gr.Dropdown(label='Existing Caption txt Action', value="ignore", choices=["ignore", "copy", "prepend", "append"])

                        with gr.Row():
                            hypernetwork_process_flip = gr.Checkbox(label='Create flipped copies')
                            hypernetwork_process_split = gr.Checkbox(label='Split oversized images')
                            hypernetwork_process_focal_crop = gr.Checkbox(label='Auto focal point crop')
                            hypernetwork_process_caption = gr.Checkbox(label='Use BLIP for caption')
                            hypernetwork_process_caption_deepbooru = gr.Checkbox(label='Use deepbooru for caption', visible=True)

                        with gr.Row(visible=False) as hypernetwork_process_split_extra_row:
                            hypernetwork_process_split_threshold = gr.Slider(label='Split image threshold', value=0.5, minimum=0.0, maximum=1.0, step=0.05)
                            hypernetwork_process_overlap_ratio = gr.Slider(label='Split image overlap ratio', value=0.2, minimum=0.0, maximum=0.9, step=0.05)

                        with gr.Row(visible=False) as hypernetwork_process_focal_crop_row:
                            hypernetwork_process_focal_crop_face_weight = gr.Slider(label='Focal point face weight', value=0.9, minimum=0.0, maximum=1.0, step=0.05)
                            hypernetwork_process_focal_crop_entropy_weight = gr.Slider(label='Focal point entropy weight', value=0.15, minimum=0.0, maximum=1.0, step=0.05)
                            hypernetwork_process_focal_crop_edges_weight = gr.Slider(label='Focal point edges weight', value=0.5, minimum=0.0, maximum=1.0, step=0.05)
                            hypernetwork_process_focal_crop_debug = gr.Checkbox(label='Create debug image')

                        hypernetwork_process_split.change(
                            fn=lambda show: gr_show(show),
                            inputs=[hypernetwork_process_split],
                            outputs=[hypernetwork_process_split_extra_row],
                        )

                        hypernetwork_process_focal_crop.change(
                            fn=lambda show: gr_show(show),
                            inputs=[hypernetwork_process_focal_crop],
                            outputs=[hypernetwork_process_focal_crop_row],
                        )
                    with gr.Box():
                        gr.HTML(value="<p style='margin-bottom: 1.5em'><b>Train settings</b></p>")

                        with gr.Row():
                            with gr.Column():
                                hypernetwork_training_instance_type = gr.Dropdown(label='Instance type', value="ml.g4dn.xlarge", choices=training_instance_types)
                            with gr.Column():
                                hypernetwork_training_instance_count = gr.Number(label='Instance count', value=1, precision=0)

                        with gr.Row():
                            hypernetwork_learn_rate = gr.Textbox(label='Hypernetwork Learning rate', placeholder="Hypernetwork Learning rate", value="0.00001")

                        hypernetwork_batch_size = gr.Number(label='Batch size', value=1, precision=0)
                        hypernetwork_gradient_step = gr.Number(label='Gradient accumulation steps', value=1, precision=0)
                        hypernetwork_training_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                        hypernetwork_training_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                        hypernetwork_steps = gr.Number(label='Max steps', value=100000, precision=0)
                        hypernetwork_create_image_every = gr.Number(label='Save an image to log directory every N steps, 0 to disable', value=500, precision=0)
                        hypernetwork_save_embedding_every = gr.Number(label='Save a copy of embedding to log directory every N steps, 0 to disable', value=500, precision=0)
                        hypernetwork_save_image_with_stored_embedding = gr.Checkbox(label='Save images with embedding in PNG chunks', value=True)
                        hypernetwork_preview_from_txt2img = gr.Checkbox(label='Read parameters (prompt, etc...) from txt2img tab when making previews', value=False)
                        with gr.Row():
                            hypernetwork_shuffle_tags = gr.Checkbox(label="Shuffle tags by ',' when creating prompts.", value=False)
                            hypernetwork_tag_drop_out = gr.Slider(minimum=0, maximum=1, step=0.1, label="Drop out tags when creating prompts.", value=0)
                        with gr.Row():
                            hypernetwork_latent_sampling_method = gr.Radio(label='Choose latent sampling method', value="once", choices=['once', 'deterministic', 'random'])

                    with gr.Row():
                        with gr.Column(scale=3):
                            hypernetwork_output = gr.Label(label='Output')
                            ##begin add train job info by River
                            hypernetwork_training_job = gr.Markdown('Job detail')
                            ##end add train job info by River

                        with gr.Column():
                            create_train_hypernetwork = gr.Button(value="Train Hypernetwork", variant='primary')

                if dreambooth_tab:
                    with gr.Tab(label="Train Dreambooth"):
                        dreambooth_tab.render()

                def sagemaker_train_embedding(
                        request: gr.Request,
                        sd_model_checkpoint,
                        new_embedding_name,
                        initialization_text,
                        nvpt,
                        overwrite_old_embedding,
                        embedding_images_s3uri,
                        embedding_models_s3uri,
                        embedding_process_width,
                        embedding_process_height,
                        embedding_preprocess_txt_action,
                        embedding_process_flip,
                        embedding_process_split,
                        embedding_process_focal_crop,
                        embedding_process_caption,
                        embedding_process_caption_deepbooru,
                        embedding_process_split_threshold,
                        embedding_process_overlap_ratio,
                        embedding_process_focal_crop_face_weight,
                        embedding_process_focal_crop_entropy_weight,
                        embedding_process_focal_crop_edges_weight,
                        embedding_process_focal_crop_debug,
                        embedding_learn_rate,
                        embedding_batch_size,
                        embedding_gradient_step,
                        embedding_training_width,
                        embedding_training_height,
                        embedding_steps,
                        embedding_shuffle_tags,
                        embedding_tag_drop_out,
                        embedding_latent_sampling_method,
                        embedding_create_image_every,
                        embedding_save_embedding_every,
                        embedding_save_image_with_stored_embedding,
                        embedding_preview_from_txt2img,
                        embedding_training_instance_type,
                        embedding_training_instance_count,
                        *txt2img_preview_params
                    ):

                    username = shared.get_webui_username(request)

                    train_args = {
                        'embedding_settings': {
                            'name': new_embedding_name,
                            'nvpt': nvpt,
                            'overwrite_old': overwrite_old_embedding,
                            'initialization_text': initialization_text
                        },
                        'images_preprocessing_settings': {
                            'process_width': embedding_process_width,
                            'process_height': embedding_process_height,
                            'preprocess_txt_action': embedding_preprocess_txt_action,
                            'process_flip': embedding_process_flip,
                            'process_split': embedding_process_split,
                            'process_caption': embedding_process_caption,
                            'process_caption_deepbooru': embedding_process_caption_deepbooru,
                            'process_split_threshold': embedding_process_split_threshold,
                            'process_overlap_ratio': embedding_process_overlap_ratio,
                            'process_focal_crop': embedding_process_focal_crop,
                            'process_focal_crop_face_weight': embedding_process_focal_crop_face_weight,
                            'process_focal_crop_entropy_weight': embedding_process_focal_crop_entropy_weight,
                            'process_focal_crop_edges_weight': embedding_process_focal_crop_edges_weight,
                            'process_focal_crop_debug': embedding_process_focal_crop_debug
                        },
                        'train_embedding_settings':{
                            'learn_rate': embedding_learn_rate,
                            'batch_size': embedding_batch_size,
                            'gradient_step': embedding_gradient_step,
                            'training_width': embedding_training_width,
                            'training_height': embedding_training_height,
                            'steps': embedding_steps,
                            'shuffle_tags': embedding_shuffle_tags,
                            'tag_drop_out': embedding_tag_drop_out,
                            'latent_sampling_method': embedding_latent_sampling_method,
                            'create_image_every': embedding_create_image_every,
                            'save_embedding_every': embedding_save_embedding_every,
                            'save_image_with_stored_embedding': embedding_save_image_with_stored_embedding,
                            'preview_from_txt2img': embedding_preview_from_txt2img,
                            'txt2img_preview_params': txt2img_preview_params
                        }
                    }

                    sd_model_checkpoint = get_sd_model_checkpoint_from_title(sd_model_checkpoint)
                    hyperparameters = {
                        'train-args': json.dumps(json.dumps(train_args)),
                        'train-task': 'embedding',
                        'ckpt': '/opt/ml/input/data/models/{0}'.format(sd_model_checkpoint),
                        'username': username,
                        'api-endpoint': shared.api_endpoint
                    }
                    
                    inputs = {
                        'images': embedding_images_s3uri,
                        'models': embedding_models_s3uri
                    }
                    
                    data = {
                        'training_job_name': '',
                        'model_algorithm': 'stable-diffusion-webui',
                        'model_hyperparameters': hyperparameters,
                        'industrial_model': shared.industrial_model,
                        'instance_type': embedding_training_instance_type,
                        'instance_count': embedding_training_instance_count,
                        'inputs': inputs
                    }

                    response = requests.post(url=f'{shared.api_endpoint}/train', json=data)
                    if response.status_code == 200:
                        ##begin add train job info by River
                        training_job_url = response.text.replace('\"','')
                        return {
                            embedding_output: gr.update(value='Submit training job sucessful'),
                            embedding_training_job:gr.update(value=f'Job detail:[{training_job_url}]({training_job_url})')
                        ##end add train job info by River
                        }
                    else:
                        return {
                            embedding_output: gr.update(value=response.text)
                        }
                    
                def sagemaker_train_hypernetwork(
                        request: gr.Request,
                        sd_model_checkpoint,
                        new_hypernetwork_name,
                        new_hypernetwork_sizes,
                        new_hypernetwork_layer_structure,
                        new_hypernetwork_activation_func,
                        new_hypernetwork_initialization_option,
                        new_hypernetwork_add_layer_norm,
                        new_hypernetwork_use_dropout,
                        overwrite_old_hypernetwork,
                        hypernetwork_images_s3uri,
                        hypernetwork_models_s3uri,
                        hypernetwork_process_width,
                        hypernetwork_process_height,
                        hypernetwork_preprocess_txt_action,      
                        hypernetwork_process_flip,
                        hypernetwork_process_split,
                        hypernetwork_process_focal_crop,
                        hypernetwork_process_caption,
                        hypernetwork_process_caption_deepbooru,
                        hypernetwork_process_split_threshold,
                        hypernetwork_process_overlap_ratio,
                        hypernetwork_process_focal_crop_face_weight,
                        hypernetwork_process_focal_crop_entropy_weight,
                        hypernetwork_process_focal_crop_edges_weight,
                        hypernetwork_process_focal_crop_debug,
                        hypernetwork_learn_rate,
                        hypernetwork_batch_size,
                        hypernetwork_gradient_step,
                        hypernetwork_training_width,
                        hypernetwork_training_height,
                        hypernetwork_steps,
                        hypernetwork_shuffle_tags,
                        hypernetwork_tag_drop_out,
                        hypernetwork_latent_sampling_method,
                        hypernetwork_create_image_every,
                        hypernetwork_save_embedding_every,
                        hypernetwork_save_image_with_stored_embedding,
                        hypernetwork_preview_from_txt2img,
                        hypernetwork_training_instance_type,
                        hypernetwork_training_instance_count,
                        *txt2img_preview_params
                    ):

                    username = shared.get_webui_username(request)

                    train_args = {
                        'hypernetwork_settings': {
                            'name': new_hypernetwork_name,
                            'enable_sizes': new_hypernetwork_sizes,
                            'overwrite_old': overwrite_old_hypernetwork,
                            'layer_structure': new_hypernetwork_layer_structure,
                            'activation_func': new_hypernetwork_activation_func,
                            'weight_init': new_hypernetwork_initialization_option,
                            'new_hypernetwork_add_layer_norm': new_hypernetwork_add_layer_norm,
                            'new_hypernetwork_use_dropout': new_hypernetwork_use_dropout,
                        },
                        'images_preprocessing_settings': {
                            'process_width': hypernetwork_process_width,
                            'process_height': hypernetwork_process_height,
                            'preprocess_txt_action': hypernetwork_preprocess_txt_action,
                            'process_flip': hypernetwork_process_flip,
                            'process_split': hypernetwork_process_split,
                            'process_caption': hypernetwork_process_caption,
                            'process_caption_deepbooru': hypernetwork_process_caption_deepbooru,
                            'process_split_threshold': hypernetwork_process_split_threshold,
                            'process_overlap_ratio': hypernetwork_process_overlap_ratio,
                            'process_focal_crop': hypernetwork_process_focal_crop,
                            'process_focal_crop_face_weight': hypernetwork_process_focal_crop_face_weight,
                            'process_focal_crop_entropy_weight': hypernetwork_process_focal_crop_entropy_weight,
                            'process_focal_crop_edges_weight': hypernetwork_process_focal_crop_edges_weight,
                            'process_focal_crop_debug': hypernetwork_process_focal_crop_debug
                        },
                        'train_hypernetwork_settings':{
                            'learn_rate': hypernetwork_learn_rate,
                            'batch_size': hypernetwork_batch_size,
                            'gradient_step': hypernetwork_gradient_step,
                            'training_width': hypernetwork_training_width,
                            'training_height': hypernetwork_training_height,
                            'steps': hypernetwork_steps,
                            'shuffle_tags': hypernetwork_shuffle_tags,
                            'tag_drop_out': hypernetwork_tag_drop_out,
                            'latent_sampling_method': hypernetwork_latent_sampling_method,
                            'create_image_every': hypernetwork_create_image_every,
                            'save_embedding_every': hypernetwork_save_embedding_every,
                            'save_image_with_stored_embedding': hypernetwork_save_image_with_stored_embedding,
                            'preview_from_txt2img': hypernetwork_preview_from_txt2img,
                            'txt2img_preview_params': txt2img_preview_params
                        }
                    }
                    
                    sd_model_checkpoint = get_sd_model_checkpoint_from_title(sd_model_checkpoint)
                    hyperparameters = {
                        'train-args': json.dumps(json.dumps(train_args)),
                        'train-task': 'hypernetwork',
                        'ckpt': '/opt/ml/input/data/models/{0}'.format(sd_model_checkpoint),
                        'username': username,
                        'api-endpoint': shared.api_endpoint
                    }
                    
                    inputs = {
                        'images': hypernetwork_images_s3uri,
                        'models': hypernetwork_models_s3uri
                    }
                    
                    data = {
                        'training_job_name': '',
                        'model_algorithm': 'stable-diffusion-webui',
                        'model_hyperparameters': hyperparameters,
                        'industrial_model': shared.industrial_model,
                        'instance_type': hypernetwork_training_instance_type,
                        'instance_count': hypernetwork_training_instance_count,
                        'inputs': inputs
                    }

                    response = requests.post(url=f'{shared.api_endpoint}/train', json=data)
                    if response.status_code == 200:
                        ##begin add train job info by River
                        training_job_url = response.text.replace('\"','')
                        return {
                            ##begin add train job info by River
                            hypernetwork_output: gr.update(value='Submit training job sucessful'),
                            hypernetwork_training_job:gr.update(value=f'Job detail:[{training_job_url}]({training_job_url})')
                            ##end add train job info by River
                        }
                    else:
                        return {
                            hypernetwork_output: gr.update(value=response.text)
                        }
                            
                create_train_embedding.click(
                    fn=sagemaker_train_embedding,
                    inputs=[
                        shared.sd_model_checkpoint_component,
                        new_embedding_name,
                        initialization_text,
                        nvpt,
                        overwrite_old_embedding,
                        embedding_images_s3uri,
                        embedding_models_s3uri,
                        embedding_process_width,
                        embedding_process_height,
                        embedding_preprocess_txt_action,
                        embedding_process_flip,
                        embedding_process_split,
                        embedding_process_focal_crop,
                        embedding_process_caption,
                        embedding_process_caption_deepbooru,
                        embedding_process_split_threshold,
                        embedding_process_overlap_ratio,
                        embedding_process_focal_crop_face_weight,
                        embedding_process_focal_crop_entropy_weight,
                        embedding_process_focal_crop_edges_weight,
                        embedding_process_focal_crop_debug,
                        embedding_learn_rate,
                        embedding_batch_size,
                        embedding_gradient_step,
                        embedding_training_width,
                        embedding_training_height,
                        embedding_steps,
                        embedding_shuffle_tags,
                        embedding_tag_drop_out,
                        embedding_latent_sampling_method,
                        embedding_create_image_every,
                        embedding_save_embedding_every,
                        embedding_save_image_with_stored_embedding,
                        embedding_preview_from_txt2img,
                        embedding_training_instance_type,
                        embedding_training_instance_count,
                        *txt2img_preview_params
                    ],
                    outputs=[embedding_output,embedding_training_job]
                )
                
                create_train_hypernetwork.click(
                    fn=sagemaker_train_hypernetwork,
                    inputs=[
                        shared.sd_model_checkpoint_component,
                        new_hypernetwork_name,
                        new_hypernetwork_sizes,
                        new_hypernetwork_layer_structure,
                        new_hypernetwork_activation_func,
                        new_hypernetwork_initialization_option,
                        new_hypernetwork_add_layer_norm,
                        new_hypernetwork_use_dropout,
                        overwrite_old_hypernetwork,
                        hypernetwork_images_s3uir,
                        hypernetwork_models_s3uri,
                        hypernetwork_process_width,
                        hypernetwork_process_height,
                        hypernetwork_preprocess_txt_action,
                        hypernetwork_process_flip,
                        hypernetwork_process_split,
                        hypernetwork_process_focal_crop,
                        hypernetwork_process_caption,
                        hypernetwork_process_caption_deepbooru,
                        hypernetwork_process_split_threshold,
                        hypernetwork_process_overlap_ratio,
                        hypernetwork_process_focal_crop_face_weight,
                        hypernetwork_process_focal_crop_entropy_weight,
                        hypernetwork_process_focal_crop_edges_weight,
                        hypernetwork_process_focal_crop_debug,
                        hypernetwork_learn_rate,
                        hypernetwork_batch_size,
                        hypernetwork_gradient_step,
                        hypernetwork_training_width,
                        hypernetwork_training_height,
                        hypernetwork_steps,
                        hypernetwork_shuffle_tags,
                        hypernetwork_tag_drop_out,
                        hypernetwork_latent_sampling_method,
                        hypernetwork_create_image_every,
                        hypernetwork_save_embedding_every,
                        hypernetwork_save_image_with_stored_embedding,
                        hypernetwork_preview_from_txt2img,
                        hypernetwork_training_instance_type,
                        hypernetwork_training_instance_count,
                        *txt2img_preview_params
                    ],
                    outputs=[hypernetwork_output,hypernetwork_training_job]
                )

    with gr.Blocks(analytics_enabled=False) as user_interface:
        user_dataframe = gr.Dataframe(
            headers=["Username", "Password", "Options", "Sagemaker Endpoints"],
            row_count=2,
            col_count=(4,"fixed"),
            label="User management (Only available for admin user)",
            interactive=True,
            visible=True,
            datatype=["str","str","str", "str"],
            type="array",
            wrap=True,
        )

        with gr.Row():
            save_userdata_btn = gr.Button(value="Save")

        def save_userdata(user_dataframe, request: gr.Request):
            username = shared.get_webui_username(request)
            if username != 'admin':
                return gr.update()
            items = []
            for user_df in user_dataframe:
                item = {
                    'username': user_df[0],
                    'password': user_df[1],
                    'options': user_df[2],
                    'attributes': {},
                }
                if user_df[3] != '':
                    item['attributes'] = {
                        'sagemaker_endpoints': user_df[3]
                    }
                items.append(item)
            inputs = {
                'action': 'save',
                'items': items
            }
            response = requests.post(url=f'{shared.api_endpoint}/sd/user', json=inputs)
            if response.status_code == 200:
                print(response.text)
                return user_dataframe

        save_userdata_btn.click(
            save_userdata,
            inputs=[user_dataframe],
            outputs=[user_dataframe],
            _js="var if alert('Only admin user can save user data')"
        )

    if cmd_opts.pureui:
        interfaces += [
            (txt2img_interface, "txt2img", "txt2img"),
            (img2img_interface, "img2img", "img2img"),
            (extras_interface, "Extras", "extras"),
            (pnginfo_interface, "PNG Info", "pnginfo"),
            (train_interface, "Train", "ti"),
            (user_interface, "User", "user")
        ]
    else:
        interfaces += [
            (txt2img_interface, "txt2img", "txt2img"),
            (img2img_interface, "img2img", "img2img"),
            (extras_interface, "Extras", "extras"),
            (pnginfo_interface, "PNG Info", "pnginfo"),
            (train_interface, "Train", "ti"),
        ]

    css = ""

    for cssfile in modules.scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue

        with open(cssfile, "r", encoding="utf8") as file:
            css += file.read() + "\n"

    if os.path.exists(os.path.join(script_path, "user.css")):
        with open(os.path.join(script_path, "user.css"), "r", encoding="utf8") as file:
            css += file.read() + "\n"

    if not cmd_opts.no_progressbar_hiding:
        css += css_hide_progressbar

#    interfaces += script_callbacks.ui_tabs_callback()
    interfaces += [(settings_interface, "Settings", "settings")]
    interfaces += [images_history_ui_tab]
    interfaces +=  [(modelmerger_interface,"Checkpoint Merger", "modelmerger")]
    # interfaces += [(imagesviewer_interface,"Images Viewer","imagesviewer")]

    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "Extensions", "extensions")]

    with gr.Blocks(css=css, analytics_enabled=False, title="Stable Diffusion") as demo:
        with gr.Row(elem_id="quicksettings"):
            for i, k, item in quicksettings_list:
                component = create_setting_component(k, is_quicksettings=True)
                component_dict[k] = component

        with gr.Row():
            with gr.Column(scale=5):
                gr.HTML(value='<h1 align="right">Current user : </h1>')
            with gr.Column(scale=1):
                username_state = gr.HTML()
                username_state.change(
                    fn=None,
                    inputs=[],
                    outputs=[username_state, user_dataframe],
                    _js="login"
                )
            with gr.Column(scale=1):
                logout_button = gr.Button(value="Logout")

            def user_logout(request: gr.Request):
                tokens = shared.demo.server_app.tokens
                cookies = shared.get_cookies(request)
                access_token = None
                for cookie in cookies:
                    if cookie.startswith('access-token'):
                        access_token = cookie[len('access-token=') : ]
                        if access_token.startswith('unsecure='):
                            access_token = access_token[len('unsecure=') : ]
                        tokens.pop(access_token)
                        break

            logout_button.click(
                fn=user_logout,
                inputs=[],
                outputs=[],
                _js="restart_reload"
            )

        parameters_copypaste.integrate_settings_paste_fields(component_dict)
        parameters_copypaste.run_bind()
        shared.default_options = shared.opts.data

        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id='tab_' + ifid):
                    interface.render()

        if os.path.exists(os.path.join(script_path, "notification.mp3")):
            audio_notification = gr.Audio(interactive=False, value=os.path.join(script_path, "notification.mp3"), elem_id="audio_notification", visible=False)

        text_settings = gr.Textbox(elem_id="settings_json", value=lambda: opts.dumpjson(), visible=False)
        settings_submit.click(
            fn=wrap_gradio_call(run_settings, extra_outputs=[gr.update()]),
            inputs=components,
            outputs=[text_settings, result],
        )

        for i, k, item in quicksettings_list:
            component = component_dict[k]

            component.change(
                fn=lambda value, k=k: run_settings_single(value, key=k),
                inputs=[component],
                outputs=[component, text_settings],
            )

        component_keys = [k for k in opts.data_labels.keys() if k in component_dict]

        def demo_load(request: gr.Request):
            username = shared.get_webui_username(request)

            inputs = {
                'action': 'load'
            }
            response = requests.post(url=f'{shared.api_endpoint}/sd/user', json=inputs)
            if response.status_code == 200:
                if username == 'admin':
                    items = []
                    for item in json.loads(response.text):
                        items.append([item['username'], item['password'], item['options'] if 'options' in item else '', shared.get_available_sagemaker_endpoints(item)])
                    additional_components = [gr.update(value=f'<h1>{username}</h1>'), gr.update(value=items if items != [] else None), gr.update(), gr.update()]
                else:
                    for item in json.loads(response.text):
                        if item['username'] == username:
                            try:
                                shared.opts.data = json.loads(item['options'])
                                break
                            except Exception as e:
                                print(e)
                    shared.refresh_sagemaker_endpoints(username)
                    shared.refresh_sd_models(username)
                    shared.refresh_checkpoints(shared.opts.sagemaker_endpoint,username)
                    additional_components = [gr.update(value=f'<h1>{username}</h1>'), gr.update(), gr.update(value=shared.opts.sagemaker_endpoint, choices=shared.sagemaker_endpoints), gr.update(value=shared.opts.sd_model_checkpoint, choices=modules.sd_models.checkpoint_tiles())]
            else:
                additional_components = [gr.update(value=username), gr.update(), gr.update(), gr.update()]

            return [getattr(opts, key) for key in component_keys] + additional_components

        demo.load(
            fn=demo_load,
            inputs=[],
            outputs=[component_dict[k] for k in component_keys] + [username_state, user_dataframe, shared.sagemaker_endpoint_component, shared.sd_model_checkpoint_component]
        )

        def modelmerger(primary_model_name, secondary_model_name,
                           tertiary_model_name, interp_method, multiplier,
                           save_as_half, custom_name, checkpoint_format,
                           output_chkpt_s3uri, submit_result,request:gr.Request):
            try:
                results = modules.model_merger.run_modelmerger_remote(primary_model_name, secondary_model_name,
                           tertiary_model_name, interp_method, multiplier,
                           save_as_half, custom_name, checkpoint_format,
                           output_chkpt_s3uri, submit_result,request)
            except Exception as e:
                print("Error loading/saving model file:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                return "Error running model merge"
            return results

        modelmerger_merge.click(
            fn=modelmerger,
            inputs=[
                primary_model_name,
                secondary_model_name,
                tertiary_model_name,
                interp_method,
                interp_amount,
                save_as_half,
                custom_name,
                checkpoint_format,
                merge_output_s3uri,
                submit_result,
            ],
            outputs=[
                submit_result,
            ]
        )

    ui_config_file = cmd_opts.ui_config_file
    ui_settings = {}
    settings_count = len(ui_settings)
    error_loading = False

    try:
        if os.path.exists(ui_config_file):
            with open(ui_config_file, "r", encoding="utf8") as file:
                ui_settings = json.load(file)
    except Exception:
        error_loading = True
        print("Error loading settings:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    def loadsave(path, x):
        def apply_field(obj, field, condition=None, init_field=None):
            key = path + "/" + field

            if getattr(obj, 'custom_script_source', None) is not None:
              key = 'customscript/' + obj.custom_script_source + '/' + key

            if getattr(obj, 'do_not_save_to_config', False):
                return

            saved_value = ui_settings.get(key, None)
            if saved_value is None:
                ui_settings[key] = getattr(obj, field)
            elif condition and not condition(saved_value):
                print(f'Warning: Bad ui setting value: {key}: {saved_value}; Default value "{getattr(obj, field)}" will be used instead.')
            else:
                setattr(obj, field, saved_value)
                if init_field is not None:
                    init_field(saved_value)

        if type(x) in [gr.Slider, gr.Radio, gr.Checkbox, gr.Textbox, gr.Number] and x.visible:
            apply_field(x, 'visible')

        if type(x) == gr.Slider:
            apply_field(x, 'value')
            apply_field(x, 'minimum')
            apply_field(x, 'maximum')
            apply_field(x, 'step')

        if type(x) == gr.Radio:
            apply_field(x, 'value', lambda val: val in x.choices)

        if type(x) == gr.Checkbox:
            apply_field(x, 'value')

        if type(x) == gr.Textbox:
            apply_field(x, 'value')

        if type(x) == gr.Number:
            apply_field(x, 'value')

        # Since there are many dropdowns that shouldn't be saved,
        # we only mark dropdowns that should be saved.
        if type(x) == gr.Dropdown and getattr(x, 'save_to_config', False):
            apply_field(x, 'value', lambda val: val in x.choices, getattr(x, 'init_field', None))
            apply_field(x, 'visible')

    visit(txt2img_interface, loadsave, "txt2img")
    visit(img2img_interface, loadsave, "img2img")
    visit(extras_interface, loadsave, "extras")
    # visit(modelmerger_interface, loadsave, "modelmerger")

    if not error_loading and (not os.path.exists(ui_config_file) or settings_count != len(ui_settings)):
        with open(ui_config_file, "w", encoding="utf8") as file:
            json.dump(ui_settings, file, indent=4)

    return demo


def reload_javascript():
    with open(os.path.join(script_path, "script.js"), "r", encoding="utf8") as jsfile:
        javascript = f'<script>{jsfile.read()}</script>'

    scripts_list = modules.scripts.list_scripts("javascript", ".js")

    for basedir, filename, path in scripts_list:
        with open(path, "r", encoding="utf8") as jsfile:
            javascript += f"\n<!-- {filename} --><script>{jsfile.read()}</script>"

    if cmd_opts.theme is not None:
        javascript += f"\n<script>set_theme('{cmd_opts.theme}');</script>\n"

    javascript += f"\n<script>{localization.localization_js(shared.opts.localization)}</script>"

    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(
            b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


if not hasattr(shared, 'GradioTemplateResponseOriginal'):
    shared.GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse
