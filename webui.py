import os
import threading
import time
import importlib
import signal
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

import requests
import json
import time
from PIL import Image
import base64
import io

from modules.paths import script_path

from modules import devices, sd_samplers, upscaler, extensions, localization
import modules.codeformer_model as codeformer
import modules.extras
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.sd_vae
import modules.shared as shared
import modules.txt2img
import modules.script_callbacks

import modules.ui
from modules import modelloader
from modules.shared import cmd_opts,  opts
import modules.hypernetworks.hypernetwork
import modules.textual_inversion.textual_inversion

import uuid

from PIL import Image, ImageOps, ImageChops

queue_lock = threading.Lock()
server_name = "0.0.0.0" if cmd_opts.listen else cmd_opts.server_name

import boto3
import traceback
from botocore.exceptions import ClientError

def wrap_queued_call(func):
    def f(*args, **kwargs):
        
        with queue_lock:
            res = func(*args, **kwargs)
        return res
    
    return f

def wrap_gradio_gpu_call(func, extra_outputs=None):    
    def encode_image_to_base64(image):
        if isinstance(image, bytes):
            encoded_string = base64.b64encode(image)
        else:
            image.tobytes("hex", "rgb")
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            encoded_string = base64.b64encode(image_bytes.getvalue())

        base64_str = str(encoded_string, "utf-8")
        mimetype = 'image/png'
        image_encoded_in_base64 = (
            "data:"
            + (mimetype if mimetype is not None else "")
            + ";base64,"
            + base64_str
        )
        return image_encoded_in_base64

    def handle_sagemaker_inference_async(response):
        s3uri = response.text
        params = {'s3uri': s3uri}
        start = time.time()
        while True:
            if shared.state.interrupted or shared.state.skipped:
                shared.job_count = 0
                return None
            
            response = requests.get(url=f'{shared.api_endpoint}/s3', params = params)
            text = json.loads(response.text)

            if text['count'] > 0:
                break
            else:
                time.sleep(1)

        httpuri = text['payload'][0]['httpuri']
        response = requests.get(url=httpuri)
        processed = json.loads(response.text)
        print(f"Time taken: {time.time() - start}s")

        return processed

    def sagemaker_inference(task, infer, *args, **kwargs):
        infer = 'async'
        if task == 'text-to-image' or task == 'image-to-image':
            if task == 'text-to-image':
                script_args = []
                for i in range(23, len(args)):
                    script_args.append(args[i])

                prompt = args[0]
                negative_prompt = args[1]
                styles = [args[2], args[3]]
                steps = args[4]
                sampler_index = sd_samplers.samplers[args[5]].name
                restore_faces = args[6]
                tiling = args[7]
                n_iter = args[8]
                batch_size = args[9]
                cfg_scale = args[10]
                seed = args[11]
                subseed = args[12]
                subseed_strength = args[13]
                seed_resize_from_h = args[14]
                seed_resize_from_w = args[15]
                seed_enable_extras = args[16]
                height = args[17]
                width = args[18]
                enable_hr = args[19]
                denoising_strength = args[20]
                firstphase_width = args[21]
                firstphase_height = args[22]

                payload = {
                    "enable_hr": enable_hr,
                    "denoising_strength": denoising_strength,
                    "firstphase_width": firstphase_width,
                    "firstphase_height": firstphase_height,
                    "prompt": prompt,
                    "styles": styles,
                    "seed": seed,
                    "subseed": subseed,
                    "subseed_strength": subseed_strength,
                    "seed_resize_from_h": seed_resize_from_h,
                    "seed_resize_from_w": seed_resize_from_w,
                    "sampler_index": sampler_index,
                    "batch_size": batch_size,
                    "n_iter": n_iter,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "width": width,
                    "height": height,
                    "restore_faces": restore_faces,
                    "tiling": tiling,
                    "negative_prompt": negative_prompt,
                    "eta": opts.eta_ddim if sd_samplers.samplers[args[5]].name == 'DDIM' or sd_samplers.samplers[args[5]].name == 'PLMS' else opts.eta_ancestral,
                    "s_churn": opts.s_churn,
                    "s_tmax": None,
                    "s_tmin": opts.s_tmin,
                    "s_noise": opts.s_noise,
                    "override_settings": {},
                    "script_args": json.dumps(script_args),
                }
                inputs = {
                    'task': task,
                    'txt2img_payload': payload,
                    'username': shared.username
                }
            else:
                mode = args[0]
                prompt = args[1]
                negative_prompt = args[2]
                styles = [args[3], args[4]]
                init_img = args[5]
                init_img_with_mask = args[6]
                init_img_inpaint = args[7]
                init_mask_inpaint = args[8]
                mask_mode = args[9]
                steps = args[10]
                sampler_index = sd_samplers.samplers[args[11]].name
                mask_blur = args[12]
                inpainting_fill = args[13]
                restore_faces = args[14]
                tiling = args[15]
                n_iter = args[16]
                batch_size = args[17]
                cfg_scale = args[18]
                denoising_strength = args[19]
                seed = args[20]
                subseed = args[21]
                subseed_strength = args[22]
                seed_resize_from_h = args[23]
                seed_resize_from_w = args[24]
                seed_enable_extras = args[25]
                height = args[26]
                width = args[27]
                resize_mode = args[28]
                inpaint_full_res = args[29]
                inpaint_full_res_padding = args[30]
                inpainting_mask_invert = args[31]
                img2img_batch_input_dir = args[32]
                img2img_batch_output_dir = args[33]

                script_args = []
                for i in range(34, len(args)):
                    script_args.append(args[i])

                if mode == 1:
                    # Drawn mask
                    if mask_mode == 0:
                        image = init_img_with_mask['image']
                        mask = init_img_with_mask['mask']
                        alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
                        mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
                        image = image.convert('RGB')
                    # Uploaded mask
                    else:
                        image = init_img_inpaint
                        mask = init_mask_inpaint
                # No mask
                else:
                    image = init_img
                    mask = None

                # Use the EXIF orientation of photos taken by smartphones.
                if image is not None:
                    image = ImageOps.exif_transpose(image)

                assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

                image_encoded_in_base64 = encode_image_to_base64(image)
                mask_encoded_in_base64 = encode_image_to_base64(mask) if mask else None

                if init_img_with_mask:
                    image = init_img_with_mask['image']
                    image_encoded_in_base64 = encode_image_to_base64(image)
                    mask_encoded_in_base64 = encode_image_to_base64(mask)
                    init_img_with_mask['image'] = image_encoded_in_base64
                    init_img_with_mask['mask'] = mask_encoded_in_base64

                payload = {
                    "init_images": [image_encoded_in_base64],
                    "resize_mode": resize_mode,
                    "denoising_strength": denoising_strength,
                    "mask": mask_encoded_in_base64,
                    "mask_blur": mask_blur,
                    "inpainting_fill": inpainting_fill,
                    "inpaint_full_res": inpaint_full_res,
                    "inpaint_full_res_padding": inpaint_full_res_padding,
                    "inpainting_mask_invert": inpainting_mask_invert,
                    "prompt": prompt,
                    "styles": styles,
                    "seed": seed,
                    "subseed": subseed,
                    "subseed_strength": subseed_strength,
                    "seed_resize_from_h": seed_resize_from_h,
                    "seed_resize_from_w": seed_resize_from_w,
                    "sampler_index": sampler_index,
                    "batch_size": batch_size,
                    "n_iter": n_iter,
                    "steps": steps,
                    "cfg_scale": args[18],
                    "width": width,
                    "height": height,
                    "restore_faces": restore_faces,
                    "tiling": tiling,
                    "negative_prompt": negative_prompt,
                    "sampler_index": sampler_index,
                    "eta": opts.eta_ddim if sd_samplers.samplers[args[11]].name == 'DDIM' or sd_samplers.samplers[args[11]].name == 'PLMS' else opts.eta_ancestral,
                    "s_churn": opts.s_churn,
                    "s_tmax": None,
                    "s_tmin": opts.s_tmin,
                    "s_noise": opts.s_noise,
                    "override_settings": {},
                    "include_init_images": False,
                    "script_args": json.dumps(script_args)
                }
                inputs = {
                    'task': task,
                    'img2img_payload': payload,
                    'username': shared.username
                }

            params = {
                'endpoint_name': shared.opts.sagemaker_endpoint
            }

            response = requests.post(url=f'{shared.api_endpoint}/inference', params=params, json=inputs)
            if infer == 'async':
                processed = handle_sagemaker_inference_async(response)
            else:
                processed = json.loads(response.text)

            if processed == None:
                return [], "", ""
                
            images = []
            for image in processed['images']:
                images.append(Image.open(io.BytesIO(base64.b64decode(image))))
            parameters = processed['parameters']
            info = processed['info']

            return images, json.dumps(payload), modules.ui.plaintext_to_html(info)
        else:
            extras_mode = args[0]
            resize_mode = args[1]
            image = args[2]
            image_folder = args[3]
            input_dir = args[4]
            output_dir = args[5]
            show_extras_results = args[6]
            gfpgan_visibility = args[7]
            codeformer_visibility = args[8]
            codeformer_weight = args[9]
            upscaling_resize = args[10]
            upscaling_resize_w = args[11]
            upscaling_resize_h = args[12]
            upscaling_crop = args[13]
            extras_upscaler_1 = shared.sd_upscalers[args[14]].name
            extras_upscaler_2 = shared.sd_upscalers[args[15]].name
            extras_upscaler_2_visibility = args[16]
            upscale_first = args[17]

            if extras_mode == 0:
                image_encoded_in_base64 = encode_image_to_base64(image)

                payload = {
                  "resize_mode": resize_mode,
                  "show_extras_results": show_extras_results if show_extras_results else True,
                  "gfpgan_visibility": gfpgan_visibility,
                  "codeformer_visibility": codeformer_visibility,
                  "codeformer_weight": codeformer_weight,
                  "upscaling_resize": upscaling_resize,
                  "upscaling_resize_w": upscaling_resize_w,
                  "upscaling_resize_h": upscaling_resize_h,
                  "upscaling_crop": upscaling_crop,
                  "upscaler_1": extras_upscaler_1,
                  "upscaler_2": extras_upscaler_2,
                  "extras_upscaler_2_visibility": extras_upscaler_2_visibility,
                  "upscale_first": upscale_first,
                  "image": image_encoded_in_base64
                }
                task = 'extras-single-image'
                inputs = {
                    'task': task,
                    'extras_single_payload': payload,
                    'username': shared.username
                }
            else:
                imageList = []
                for img in image_folder:
                    image_encoded_in_base64 = encode_image_to_base64(Image.open(img))
                    imageList.append(
                        {
                            'data': image_encoded_in_base64,
                            'name': img.name
                        }
                    )
                payload = {
                  "resize_mode": resize_mode,
                  "show_extras_results": show_extras_results if show_extras_results else True,
                  "gfpgan_visibility": gfpgan_visibility,
                  "codeformer_visibility": codeformer_visibility,
                  "codeformer_weight": codeformer_weight,
                  "upscaling_resize": upscaling_resize,
                  "upscaling_resize_w": upscaling_resize_w,
                  "upscaling_resize_h": upscaling_resize_h,
                  "upscaling_crop": upscaling_crop,
                  "upscaler_1": extras_upscaler_1,
                  "upscaler_2": extras_upscaler_2,
                  "extras_upscaler_2_visibility": extras_upscaler_2_visibility,
                  "upscale_first": upscale_first,
                  "imageList": imageList
                }
                task = 'extras-batch-images'
                inputs = {
                    'task': task,
                    'extras_batch_payload': payload,
                    'username': shared.username
                }

            params = {
                'endpoint_name': shared.opts.sagemaker_endpoint
            }
            response = requests.post(url=f'{shared.api_endpoint}/inference', params=params, json=inputs)
            if infer == 'async':
                processed = handle_sagemaker_inference_async(response)
            else:
                processed = json.loads(response.text)

            if task == 'extras-single-image':
                images = [Image.open(io.BytesIO(base64.b64decode(processed['image'])))]
            else:
                images = []
                for image in processed['images']:
                    images.append(Image.open(io.BytesIO(base64.b64decode(image))))
            info = processed['html_info']
            return images, modules.ui.plaintext_to_html(info), ''

    def f(*args, **kwargs):
        if cmd_opts.pureui and func == modules.txt2img.txt2img:
            res = sagemaker_inference('text-to-image', 'sync', *args, **kwargs)
        elif cmd_opts.pureui and func == modules.img2img.img2img:
            res = sagemaker_inference('image-to-image', 'sync', *args, **kwargs)
        elif cmd_opts.pureui and func == modules.extras.run_extras:
            res = sagemaker_inference('extras', 'sync', *args, **kwargs)
        else:
            shared.state.begin()

            with queue_lock:
                res = func(*args, **kwargs)

            shared.state.end()

        return res

    return modules.ui.wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True)


def initialize():
    extensions.list_extensions()
    localization.list_localizations(cmd_opts.localizations_dir)

    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        modules.scripts.load_scripts()
        return

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    shared.face_restorers.append(modules.face_restoration.FaceRestoration())
    modelloader.load_upscalers()

    modules.scripts.load_scripts()

    modules.sd_vae.refresh_vae_list()

    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)

    if not cmd_opts.pureui:
        modules.sd_models.load_model()
        shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()))
        shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: modules.hypernetworks.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)))
        shared.opts.onchange("sd_hypernetwork_strength", modules.hypernetworks.hypernetwork.apply_strength)

    if cmd_opts.tls_keyfile is not None and cmd_opts.tls_keyfile is not None:

        try:
            if not os.path.exists(cmd_opts.tls_keyfile):
                print("Invalid path to TLS keyfile given")
            if not os.path.exists(cmd_opts.tls_certfile):
                print(f"Invalid path to TLS certfile: '{cmd_opts.tls_certfile}'")
        except TypeError:
            cmd_opts.tls_keyfile = cmd_opts.tls_certfile = None
            print("TLS setup invalid, running webui without TLS")
        else:
            print("Running with TLS")

    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)


def setup_cors(app):
    if cmd_opts.cors_allow_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_methods=['*'])


def create_api(app):
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api


def wait_on_server(demo=None):
    while 1:
        time.sleep(0.5)
        if shared.state.need_restart:
            shared.state.need_restart = False
            time.sleep(0.5)
            demo.close()
            time.sleep(0.5)
            break


def api_only():
    initialize()

    app = FastAPI()
    setup_cors(app)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    api = create_api(app)

    modules.script_callbacks.app_started_callback(None, app)

    api.launch(server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1", port=cmd_opts.port if cmd_opts.port else 7861)


def webui():
    launch_api = cmd_opts.api
    initialize()

    while 1:
        demo = modules.ui.create_ui(wrap_gradio_gpu_call=wrap_gradio_gpu_call)

        app, local_url, share_url = demo.launch(
            share=cmd_opts.share,
            server_name=server_name,
            server_port=cmd_opts.port,
            ssl_keyfile=cmd_opts.tls_keyfile,
            ssl_certfile=cmd_opts.tls_certfile,
            debug=cmd_opts.gradio_debug,
            auth=[tuple(cred.split(':')) for cred in cmd_opts.gradio_auth.strip('"').split(',')] if cmd_opts.gradio_auth else None,
            inbrowser=cmd_opts.autolaunch,
            prevent_thread_lock=True
        )
        # after initial launch, disable --autolaunch for subsequent restarts
        cmd_opts.autolaunch = False

        # gradio uses a very open CORS policy via app.user_middleware, which makes it possible for
        # an attacker to trick the user into opening a malicious HTML page, which makes a request to the
        # running web ui and do whatever the attcker wants, including installing an extension and
        # runnnig its code. We disable this here. Suggested by RyotaK.
        app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']

        setup_cors(app)

        app.add_middleware(GZipMiddleware, minimum_size=1000)

        if launch_api:
            create_api(app)

        modules.script_callbacks.app_started_callback(demo, app)

        wait_on_server(demo)

        sd_samplers.set_samplers()

        print('Reloading extensions')
        extensions.list_extensions()

        localization.list_localizations(cmd_opts.localizations_dir)

        print('Reloading custom scripts')
        modules.scripts.reload_scripts()
        print('Reloading modules: modules.ui')
        importlib.reload(modules.ui)
        print('Refreshing Model List')
        modules.sd_models.list_models()
        print('Restarting Gradio')

def upload_s3file(s3uri, file_path, file_name):
    s3_client = boto3.client('s3', region_name = cmd_opts.region_name)

    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]

    binary = io.BytesIO(open(file_path, 'rb').read())
    key = key + file_name
    try:
        s3_client.upload_fileobj(binary, bucket, key)
    except ClientError as e:
        print(e)
        return False
    return True

def train():
    initialize()

    train_task = cmd_opts.train_task
    train_args = json.loads(cmd_opts.train_args)

    embeddings_s3uri = cmd_opts.embeddings_s3uri
    hypernetwork_s3uri = cmd_opts.hypernetwork_s3uri
    api_endpoint = cmd_opts.api_endpoint   
    username = cmd_opts.username

    default_options = opts.data
    if username != '':
        inputs = {
            'action': 'get',
            'username': username
        }
        response = requests.post(url=f'{api_endpoint}/sd/user', json=inputs)
        if response.status_code == 200 and response.text != '':
            opts.data = json.loads(response.text)
            modules.sd_models.load_model()

    if train_task == 'embedding':
        name = train_args['embedding_settings']['name']
        nvpt = train_args['embedding_settings']['nvpt']
        overwrite_old = train_args['embedding_settings']['overwrite_old']
        initialization_text = train_args['embedding_settings']['initialization_text']
        modules.textual_inversion.textual_inversion.create_embedding(
            name, 
            nvpt, 
            overwrite_old, 
            init_text=initialization_text
        )
        if not cmd_opts.pureui:
            modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
        process_src = '/opt/ml/input/data/images'
        process_dst = str(uuid.uuid4())
        process_width = train_args['images_preprocessing_settings']['process_width']
        process_height = train_args['images_preprocessing_settings']['process_height']
        preprocess_txt_action = train_args['images_preprocessing_settings']['preprocess_txt_action']
        process_flip = train_args['images_preprocessing_settings']['process_flip']
        process_split = train_args['images_preprocessing_settings']['process_split']
        process_caption = train_args['images_preprocessing_settings']['process_caption']    
        process_caption_deepbooru = train_args['images_preprocessing_settings']['process_caption_deepbooru']    
        process_split_threshold = train_args['images_preprocessing_settings']['process_split_threshold']
        process_overlap_ratio = train_args['images_preprocessing_settings']['process_overlap_ratio']
        process_focal_crop = train_args['images_preprocessing_settings']['process_focal_crop']
        process_focal_crop_face_weight = train_args['images_preprocessing_settings']['process_focal_crop_face_weight']    
        process_focal_crop_entropy_weight = train_args['images_preprocessing_settings']['process_focal_crop_entropy_weight']    
        process_focal_crop_edges_weight = train_args['images_preprocessing_settings']['process_focal_crop_debug']
        process_focal_crop_debug = train_args['images_preprocessing_settings']['process_focal_crop_debug']
        modules.textual_inversion.preprocess.preprocess(
            process_src,
            process_dst,
            process_width,
            process_height,
            preprocess_txt_action,
            process_flip,
            process_split,
            process_caption,
            process_caption_deepbooru,
            process_split_threshold,
            process_overlap_ratio,
            process_focal_crop,
            process_focal_crop_face_weight,
            process_focal_crop_entropy_weight,
            process_focal_crop_edges_weight,
            process_focal_crop_debug,
        )
        train_embedding_name = name
        learn_rate = train_args['train_embedding_settings']['learn_rate']
        batch_size = train_args['train_embedding_settings']['batch_size']
        data_root = process_dst
        log_directory = 'textual_inversion'
        training_width = train_args['train_embedding_settings']['training_width']
        training_height = train_args['train_embedding_settings']['training_height']
        steps = train_args['train_embedding_settings']['steps']
        create_image_every = train_args['train_embedding_settings']['create_image_every']
        save_embedding_every = train_args['train_embedding_settings']['save_embedding_every']
        template_file = os.path.join(script_path, "textual_inversion_templates", "style_filewords.txt")
        save_image_with_stored_embedding = train_args['train_embedding_settings']['save_image_with_stored_embedding']
        preview_from_txt2img = train_args['train_embedding_settings']['preview_from_txt2img']
        txt2img_preview_params = train_args['train_embedding_settings']['txt2img_preview_params']
        _, filename = modules.textual_inversion.textual_inversion.train_embedding(
            train_embedding_name,
            learn_rate,
            batch_size,
            data_root,
            log_directory,
            training_width,
            training_height,
            steps,
            create_image_every,
            save_embedding_every,
            template_file,
            save_image_with_stored_embedding,
            preview_from_txt2img,
            *txt2img_preview_params
        )
        try:
            upload_s3file(embeddings_s3uri, os.path.join(cmd_opts.embeddings_dir, '{0}.pt'.format(train_embedding_name)), '{0}.pt'.format(train_embedding_name))
        except Exception as e:
            traceback.print_exc()
            print(e)
        opts.data = default_options
    elif train_task == 'hypernetwork':
        name = train_args['hypernetwork_settings']['name']
        enable_sizes = train_args['hypernetwork_settings']['enable_sizes']
        overwrite_old = train_args['hypernetwork_settings']['overwrite_old']
        layer_structure = train_args['hypernetwork_settings']['layer_structure'] if 'layer_structure' in train_args['hypernetwork_settings'] else None
        activation_func = train_args['hypernetwork_settings']['activation_func'] if 'activation_func' in train_args['hypernetwork_settings'] else None
        weight_init = train_args['hypernetwork_settings']['weight_init'] if 'weight_init' in train_args['hypernetwork_settings'] else None
        add_layer_norm = train_args['hypernetwork_settings']['add_layer_norm'] if 'add_layer_norm' in train_args['hypernetwork_settings'] else False
        use_dropout = train_args['hypernetwork_settings']['use_dropout'] if 'use_dropout' in train_args['hypernetwork_settings'] else False
        
        name = "".join( x for x in name if (x.isalnum() or x in "._- "))

        fn = os.path.join(cmd_opts.hypernetwork_dir, f"{name}.pt")
        if not overwrite_old:
            assert not os.path.exists(fn), f"file {fn} already exists"

        if type(layer_structure) == str:
            layer_structure = [float(x.strip()) for x in layer_structure.split(",")]

        hypernet = modules.hypernetworks.hypernetwork.Hypernetwork(
            name=name,
            enable_sizes=[int(x) for x in enable_sizes],
            layer_structure=layer_structure,
            activation_func=activation_func,
            weight_init=weight_init,
            add_layer_norm=add_layer_norm,
            use_dropout=use_dropout,
        )
        hypernet.save(fn)

        process_src = '/opt/ml/input/data/images'
        process_dst = str(uuid.uuid4())
        process_width = train_args['images_preprocessing_settings']['process_width']
        process_height = train_args['images_preprocessing_settings']['process_height']
        preprocess_txt_action = train_args['images_preprocessing_settings']['preprocess_txt_action']
        process_flip = train_args['images_preprocessing_settings']['process_flip']
        process_split = train_args['images_preprocessing_settings']['process_split']
        process_caption = train_args['images_preprocessing_settings']['process_caption']    
        process_caption_deepbooru = train_args['images_preprocessing_settings']['process_caption_deepbooru']    
        process_split_threshold = train_args['images_preprocessing_settings']['process_split_threshold']
        process_overlap_ratio = train_args['images_preprocessing_settings']['process_overlap_ratio']
        process_focal_crop = train_args['images_preprocessing_settings']['process_focal_crop']
        process_focal_crop_face_weight = train_args['images_preprocessing_settings']['process_focal_crop_face_weight']    
        process_focal_crop_entropy_weight = train_args['images_preprocessing_settings']['process_focal_crop_entropy_weight']    
        process_focal_crop_edges_weight = train_args['images_preprocessing_settings']['process_focal_crop_debug']
        process_focal_crop_debug = train_args['images_preprocessing_settings']['process_focal_crop_debug']
        modules.textual_inversion.preprocess.preprocess(
            process_src,
            process_dst,
            process_width,
            process_height,
            preprocess_txt_action,
            process_flip,
            process_split,
            process_caption,
            process_caption_deepbooru,
            process_split_threshold,
            process_overlap_ratio,
            process_focal_crop,
            process_focal_crop_face_weight,
            process_focal_crop_entropy_weight,
            process_focal_crop_edges_weight,
            process_focal_crop_debug,
        )
        train_hypernetwork_name = name
        learn_rate = train_args['train_hypernetwork_settings']['learn_rate']
        batch_size = train_args['train_hypernetwork_settings']['batch_size']
        dataset_directory = process_dst
        log_directory = 'textual_inversion'
        training_width = train_args['train_hypernetwork_settings']['training_width']
        training_height = train_args['train_hypernetwork_settings']['training_height']
        steps = train_args['train_hypernetwork_settings']['steps']
        create_image_every = train_args['train_hypernetwork_settings']['create_image_every']
        save_hypernetwork_every = train_args['train_hypernetwork_settings']['save_embedding_every']
        template_file = os.path.join(script_path, "textual_inversion_templates", "style_filewords.txt")
        save_image_with_stored_embedding = train_args['train_hypernetwork_settings']['save_image_with_stored_embedding']
        preview_from_txt2img = train_args['train_hypernetwork_settings']['preview_from_txt2img']
        txt2img_preview_params = train_args['train_hypernetwork_settings']['txt2img_preview_params']        
        _, filename = modules.hypernetworks.hypernetwork.train_hypernetwork(
            train_hypernetwork_name,
            learn_rate,
            batch_size,
            dataset_directory,
            log_directory,
            training_width,
            training_height,
            steps,
            create_image_every,
            save_hypernetwork_every,
            template_file,
            preview_from_txt2img,
            *txt2img_preview_params
        )
        try:
            upload_s3file(hypernetwork_s3uri, os.path.join(cmd_opts.hypernetwork_dir, '{0}.pt'.format(train_hypernetwork_name)), '{0}.pt'.format(train_hypernetwork_name))
        except Exception as e:
            traceback.print_exc()
            print(e)
        opts.data = default_options
    else:
        print('Incorrect training task')
        exit(-1)

if __name__ == "__main__":
    if cmd_opts.train:
        train()
    elif cmd_opts.nowebui:
        api_only()
    else:
        webui()
