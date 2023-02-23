import os
import threading
import time
import importlib
import signal
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from modules.call_queue import wrap_queued_call, queue_lock, wrap_gradio_gpu_call
from modules.paths import script_path

from modules import shared, devices, sd_samplers, upscaler, extensions, localization, ui_tempdir
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
import modules.txt2img
import modules.script_callbacks

import modules.ui
from modules import modelloader
from modules.shared import cmd_opts, opts
import modules.hypernetworks.hypernetwork
import boto3
import traceback
from botocore.exceptions import ClientError
import requests
import io
import json
import uuid
if not cmd_opts.api:
    from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig, sanitize_name
    from extensions.sd_dreambooth_extension.dreambooth.sd_to_diff import extract_checkpoint
    from extensions.sd_dreambooth_extension.dreambooth.dreambooth import start_training_from_config
    from extensions.sd_dreambooth_extension.dreambooth.dreambooth import performance_wizard, training_wizard
    from extensions.sd_dreambooth_extension.dreambooth.db_config import from_file
    from modules import paths
import glob

if cmd_opts.server_name:
    server_name = cmd_opts.server_name
else:
    server_name = "0.0.0.0" if cmd_opts.listen else None


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

    modules.scripts.load_scripts()

    modelloader.load_upscalers()

    modules.sd_vae.refresh_vae_list()
    if not cmd_opts.pureui:
        modules.sd_models.load_model()
        shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()))
    
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_as_default", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: shared.reload_hypernetworks()))
    shared.opts.onchange("sd_hypernetwork_strength", modules.hypernetworks.hypernetwork.apply_strength)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)

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
    if cmd_opts.cors_allow_origins and cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'])
    elif cmd_opts.cors_allow_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_methods=['*'])
    elif cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'])


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

    ckpt_dir = cmd_opts.ckpt_dir
    sd_models_path = os.path.join(shared.models_path, "Stable-diffusion")
    if ckpt_dir is not None:
        sd_models_path = ckpt_dir

    if 'endpoint_name' in os.environ:
        items = []
        api_endpoint = os.environ['api_endpoint']
        endpoint_name = os.environ['endpoint_name']
        for file in os.listdir(sd_models_path):
            if os.path.isfile(os.path.join(sd_models_path, file)) and file.endswith('.ckpt'):
                hash = modules.sd_models.model_hash(os.path.join(sd_models_path, file))
                item = {}
                item['model_name'] = file
                item['config'] = '/opt/ml/code/stable-diffusion-webui/repositories/stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
                item['filename'] = '/opt/ml/code/stable-diffusion-webui/models/Stable-diffusion/{0}'.format(file)
                item['hash'] = hash
                item['title'] = '{0} [{1}]'.format(file, hash)
                item['endpoint_name'] = endpoint_name
                items.append(item)
        inputs = {
            'items': items
        }
        if api_endpoint.startswith('http://') or api_endpoint.startswith('https://'):
            response = requests.post(url=f'{api_endpoint}/sd/models', json=inputs)

    modules.script_callbacks.app_started_callback(None, app)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
        )

    api.launch(server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1", port=cmd_opts.port if cmd_opts.port else 7861)


def webui():
    launch_api = cmd_opts.api
    initialize()

    while 1:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()

        shared.demo = modules.ui.create_ui()

        app, local_url, share_url = shared.demo.launch(
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

        modules.script_callbacks.app_started_callback(shared.demo, app)
        modules.script_callbacks.app_started_callback(shared.demo, app)

        wait_on_server(shared.demo)

        sd_samplers.set_samplers()

        print('Reloading extensions')
        extensions.list_extensions()

        localization.list_localizations(cmd_opts.localizations_dir)

        print('Reloading custom scripts')
        modules.scripts.reload_scripts()
        modelloader.load_upscalers()

        print('Reloading modules: modules.ui')
        importlib.reload(modules.ui)
        print('Refreshing Model List')
        modules.sd_models.list_models()
        print('Restarting Gradio')

def upload_s3files(s3uri, file_path_with_pattern):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]

    s3_resource = boto3.resource('s3')
    s3_bucket = s3_resource.Bucket(bucket)

    try:
        for file_path in glob.glob(file_path_with_pattern):
            file_name = os.path.basename(file_path)
            __s3file = f'{key}{file_name}'
            print(file_path, __s3file)
            s3_bucket.upload_file(file_path, __s3file)
    except ClientError as e:
        print(e)
        return False
    return True

def upload_s3folder(s3uri, file_path):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]

    s3_resource = boto3.resource('s3')
    s3_bucket = s3_resource.Bucket(bucket)

    try:
        for path, _, files in os.walk(file_path):
            for file in files:
                dest_path = path.replace(file_path,"")
                __s3file = f'{key}{dest_path}/{file}'
                __local_file = os.path.join(path, file)
                print(__local_file, __s3file)
                s3_bucket.upload_file(__local_file, __s3file)
    except Exception as e:
        print(e)

if cmd_opts.train:
    def train():
        initialize()

        train_task = cmd_opts.train_task
        train_args = json.loads(cmd_opts.train_args)

        embeddings_s3uri = cmd_opts.embeddings_s3uri
        hypernetwork_s3uri = cmd_opts.hypernetwork_s3uri
        sd_models_s3uri = cmd_opts.sd_models_s3uri
        db_models_s3uri = cmd_opts.db_models_s3uri
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
            gradient_step = train_args['train_embedding_settings']['gradient_step']
            data_root = process_dst
            log_directory = 'textual_inversion'
            training_width = train_args['train_embedding_settings']['training_width']
            training_height = train_args['train_embedding_settings']['training_height']
            steps = train_args['train_embedding_settings']['steps']
            shuffle_tags = train_args['train_embedding_settings']['shuffle_tags']
            tag_drop_out = train_args['train_embedding_settings']['tag_drop_out']
            latent_sampling_method = train_args['train_embedding_settings']['latent_sampling_method']
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
                gradient_step,
                data_root,
                log_directory,
                training_width,
                training_height,
                steps,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                create_image_every,
                save_embedding_every,
                template_file,
                save_image_with_stored_embedding,
                preview_from_txt2img,
                *txt2img_preview_params
            )
            try:
                upload_s3files(embeddings_s3uri, os.path.join(cmd_opts.embeddings_dir, '{0}.pt'.format(train_embedding_name)))
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

            shared.hypernetworks = modules.hypernetworks.hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)
            
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
            gradient_step = train_args['train_hypernetwork_settings']['gradient_step']
            dataset_directory = process_dst
            log_directory = 'textual_inversion'
            training_width = train_args['train_hypernetwork_settings']['training_width']
            training_height = train_args['train_hypernetwork_settings']['training_height']
            steps = train_args['train_hypernetwork_settings']['steps']
            shuffle_tags = train_args['train_hypernetwork_settings']['shuffle_tags']
            tag_drop_out = train_args['train_hypernetwork_settings']['tag_drop_out']
            latent_sampling_method = train_args['train_hypernetwork_settings']['latent_sampling_method']
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
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                steps,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                create_image_every,
                save_hypernetwork_every,
                template_file,
                preview_from_txt2img,
                *txt2img_preview_params
            )
            try:
                upload_s3files(hypernetwork_s3uri, os.path.join(cmd_opts.hypernetwork_dir, '{0}.pt'.format(train_hypernetwork_name)))
            except Exception as e:
                traceback.print_exc()
                print(e)
            opts.data = default_options
        elif train_task == 'dreambooth':
            db_create_new_db_model = train_args['train_dreambooth_settings']['db_create_new_db_model']

            db_lora_model_name = train_args['train_dreambooth_settings']['db_lora_model_name']
            db_lora_weight = train_args['train_dreambooth_settings']['db_lora_weight']
            db_lora_txt_weight = train_args['train_dreambooth_settings']['db_lora_txt_weight']
            db_train_imagic_only = train_args['train_dreambooth_settings']['db_train_imagic_only']
            db_use_subdir = train_args['train_dreambooth_settings']['db_use_subdir']
            db_custom_model_name = train_args['train_dreambooth_settings']['db_custom_model_name']
            db_train_wizard_person = train_args['train_dreambooth_settings']['db_train_wizard_person']
            db_train_wizard_object = train_args['train_dreambooth_settings']['db_train_wizard_object']
            db_performance_wizard = train_args['train_dreambooth_settings']['db_performance_wizard']

            if db_create_new_db_model:
                db_new_model_name = train_args['train_dreambooth_settings']['db_new_model_name']
                db_new_model_src = train_args['train_dreambooth_settings']['db_new_model_src']
                db_new_model_scheduler = train_args['train_dreambooth_settings']['db_new_model_scheduler']
                db_create_from_hub = train_args['train_dreambooth_settings']['db_create_from_hub']
                db_new_model_url = train_args['train_dreambooth_settings']['db_new_model_url']
                db_new_model_token = train_args['train_dreambooth_settings']['db_new_model_token']
                db_new_model_extract_ema = train_args['train_dreambooth_settings']['db_new_model_extract_ema']
                db_model_name, _, db_revision, db_scheduler, db_src, db_has_ema, db_v2, db_resolution = extract_checkpoint(
                        db_new_model_name,
                        db_new_model_src,
                        db_new_model_scheduler,
                        db_create_from_hub,
                        db_new_model_url,
                        db_new_model_token,
                        db_new_model_extract_ema
                    )
                dreambooth_config_id = cmd_opts.dreambooth_config_id
                try:
                    with open(f'/opt/ml/input/data/config/{dreambooth_config_id}.json', 'r') as f:
                        content = f.read()
                except Exception:
                    params = {'module': 'dreambooth_config', 'dreambooth_config_id': dreambooth_config_id}
                    response = requests.get(url=f'{api_endpoint}/sd/models', params=params)
                    if response.status_code == 200:
                        content = response.text
                    else:
                        content = None

                if content:
                    config_dict = json.loads(content)
                    print(db_model_name, db_revision, db_scheduler, db_src, db_has_ema, db_v2, db_resolution)

                    config_dict[0] = db_model_name
                    config_dict[31] = db_revision
                    config_dict[39] = db_scheduler
                    config_dict[40] = db_src
                    config_dict[14] = db_has_ema
                    config_dict[49] = db_v2
                    config_dict[30] = db_resolution

                    db_config = DreamboothConfig(*config_dict)

                    if db_train_wizard_person:
                        _, \
                        max_train_steps, \
                        num_train_epochs, \
                        c1_max_steps, \
                        c1_num_class_images, \
                        c2_max_steps, \
                        c2_num_class_images, \
                        c3_max_steps, \
                        c3_num_class_images = training_wizard(db_config, True)

                        config_dict[22] = int(max_train_steps)
                        config_dict[26] = int(num_train_epochs)
                        config_dict[59] = c1_max_steps
                        config_dict[61] = c1_num_class_images
                        config_dict[77] = c2_max_steps
                        config_dict[79] = c2_num_class_images
                        config_dict[95] = c3_max_steps
                        config_dict[97] = c3_num_class_images
                    if db_train_wizard_object:
                        _, \
                        max_train_steps, \
                        num_train_epochs, \
                        c1_max_steps, \
                        c1_num_class_images, \
                        c2_max_steps, \
                        c2_num_class_images, \
                        c3_max_steps, \
                        c3_num_class_images = training_wizard(db_config, False)

                        config_dict[22] = int(max_train_steps)
                        config_dict[26] = int(num_train_epochs)
                        config_dict[59] = c1_max_steps
                        config_dict[61] = c1_num_class_images
                        config_dict[77] = c2_max_steps
                        config_dict[79] = c2_num_class_images
                        config_dict[95] = c3_max_steps
                        config_dict[97] = c3_num_class_images
                    if db_performance_wizard:
                        _, \
                        attention, \
                        gradient_checkpointing, \
                        mixed_precision, \
                        not_cache_latents, \
                        sample_batch_size, \
                        train_batch_size, \
                        train_text_encoder, \
                        use_8bit_adam, \
                        use_cpu, \
                        use_ema = performance_wizard()

                        config_dict[5] = attention
                        config_dict[12] = gradient_checkpointing
                        config_dict[23] = mixed_precision
                        config_dict[25] = not_cache_latents
                        config_dict[32] = sample_batch_size
                        config_dict[42] = train_batch_size
                        config_dict[43] = train_text_encoder
                        config_dict[44] = use_8bit_adam
                        config_dict[46] = use_cpu
                        config_dict[47] = use_ema
            else:
                db_model_name = train_args['train_dreambooth_settings']['db_model_name']
                db_model_name = sanitize_name(db_model_name)
                db_models_path = cmd_opts.dreambooth_models_path
                if db_models_path == "" or db_models_path is None:
                    db_models_path = os.path.join(shared.models_path, "dreambooth")
                working_dir = os.path.join(db_models_path, db_model_name, "working")
                config_dict = from_file(os.path.join(db_models_path, db_model_name))
                config_dict["pretrained_model_name_or_path"] = working_dir

            db_config = DreamboothConfig(*config_dict)

            ckpt_dir = cmd_opts.ckpt_dir
            sd_models_path = os.path.join(shared.models_path, "Stable-diffusion")
            if ckpt_dir is not None:
                sd_models_path = ckpt_dir

            print(vars(db_config))
            start_training_from_config(
                db_config,
                db_lora_model_name if db_lora_model_name != '' else None,
                db_lora_weight,
                db_lora_txt_weight,
                db_train_imagic_only,
                db_use_subdir,
                db_custom_model_name
            )

            try:
                cmd_dreambooth_models_path = cmd_opts.dreambooth_models_path
            except:
                cmd_dreambooth_models_path = None

            db_model_dir = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
            db_model_dir = os.path.join(db_model_dir, "dreambooth")

            try:
                print('Uploading SD Models...')                
                upload_s3files(
                    sd_models_s3uri,
                    os.path.join(sd_models_path, f'{sd_models_path}/{db_model_name}_*.yaml')
                )                
                upload_s3files(
                    sd_models_s3uri,
                    os.path.join(sd_models_path, f'{sd_models_path}/{db_model_name}_*.ckpt')
                )
                print('Uploading DB Models...')
                upload_s3folder(
                    f'{db_models_s3uri}{db_model_name}',
                    os.path.join(db_model_dir, db_model_name)
                )
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
