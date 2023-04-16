import os
import shutil
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

import psutil

from modules import extra_networks, ui_extra_networks_checkpoints
from modules import extra_networks_hypernet, ui_extra_networks_hypernets, ui_extra_networks_textual_inversion

from modules.call_queue import wrap_queued_call, queue_lock
from modules.paths import script_path
from collections import OrderedDict

from modules import shared, sd_samplers, upscaler, extensions, localization, ui_tempdir, ui_extra_networks
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
from modules.shared import cmd_opts, opts, sd_model,syncLock,de_register_model
import modules.hypernetworks.hypernetwork
import boto3
import threading
import time

import traceback
from botocore.exceptions import ClientError
import requests
import json
import uuid

from huggingface_hub import hf_hub_download
import shutil
import glob

if not cmd_opts.api:
    from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig
    from extensions.sd_dreambooth_extension.scripts.dreambooth import start_training_from_config, create_model
    from extensions.sd_dreambooth_extension.scripts.dreambooth import performance_wizard, training_wizard
    from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept
    from modules import paths

import requests
cache = dict()
s3_client = boto3.client('s3')
s3_resource= boto3.resource('s3')

def s3_download(s3uri, path):
    global cache

    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]

    s3_bucket = s3_resource.Bucket(bucket)
    objs = list(s3_bucket.objects.filter(Prefix=key))

    if os.path.isfile('cache'):
        cache = json.load(open('cache', 'r'))

    for obj in objs:
        response = s3_client.head_object(
            Bucket = bucket,
            Key =  obj.key
        )
        obj_key = 's3://{0}/{1}'.format(bucket, obj.key)
        if obj_key not in  cache or cache[obj_key] != response['ETag']:
            filename = obj.key[obj.key.rfind('/') + 1 : ]

            s3_client.download_file(bucket, obj.key, os.path.join(path, filename))
            cache[obj_key] = response['ETag']

    json.dump(cache, open('cache', 'w'))

def http_download(httpuri, path):
    with requests.get(httpuri, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

if cmd_opts.server_name:
    server_name = cmd_opts.server_name
else:
    server_name = "0.0.0.0" if cmd_opts.listen else None

FREESPACE = 20

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

    ui_extra_networks.intialize()
    ui_extra_networks.register_page(ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion())
    ui_extra_networks.register_page(ui_extra_networks_hypernets.ExtraNetworksPageHypernetworks())
    ui_extra_networks.register_page(ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints())

    extra_networks.initialize()
    extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())

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

    modules.script_callbacks.app_started_callback(None, app)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
        )

    api.launch(server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1", port=cmd_opts.port if cmd_opts.port else 7861)

def user_auth(username, password):
    inputs = {
        'username': username,
        'password': password
    }
    api_endpoint = os.environ['api_endpoint']
    response = requests.post(url=f'{api_endpoint}/sd/login', json=inputs)
    return response.status_code == 200

def get_bucket_and_key(s3uri):
    pos = s3uri.find('/', 5)
    bucket = s3uri[5 : pos]
    key = s3uri[pos + 1 : ]
    return bucket, key

def get_models(path, extensions):
    candidates = []
    models = []
    for extension in extensions:
        candidates = candidates + glob.glob(os.path.join(path, f'**/{extension}'), recursive=True)

    for filename in sorted(candidates, key=str.lower):
        if os.path.isdir(filename):
            continue
        models.append(filename)
    return models

def check_space_s3_download(s3_client,bucket_name,s3_folder,local_folder,file,size,mode):
    src = s3_folder + '/' + file
    dist =  os.path.join(local_folder, file)
    os.makedirs(os.path.dirname(dist), exist_ok=True)
    # Get disk usage statistics
    disk_usage = psutil.disk_usage('/tmp')
    freespace = disk_usage.free/(1024**3)
    print(f"Total space: {disk_usage.total/(1024**3)}, Used space: {disk_usage.used/(1024**3)}, Free space: {freespace}")
    if freespace - size >= FREESPACE:
        try:
            s3_client.download_file(bucket_name, src, dist)
            #init ref cnt to 0, when the model file first time download
            hash = modules.sd_models.model_hash(dist)
            if mode == 'sd' :
                shared.sd_models_Ref.add_models_ref('{0} [{1}]'.format(file, hash))
            elif mode == 'cn':
                shared.cn_models_Ref.add_models_ref('{0} [{1}]'.format(os.path.splitext(file)[0], hash))
            elif mode == 'lora':
                shared.lora_models_Ref.add_models_ref('{0} [{1}]'.format(os.path.splitext(file)[0], hash))
            print(f'download_file success:from {bucket_name}/{src} to {dist}')
        except Exception as e:
            print(f'download_file error: from {bucket_name}/{src} to {dist}')
            print(f"An error occurred: {e}") 
            return False
        return True
    else:
        return False

def free_local_disk(local_folder,size,mode):
    disk_usage = psutil.disk_usage('/tmp')
    freespace = disk_usage.free/(1024**3)
    if freespace - size >= FREESPACE:
        return
    models_Ref = None
    if mode == 'sd' :
        models_Ref = shared.sd_models_Ref
    elif mode == 'cn':
        models_Ref = shared.cn_models_Ref
    elif mode == 'lora':
        models_Ref = shared.lora_models_Ref
    model_name,ref_cnt  = models_Ref.get_least_ref_model()
    print (f'shared.{mode}_models_Ref:{models_Ref.get_models_ref_dict()} -- model_name:{model_name}')
    if model_name and ref_cnt:
        filename = model_name[:model_name.rfind("[")]
        os.remove(os.path.join(local_folder, filename))
        disk_usage = psutil.disk_usage('/tmp')
        freespace = disk_usage.free/(1024**3)
        print(f"Remove file: {os.path.join(local_folder, filename)} now left space:{freespace}") 
        de_register_model(filename,mode)
    else:
        ## if ref_cnt == 0, then delete the oldest zero_ref one
        zero_ref_models = set([model[:model.rfind(" [")] for model, count in models_Ref.get_models_ref_dict().items() if count == 0])
        local_files = set(os.listdir(local_folder))
        # join with local
        files = [(os.path.join(local_folder, file), os.path.getctime(os.path.join(local_folder, file))) for file in zero_ref_models.intersection(local_files)]
        if len(files) == 0:
            print(f"No files to remove in folder: {local_folder}, please remove some files in S3 bucket") 
            return
        files.sort(key=lambda x: x[1])
        oldest_file = files[0][0]
        os.remove(oldest_file)
        disk_usage = psutil.disk_usage('/tmp')
        freespace = disk_usage.free/(1024**3)
        print(f"Remove file: {oldest_file} now left space:{freespace}") 
        filename = os.path.basename(oldest_file)
        de_register_model(filename,mode)

def list_s3_objects(s3_client,bucket_name, prefix=''):
    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    # iterate over pages
    for page in page_iterator:
        # loop through objects in page
        if 'Contents' in page:
            for obj in page['Contents']:
                # add object to list
                objects.append(obj)
        # if there are more pages to fetch, continue
        if 'NextContinuationToken' in page:
            page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix,
                                                ContinuationToken=page['NextContinuationToken'])
    return objects


def initial_s3_download(s3_folder, local_folder,cache_dir,mode):
    # Create tmp folders 
    os.makedirs(os.path.dirname(local_folder), exist_ok=True)
    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    print(f'create dir: {os.path.dirname(local_folder)}')
    print(f'create dir: {os.path.dirname(cache_dir)}')
    s3_file_name = os.path.join(cache_dir,f's3_files_{mode}.json')
    # Create an empty file if not exist
    if os.path.isfile(s3_file_name) == False:
        s3_files = {}
        with open(s3_file_name, "w") as f:
            json.dump(s3_files, f)
    s3 = boto3.client('s3')
    # List all objects in the S3 folder
    s3_objects = list_s3_objects(s3_client=s3, bucket_name=shared.models_s3_bucket, prefix=s3_folder)
    # only download on model at initialization
    fnames_dict = {}
    # if there v2 models, one root should have two files (.ckpt,.yaml)
    for obj in s3_objects:
        filename = obj['Key'].replace(s3_folder, '').lstrip('/')
        root, ext = os.path.splitext(filename)
        model = fnames_dict.get(root)
        if model:
            model.append(filename)
        else:
            fnames_dict[root] = [filename]
    tmp_s3_files = {}
    for i, obj in enumerate (s3_objects):
        etag = obj['ETag'].strip('"').strip("'")   
        size = obj['Size']/(1024**3)
        filename = obj['Key'].replace(s3_folder, '').lstrip('/')
        tmp_s3_files[filename] = [etag,size]
    
    #only fetch the first model to download. 
    if mode == 'sd':
        s3_files = {}
        try:
            _, file_names =  next(iter(fnames_dict.items()))
            for fname in file_names:
                s3_files[fname] = tmp_s3_files.get(fname)
                check_space_s3_download(s3,shared.models_s3_bucket, s3_folder,local_folder, fname, tmp_s3_files.get(fname)[1], mode)
                register_models(local_folder,mode)
        except Exception as e:
            traceback.print_stack()
            print(e)

    print(f'-----s3_files---{s3_files}')
    # save the lastest one
    with open(s3_file_name, "w") as f:
        json.dump(s3_files, f)
    


def sync_s3_folder(local_folder,cache_dir,mode):
    s3 = boto3.client('s3')
    def sync(mode):
        # print (f'sync:{mode}')
        if mode == 'sd':
            s3_folder = shared.s3_folder_sd 
        elif mode == 'cn':
            s3_folder = shared.s3_folder_cn 
        elif mode == 'lora':
            s3_folder = shared.s3_folder_lora
        else: 
            s3_folder = ''
        # Check and Create tmp folders 
        os.makedirs(os.path.dirname(local_folder), exist_ok=True)
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        s3_file_name = os.path.join(cache_dir,f's3_files_{mode}.json')
        # Create an empty file if not exist
        if os.path.isfile(s3_file_name) == False:
            s3_files = {}
            with open(s3_file_name, "w") as f:
                json.dump(s3_files, f)

        # List all objects in the S3 folder
        s3_objects = list_s3_objects(s3_client=s3,bucket_name=shared.models_s3_bucket, prefix=s3_folder)
        # Check if there are any new or deleted files
        s3_files = {}
        for obj in s3_objects:
            etag = obj['ETag'].strip('"').strip("'")   
            size = obj['Size']/(1024**3)
            key = obj['Key'].replace(s3_folder, '').lstrip('/')
            s3_files[key] = [etag,size]

        # to compared the latest s3 list with last time saved in local json,
        # read it first
        s3_files_local = {}
        with open(s3_file_name, "r") as f:
            s3_files_local = json.load(f)
        # print (f's3_files:{s3_files}')
        # print (f's3_files_local:{s3_files_local}')
        # save the lastest one
        with open(s3_file_name, "w") as f:
            json.dump(s3_files, f)
        mod_files = set()
        new_files = set([key for key in s3_files if key not in s3_files_local])
        del_files = set([key for key in s3_files_local if key not in s3_files])
        registerflag = False
        #compare etag changes
        for key in set(s3_files_local.keys()).intersection(s3_files.keys()):
            local_etag  = s3_files_local.get(key)[0]
            if local_etag and local_etag != s3_files[key][0]:
                mod_files.add(key)
        # Delete vanished files from local folder
        for file in del_files:
            if os.path.isfile(os.path.join(local_folder, file)):
                os.remove(os.path.join(local_folder, file))
                print(f'remove file {os.path.join(local_folder, file)}')
                de_register_model(file,mode)
        # Add new files 
        for file in new_files.union(mod_files):
            registerflag = True
            retry = 3 ##retry limit times to prevent dead loop in case other folders is empty
            while retry:
                ret = check_space_s3_download(s3,shared.models_s3_bucket, s3_folder,local_folder, file, s3_files[file][1], mode)
                #if the space is not enough free
                if ret:
                    retry = 0
                else:
                    free_local_disk(local_folder,s3_files[file][1],mode)
                    retry = retry - 1
        if registerflag:
            register_models(local_folder,mode)
            if mode == 'sd':
                #Refreshing Model List
                modules.sd_models.list_models()
            # cn models sync not supported temporally due to an unfixed bug
            elif mode == 'cn':
                modules.script_callbacks.update_cn_models_callback()
            elif mode == 'lora':
                print('To do: update lora??')


    # Create a thread function to keep syncing with the S3 folder
    def sync_thread(mode):  
        while True:
            syncLock.acquire()
            sync(mode)
            syncLock.release()
            time.sleep(30)
    thread = threading.Thread(target=sync_thread,args=(mode,))
    thread.start()
    print (f'{mode}_sync thread start')
    return thread

def register_models(models_dir,mode):
    if mode == 'sd':
        register_sd_models(models_dir)
    elif mode == 'cn':
        register_cn_models(models_dir)
    elif mode == 'lora':
        register_lora_models(models_dir)


def register_lora_models(lora_models_dir):
    print ('---register_lora_models()----')
    if 'endpoint_name' in os.environ:
        items = []
        params = {
            'module': 'Lora'
        }
        api_endpoint = os.environ['api_endpoint']
        endpoint_name = os.environ['endpoint_name']
        for file in get_models(lora_models_dir, ['*.pt', '*.ckpt', '*.safetensors']):
            hash = modules.sd_models.model_hash(os.path.join(lora_models_dir, file))
            item = {}
            item['model_name'] = os.path.basename(file)
            item['title'] = '{0} [{1}]'.format(os.path.splitext(os.path.basename(file))[0], hash)
            item['endpoint_name'] = endpoint_name
            items.append(item)
        inputs = {
            'items': items
        }
        if api_endpoint.startswith('http://') or api_endpoint.startswith('https://'):
            response = requests.post(url=f'{api_endpoint}/sd/models', json=inputs, params=params)
            print(response)

def register_sd_models(sd_models_dir):
    print ('---register_sd_models()----')
    if 'endpoint_name' in os.environ:
        items = []
        api_endpoint = os.environ['api_endpoint']
        endpoint_name = os.environ['endpoint_name']
        for file in get_models(sd_models_dir, ['*.ckpt', '*.safetensors']):
            hash = modules.sd_models.model_hash(file)
            item = {}
            item['model_name'] = file.replace("/tmp/models/Stable-diffusion/",'')
            item['hash'] = hash
            item['filename'] = file
            item['config'] = '/opt/ml/code/stable-diffusion-webui/repositories/stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
            item['title'] = '{0} [{1}]'.format(file.replace("/tmp/models/Stable-diffusion/",''), hash)
            item['endpoint_name'] = endpoint_name
            items.append(item)
        inputs = {
            'items': items
        }
        params = {
            'module': 'Stable-diffusion'
        }
        if api_endpoint.startswith('http://') or api_endpoint.startswith('https://'):
            response = requests.post(url=f'{api_endpoint}/sd/models', json=inputs, params=params)
            print(response)

def register_cn_models(cn_models_dir):
    print ('---register_cn_models()----')
    if 'endpoint_name' in os.environ:
        items = []
        api_endpoint = os.environ['api_endpoint']
        endpoint_name = os.environ['endpoint_name']
        params = {
            'module': 'ControlNet'
        }
        for file in get_models(cn_models_dir, ['*.pt', '*.pth', '*.ckpt', '*.safetensors']):
            hash = modules.sd_models.model_hash(os.path.join(cn_models_dir, file))
            item = {}
            item['model_name'] = os.path.basename(file)
            item['title'] = '{0} [{1}]'.format(os.path.splitext(os.path.basename(file))[0], hash)
            item['endpoint_name'] = endpoint_name
            items.append(item)
        inputs = {
            'items': items
        }
        if api_endpoint.startswith('http://') or api_endpoint.startswith('https://'):
            response = requests.post(url=f'{api_endpoint}/sd/models', json=inputs, params=params)
            print(response)

def webui():
    launch_api = cmd_opts.api

    if launch_api:
        models_config_s3uri = os.environ.get('models_config_s3uri', None)
        if models_config_s3uri:
            bucket, key = get_bucket_and_key(models_config_s3uri)
            s3_object = s3_client.get_object(Bucket=bucket, Key=key)
            bytes = s3_object["Body"].read()
            payload = bytes.decode('utf8')
            huggingface_models = json.loads(payload).get('huggingface_models', None)
            s3_models = json.loads(payload).get('s3_models', None)
            http_models = json.loads(payload).get('http_models', None)
        else:
            huggingface_models = os.environ.get('huggingface_models', None)
            s3_models = os.environ.get('s3_models', None)
            http_models = os.environ.get('http_models', None)

        if huggingface_models:
            huggingface_models = json.loads(huggingface_models)
            for huggingface_model in huggingface_models:
                repo_id = huggingface_model['repo_id']
                filename = huggingface_model['filename']
                name = huggingface_model['name']

                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=f'/tmp/models/{name}',
                    cache_dir='/tmp/cache/huggingface'
                )

        if s3_models:
            s3_models = json.loads(s3_models)
            for s3_model in s3_models:
                uri = s3_model['uri']
                name = s3_model['name']
                s3_download(uri, f'/tmp/models/{name}')

        if http_models:
            http_models = json.loads(http_models)
            for http_model in http_models:
                uri = http_model['uri']
                filename = http_model['filename']
                name = http_model['name']
                http_download(uri, f'/tmp/models/{name}/{filename}')

    ## auto reload new models from s3 add by River
    if not cmd_opts.pureui and not cmd_opts.train:
        print(os.system('df -h'))
        sd_models_tmp_dir = f"{shared.tmp_models_dir}/Stable-diffusion/"
        cn_models_tmp_dir = f"{shared.tmp_models_dir}/ControlNet/"
        lora_models_tmp_dir = f"{shared.tmp_models_dir}/Lora/"
        cache_dir = f"{shared.tmp_cache_dir}/"
        session = boto3.Session()
        region_name = session.region_name
        sts_client = session.client('sts')
        account_id = sts_client.get_caller_identity()['Account']
        sg_s3_bucket = f"sagemaker-{region_name}-{account_id}"
        if not shared.models_s3_bucket:
            shared.models_s3_bucket = os.environ['sg_default_bucket'] if os.environ.get('sg_default_bucket') else sg_s3_bucket
            shared.s3_folder_sd = "stable-diffusion-webui/models/Stable-diffusion"
            shared.s3_folder_cn = "stable-diffusion-webui/models/ControlNet"
            shared.s3_folder_lora = "stable-diffusion-webui/models/Lora"

         #only download the cn models and the first sd model from default bucket, to accerlate the startup time
        initial_s3_download(shared.s3_folder_sd,sd_models_tmp_dir,cache_dir,'sd')
        sync_s3_folder(sd_models_tmp_dir,cache_dir,'sd')
        sync_s3_folder(cn_models_tmp_dir,cache_dir,'cn')
        sync_s3_folder(lora_models_tmp_dir,cache_dir,'lora')

    ## end

    initialize()

    while 1:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()

        modules.script_callbacks.before_ui_callback()
        shared.demo = modules.ui.create_ui()

        app, local_url, share_url = shared.demo.queue(concurrency_count=5, max_size=20).launch(
            share=cmd_opts.share,
            server_name=server_name,
            server_port=cmd_opts.port,
            ssl_keyfile=cmd_opts.tls_keyfile,
            ssl_certfile=cmd_opts.tls_certfile,
            debug=cmd_opts.gradio_debug,
            auth=user_auth,
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

            

            cmd_sd_models_path = cmd_opts.ckpt_dir
            sd_models_dir = os.path.join(shared.models_path, "Stable-diffusion")
            if cmd_sd_models_path is not None:
                sd_models_dir = cmd_sd_models_path

            cmd_controlnet_models_path = cmd_opts.controlnet_dir
            cn_models_dir = os.path.join(shared.models_path, "ControlNet")
            if cmd_controlnet_models_path is not None:
                cn_models_dir = cmd_controlnet_models_path

            cmd_lora_models_path = cmd_opts.lora_dir
            lora_models_dir = os.path.join(shared.models_path, "Lora")
            if cmd_lora_models_path is not None:
                lora_models_dir = cmd_lora_models_path
            register_sd_models(sd_models_dir)
            register_cn_models(cn_models_dir)
            register_lora_models(lora_models_dir)

        ui_extra_networks.add_pages_to_demo(app)

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

        ui_extra_networks.intialize()
        ui_extra_networks.register_page(ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion())
        ui_extra_networks.register_page(ui_extra_networks_hypernets.ExtraNetworksPageHypernetworks())
        ui_extra_networks.register_page(ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints())

        extra_networks.initialize()
        extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())


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
        if cmd_opts.model_name != '':
            for huggingface_model in shared.huggingface_models:
                repo_id = huggingface_model['repo_id']
                filename = huggingface_model['filename']
                if filename == cmd_opts.model_name:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir='/opt/ml/input/data/models',
                        cache_dir='/opt/ml/input/data/models'
                    )
                    if filename in ['v2-1_768-ema-pruned.ckpt', '']:
                        name = os.path.splitext(filename)[0]
                        http_download(
                            'https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml',
                            f'/opt/ml/input/data/models/{name}.yaml'
                        )

        initialize()

        train_task = cmd_opts.train_task
        train_args = json.loads(cmd_opts.train_args)

        embeddings_s3uri = cmd_opts.embeddings_s3uri
        hypernetwork_s3uri = cmd_opts.hypernetwork_s3uri
        sd_models_s3uri = cmd_opts.sd_models_s3uri
        db_models_s3uri = cmd_opts.db_models_s3uri
        lora_models_s3uri = cmd_opts.lora_models_s3uri
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
                data = json.loads(response.text)
                try:
                    opts.data = json.loads(data['options'])
                except Exception as e:
                    print(e)
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
            db_use_txt2img = train_args['train_dreambooth_settings']['db_use_txt2img']
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
                db_train_unfrozen = train_args['train_dreambooth_settings']['db_train_unfrozen']
                db_512_model = train_args['train_dreambooth_settings']['db_512_model']
                db_save_safetensors = train_args['train_dreambooth_settings']['db_save_safetensors']

                db_model_name, db_model_path, db_revision, db_epochs, db_scheduler, db_src, db_has_ema, db_v2, db_resolution = create_model(
                    db_new_model_name,
                    db_new_model_src,
                    db_new_model_scheduler,
                    db_create_from_hub,
                    db_new_model_url,
                    db_new_model_token,
                    db_new_model_extract_ema,
                    db_train_unfrozen,
                    db_512_model
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
                    params_dict = json.loads(content)

                    params_dict['db_model_name'] = db_model_name
                    params_dict['db_model_path'] = db_model_path
                    params_dict['db_revision'] = db_revision
                    params_dict['db_epochs'] = db_epochs
                    params_dict['db_scheduler'] = db_scheduler
                    params_dict['db_src'] = db_src
                    params_dict['db_has_ema'] = db_has_ema
                    params_dict['db_v2'] = db_v2
                    params_dict['db_resolution'] = db_resolution

                    if db_train_wizard_person or db_train_wizard_object:
                        db_num_train_epochs, \
                        c1_num_class_images_per, \
                        c2_num_class_images_per, \
                        c3_num_class_images_per, \
                        c4_num_class_images_per = training_wizard(db_train_wizard_person if db_train_wizard_person else db_train_wizard_object)

                        params_dict['db_num_train_epochs'] = db_num_train_epochs
                        params_dict['c1_num_class_images_per'] = c1_num_class_images_per
                        params_dict['c1_num_class_images_per'] = c2_num_class_images_per
                        params_dict['c1_num_class_images_per'] = c3_num_class_images_per
                        params_dict['c1_num_class_images_per'] = c4_num_class_images_per
                    if db_performance_wizard:
                        attention, \
                        gradient_checkpointing, \
                        gradient_accumulation_steps, \
                        mixed_precision, \
                        cache_latents, \
                        sample_batch_size, \
                        train_batch_size, \
                        stop_text_encoder, \
                        use_8bit_adam, \
                        use_lora, \
                        use_ema, \
                        save_samples_every, \
                        save_weights_every = performance_wizard()

                        params_dict['attention'] = attention
                        params_dict['gradient_checkpointing'] = gradient_checkpointing
                        params_dict['gradient_accumulation_steps'] = gradient_accumulation_steps
                        params_dict['mixed_precision'] = mixed_precision
                        params_dict['cache_latents'] = cache_latents
                        params_dict['sample_batch_size'] = sample_batch_size
                        params_dict['train_batch_size'] = train_batch_size
                        params_dict['stop_text_encoder'] = stop_text_encoder
                        params_dict['use_8bit_adam'] = use_8bit_adam
                        params_dict['use_lora'] = use_lora
                        params_dict['use_ema'] = use_ema
                        params_dict['save_samples_every'] = save_samples_every 
                        params_dict['params_dict'] = save_weights_every

                    db_config = DreamboothConfig(db_model_name)
                    concept_keys = ["c1_", "c2_", "c3_", "c4_"]
                    concepts_list = []
                    # If using a concepts file/string, keep concepts_list empty.
                    if params_dict["db_use_concepts"] and params_dict["db_concepts_path"]:
                        concepts_list = []
                        params_dict["concepts_list"] = concepts_list
                    else:
                        for concept_key in concept_keys:
                            concept_dict = {}
                            for key, param in params_dict.items():
                                if concept_key in key and param is not None:
                                    concept_dict[key.replace(concept_key, "")] = param
                            concept_test = Concept(concept_dict)
                            if concept_test.is_valid:
                                concepts_list.append(concept_test.__dict__)
                        existing_concepts = params_dict["concepts_list"] if "concepts_list" in params_dict else []
                        if len(concepts_list) and not len(existing_concepts):
                            params_dict["concepts_list"] = concepts_list

                    db_config.load_params(params_dict)
            else:
                db_model_name = train_args['train_dreambooth_settings']['db_model_name']
                db_config = DreamboothConfig(db_model_name)

            print(vars(db_config))
            start_training_from_config(
                db_config,
                db_use_txt2img,
            )

            cmd_sd_models_path = cmd_opts.ckpt_dir
            sd_models_dir = os.path.join(shared.models_path, "Stable-diffusion")
            if cmd_sd_models_path is not None:
                sd_models_dir = cmd_sd_models_path

            try:
                cmd_dreambooth_models_path = cmd_opts.dreambooth_models_path
            except:
                cmd_dreambooth_models_path = None

            try:
                cmd_lora_models_path = shared.cmd_opts.lora_models_path
            except:
                cmd_lora_models_path = None

            db_model_dir = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
            db_model_dir = os.path.join(db_model_dir, "dreambooth")

            lora_model_dir = os.path.dirname(cmd_lora_models_path) if cmd_lora_models_path else paths.models_path
            lora_model_dir = os.path.join(lora_model_dir, "lora")

            print('---models path---', sd_models_dir, lora_model_dir)
            print(os.system(f'ls -l {sd_models_dir}'))
            print(os.system('ls -l {0}'.format(os.path.join(sd_models_dir, db_model_name))))
            print(os.system(f'ls -l {lora_model_dir}'))

            try:
                print('Uploading SD Models...')
                if db_config.v2:
                    upload_s3files(
                        f'{sd_models_s3uri}{username}/',
                        os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.yaml')
                    )
                if db_config.save_safetensors:
                    upload_s3files(
                        f'{sd_models_s3uri}{username}/',
                        os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.safetensors')
                    )
                else:
                    upload_s3files(
                        f'{sd_models_s3uri}{username}/',
                        os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.ckpt')
                    )
                print('Uploading DB Models...')
                upload_s3folder(
                    f'{db_models_s3uri}{username}/{db_model_name}',
                    os.path.join(db_model_dir, db_model_name)
                )
                if db_config.use_lora:
                    print('Uploading Lora Models...')
                    upload_s3files(
                        f'{lora_models_s3uri}{username}/',
                        os.path.join(lora_model_dir, f'{db_model_name}_*.pt')
                    )
                #automatic tar latest checkpoint and upload to s3 by zheng on 2023.03.22
                os.makedirs(os.path.dirname("/opt/ml/model/"), exist_ok=True)
                os.makedirs(os.path.dirname("/opt/ml/model/Stable-diffusion/"), exist_ok=True)
                os.makedirs(os.path.dirname("/opt/ml/model/ControlNet/"), exist_ok=True)
                train_steps=int(db_config.revision)
                model_file_basename = f'{db_model_name}_{train_steps}_lora' if db_config.use_lora else f'{db_model_name}_{train_steps}'
                if db_config.v2:
                    f1=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.yaml')
                    if os.path.exists(f1):
                        shutil.copy(f1,"/opt/ml/model/Stable-diffusion/")
                if db_config.save_safetensors:
                    f2=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.safetensors')
                    if os.path.exists(f2):
                        shutil.copy(f2,"/opt/ml/model/Stable-diffusion/")
                else:
                    f2=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.ckpt')
                    if os.path.exists(f2):
                        shutil.copy(f2,"/opt/ml/model/Stable-diffusion/")
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
