import sys

import gradio as gr

from modules import shared_cmd_options, shared_gradio_themes, options, shared_items, sd_models_types
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir  # noqa: F401
from modules import util

cmd_opts = shared_cmd_options.cmd_opts
parser = shared_cmd_options.parser

batch_cond_uncond = True  # old field, unused now in favor of shared.opts.batch_cond_uncond
parallel_processing_allowed = True
styles_filename = cmd_opts.styles_file
config_filename = cmd_opts.ui_settings_file
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}

demo = None

device = None

weight_load_location = None

xformers_available = False

hypernetworks = {}

loaded_hypernetworks = []

state = None

prompt_styles = None

interrogator = None

face_restorers = []

options_templates = None
opts = None
restricted_opts = None

sd_model: sd_models_types.WebuiSdModel = None

settings_components = None
"""assinged from ui.py, a mapping on setting names to gradio components repsponsible for those settings"""

tab_names = []

latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

sd_upscalers = []

clip_model = None

progress_print_out = sys.stdout

gradio_theme = gr.themes.Base()

total_tqdm = None

mem_mon = None

options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

natural_sort_key = util.natural_sort_key
listfiles = util.listfiles
html_path = util.html_path
html = util.html
walk_files = util.walk_files
ldm_print = util.ldm_print

reload_gradio_theme = shared_gradio_themes.reload_gradio_theme

list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
reload_hypernetworks = shared_items.reload_hypernetworks

import os
import json
import boto3
import requests
import glob
from botocore.errorfactory import ClientError

cache = dict()
region_name = boto3.session.Session().region_name if not cmd_opts.train else cmd_opts.region_name
s3_client = boto3.client('s3', region_name=region_name)
endpointUrl = s3_client.meta.endpoint_url
s3_client = boto3.client('s3', endpoint_url=endpointUrl, region_name=region_name)
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

    try:
        if os.path.isfile('cache'):
            cache = json.load(open('cache', 'r'))
    except:
        pass

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