import base64
import io
import time
import uvicorn
from threading import Lock
from io import BytesIO
from gradio_client import utils as client_utils
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from secrets import compare_digest
from modules.shared import de_register_model
import modules.shared as shared
from modules import sd_samplers, deepbooru
from modules.api.models import *
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.extras import run_extras, run_pnginfo
from PIL import PngImagePlugin,Image
from modules.sd_models import checkpoints_list
from modules.realesrgan_model import get_realesrgan_models
from typing import List
from modules.paths import script_path
import json
import os
import boto3
from modules import sd_hijack, hypernetworks, sd_models
from typing import Union
import traceback
import requests
import piexif
import piexif.helper
import numpy as np
import uuid
import modules.sd_vae
from datetime import datetime

def upscaler_to_index(name: str):
    try:
        return [x.name.lower() for x in shared.sd_upscalers].index(name.lower())
    except:
        raise HTTPException(status_code=400, detail=f"Invalid upscaler, needs to be on of these: {' , '.join([x.name for x in sd_upscalers])}")


def validate_sampler_name(name):
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is None:
        raise HTTPException(status_code=404, detail="Sampler not found")

    return name

def setUpscalers(req: dict):
    reqDict = vars(req)
    reqDict['extras_upscaler_1'] = upscaler_to_index(req.upscaler_1)
    reqDict['extras_upscaler_2'] = upscaler_to_index(req.upscaler_2)
    reqDict.pop('upscaler_1')
    reqDict.pop('upscaler_2')
    return reqDict

def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    return Image.open(BytesIO(base64.b64decode(encoding)))

def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return encode_pil_to_base64(image)
    elif type(image) is np.ndarray:
        return encode_np_to_base64(image)
    else:
        return ""

def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        if opts.samples_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            })
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif = exif_bytes, quality=opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif = exif_bytes, quality=opts.jpeg_quality)

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)

def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return encode_pil_to_base64(pil)

def export_pil_to_bytes(image):
    with io.BytesIO() as output_bytes:

        if opts.samples_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            })
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif = exif_bytes, quality=opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif = exif_bytes, quality=opts.jpeg_quality)

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return bytes_data

class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        if shared.cmd_opts.api_auth:
            self.credenticals = dict()
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credenticals[user] = password

        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
        self.add_api_route("/sdapi/v1/txt2img", self.text2imgapi, methods=["POST"], response_model=TextToImageResponse)
        self.add_api_route("/sdapi/v1/img2img", self.img2imgapi, methods=["POST"], response_model=ImageToImageResponse)
        self.add_api_route("/sdapi/v1/extra-single-image", self.extras_single_image_api, methods=["POST"], response_model=ExtrasSingleImageResponse)
        self.add_api_route("/sdapi/v1/extra-batch-images", self.extras_batch_images_api, methods=["POST"], response_model=ExtrasBatchImagesResponse)
        self.add_api_route("/sdapi/v1/png-info", self.pnginfoapi, methods=["POST"], response_model=PNGInfoResponse)
        self.add_api_route("/sdapi/v1/progress", self.progressapi, methods=["GET"], response_model=ProgressResponse)
        self.add_api_route("/sdapi/v1/interrogate", self.interrogateapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/interrupt", self.interruptapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/skip", self.skip, methods=["POST"])
        self.add_api_route("/sdapi/v1/options", self.get_config, methods=["GET"], response_model=OptionsModel)
        self.add_api_route("/sdapi/v1/options", self.set_config, methods=["POST"])
        self.add_api_route("/sdapi/v1/cmd-flags", self.get_cmd_flags, methods=["GET"], response_model=FlagsModel)
        self.add_api_route("/sdapi/v1/samplers", self.get_samplers, methods=["GET"], response_model=List[SamplerItem])
        self.add_api_route("/sdapi/v1/upscalers", self.get_upscalers, methods=["GET"], response_model=List[UpscalerItem])
        self.add_api_route("/sdapi/v1/sd-models", self.get_sd_models, methods=["GET"], response_model=List[SDModelItem])
        self.add_api_route("/sdapi/v1/hypernetworks", self.get_hypernetworks, methods=["GET"], response_model=List[HypernetworkItem])
        self.add_api_route("/sdapi/v1/face-restorers", self.get_face_restorers, methods=["GET"], response_model=List[FaceRestorerItem])
        self.add_api_route("/sdapi/v1/realesrgan-models", self.get_realesrgan_models, methods=["GET"], response_model=List[RealesrganItem])
        self.add_api_route("/sdapi/v1/prompt-styles", self.get_promp_styles, methods=["GET"], response_model=List[PromptStyleItem])
        self.add_api_route("/sdapi/v1/artist-categories", self.get_artists_categories, methods=["GET"], response_model=List[str])
        self.add_api_route("/sdapi/v1/artists", self.get_artists, methods=["GET"], response_model=List[ArtistItem])
        self.app.add_api_route("/invocations", self.invocations, methods=["POST"], response_model=Union[TextToImageResponse, ImageToImageResponse, ExtrasSingleImageResponse, ExtrasBatchImagesResponse, List[SDModelItem]])
        self.app.add_api_route("/ping", self.ping, methods=["GET"], response_model=PingResponse)
        self.cache = dict()
        self.s3_client = boto3.client('s3')
        self.s3_resource= boto3.resource('s3')

    def add_api_route(self, path: str, endpoint, **kwargs):
        if shared.cmd_opts.api_auth:
            return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def auth(self, credenticals: HTTPBasicCredentials = Depends(HTTPBasic())):
        if credenticals.username in self.credenticals:
            if compare_digest(credenticals.password, self.credenticals[credenticals.username]):
                return True

        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})

    def text2imgapi(self, txt2imgreq: StableDiffusionTxt2ImgProcessingAPI):
        populate = txt2imgreq.copy(update={ # Override __init__ params
            "sd_model": shared.sd_model,
            "sampler_name": validate_sampler_name(txt2imgreq.sampler_name or txt2imgreq.sampler_index),
            "do_not_save_samples": True,
            "do_not_save_grid": False
            }
        )
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on
        p = StableDiffusionProcessingTxt2Img(**vars(populate))
        # Override object param

        shared.state.begin()

        with self.queue_lock:
            processed = p.scripts.run(p, *p.script_args)
            if processed is None:
                processed = process_images(p)

        shared.state.end()

        b64images = list(map(encode_to_base64, processed.images))

        return TextToImageResponse(images=b64images, parameters=vars(txt2imgreq), info=processed.js())

    def img2imgapi(self, img2imgreq: StableDiffusionImg2ImgProcessingAPI):
        init_images = img2imgreq.init_images
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found")

        mask = img2imgreq.mask
        if mask:
            mask = decode_base64_to_image(mask)

        populate = img2imgreq.copy(update={ # Override __init__ params
            "sd_model": shared.sd_model,
            "sampler_name": validate_sampler_name(img2imgreq.sampler_name or img2imgreq.sampler_index),
            "do_not_save_samples": True,
            "do_not_save_grid": False,
            "mask": mask
            }
        )
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop('include_init_images', None)  # this is meant to be done by "exclude": True in model, but it's for a reason that I cannot determine.
        p = StableDiffusionProcessingImg2Img(**args)

        imgs = []
        for img in init_images:
            img = decode_base64_to_image(img)
            imgs = [img] * p.batch_size

        p.init_images = imgs

        shared.state.begin()

        with self.queue_lock:
            processed = p.scripts.run(p, *p.script_args)
            if processed is None:
                processed = process_images(p)

        shared.state.end()

        b64images = list(map(encode_to_base64, processed.images))

        if not img2imgreq.include_init_images:
            img2imgreq.init_images = None
            img2imgreq.mask = None

        return ImageToImageResponse(images=b64images, parameters=vars(img2imgreq), info=processed.js())

    def extras_single_image_api(self, req: ExtrasSingleImageRequest):
        reqDict = setUpscalers(req)

        reqDict['image'] = decode_base64_to_image(reqDict['image'])

        with self.queue_lock:
            result = run_extras(extras_mode=0, image_folder="", input_dir="", output_dir="", **reqDict)

        return ExtrasSingleImageResponse(image=encode_to_base64(result[0][0]), html_info=result[1])

    def extras_batch_images_api(self, req: ExtrasBatchImagesRequest):
        reqDict = setUpscalers(req)

        def prepareFiles(file):
            file = client_utils.decode_base64_to_file(file.data, file_path=file.name)
            file.orig_name = file.name
            return file

        reqDict['image_folder'] = list(map(prepareFiles, reqDict['imageList']))
        reqDict.pop('imageList')

        with self.queue_lock:
            result = run_extras(extras_mode=1, image="", input_dir="", output_dir="", **reqDict)

        return ExtrasBatchImagesResponse(images=list(map(encode_to_base64, result[0])), html_info=result[1])

    def pnginfoapi(self, req: PNGInfoRequest):
        if(not req.image.strip()):
            return PNGInfoResponse(info="")

        result = run_pnginfo(decode_base64_to_image(req.image.strip()))

        return PNGInfoResponse(info=result[1])

    def progressapi(self, req: ProgressRequest = Depends()):
        # copy from check_progress_call of ui.py

        if shared.state.job_count == 0:
            return ProgressResponse(progress=0, eta_relative=0, state=shared.state.dict())

        # avoid dividing zero
        progress = 0.01

        if shared.state.job_count > 0:
            progress += shared.state.job_no / shared.state.job_count
        if shared.state.sampling_steps > 0:
            progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start

        progress = min(progress, 1)

        shared.state.set_current_image()

        current_image = None
        if shared.state.current_image and not req.skip_current_image:
            current_image = encode_to_base64(shared.state.current_image)

        return ProgressResponse(progress=progress, eta_relative=eta_relative, state=shared.state.dict(), current_image=current_image)

    def interrogateapi(self, interrogatereq: InterrogateRequest):
        image_b64 = interrogatereq.image
        if image_b64 is None:
            raise HTTPException(status_code=404, detail="Image not found") 

        img = decode_base64_to_image(image_b64)
        img = img.convert('RGB')

        # Override object param
        with self.queue_lock:
            if interrogatereq.model == "clip":
                processed = shared.interrogator.interrogate(img)
            elif interrogatereq.model == "deepdanbooru":
                processed = deepbooru.model.tag(img)
            else:
                raise HTTPException(status_code=404, detail="Model not found")
        
        return InterrogateResponse(caption=processed)

    def interruptapi(self):
        shared.state.interrupt()

        return {}

    def skip(self):
        shared.state.skip()

    def get_config(self):
        options = {}
        for key in shared.opts.data.keys():
            metadata = shared.opts.data_labels.get(key)
            if(metadata is not None):
                options.update({key: shared.opts.data.get(key, shared.opts.data_labels.get(key).default)})
            else:
                options.update({key: shared.opts.data.get(key, None)})

        return options

    def set_config(self, req: Dict[str, Any]):
        for k, v in req.items():
            shared.opts.set(k, v)

        shared.opts.save(shared.config_filename)
        return

    def get_cmd_flags(self):
        return vars(shared.cmd_opts)

    def get_samplers(self):
        return [{"name": sampler[0], "aliases":sampler[2], "options":sampler[3]} for sampler in sd_samplers.all_samplers]

    def get_upscalers(self):
        upscalers = []

        for upscaler in shared.sd_upscalers:
            u = upscaler.scaler
            upscalers.append({"name":u.name, "model_name":u.model_name, "model_path":u.model_path, "model_url":u.model_url})

        return upscalers

    def get_sd_models(self):
        return [{"title":x.title, "model_name":x.model_name, "hash":x.hash, "filename": x.filename, "config": x.config} for x in checkpoints_list.values()]

    def get_hypernetworks(self):
        return [{"name": name, "path": shared.hypernetworks[name]} for name in shared.hypernetworks]

    def get_face_restorers(self):
        return [{"name":x.name(), "cmd_dir": getattr(x, "cmd_dir", None)} for x in shared.face_restorers]

    def get_realesrgan_models(self):
        return [{"name":x.name,"path":x.data_path, "scale":x.scale} for x in get_realesrgan_models(None)]

    def get_promp_styles(self):
        styleList = []
        for k in shared.prompt_styles.styles:
            style = shared.prompt_styles.styles[k]
            styleList.append({"name":style[0], "prompt": style[1], "negative_prompt": style[2]})

        return styleList

    def get_artists_categories(self):
        return shared.artist_db.cats

    def get_artists(self):
        return [{"name":x[0], "score":x[1], "category":x[2]} for x in shared.artist_db.artists]

    def download_s3files(self, s3uri, path):
        pos = s3uri.find('/', 5)
        bucket = s3uri[5 : pos]
        key = s3uri[pos + 1 : ]

        s3_bucket = self.s3_resource.Bucket(bucket)
        objs = list(s3_bucket.objects.filter(Prefix=key))

        if os.path.isfile('cache'):
            self.cache = json.load(open('cache', 'r'))

        for obj in objs:
            response = self.s3_client.head_object(
                Bucket = bucket,
                Key =  obj.key
            )
            obj_key = 's3://{0}/{1}'.format(bucket, obj.key)
            if obj_key not in self.cache or self.cache[obj_key] != response['ETag']:
                filename = obj.key[obj.key.rfind('/') + 1 : ]

                self.s3_client.download_file(bucket, obj.key, os.path.join(path, filename))
                self.cache[obj_key] = response['ETag']

        json.dump(self.cache, open('cache', 'w'))

    def post_invocations(self, username, b64images,task):
        generated_images_s3uri = os.environ.get('generated_images_s3uri', None)

        if generated_images_s3uri:
            generated_images_s3uri = f'{generated_images_s3uri}{username}/{task}/'
            bucket, key = self.get_bucket_and_key(generated_images_s3uri)
            for b64image in b64images:
                bytes_data = export_pil_to_bytes(decode_base64_to_image(b64image))
                image_id = datetime.now().strftime(f"%Y%m%d%H%M%S-{uuid.uuid4()}")
                suffix = opts.samples_format.lower()
                self.s3_client.put_object(
                    Body=bytes_data,
                    Bucket=bucket,
                    Key=f'{key}{image_id}.{suffix}'
                )

    def invocations(self, req: InvocationsRequest):
        print('-------invocation------')
        print(req)

        embeddings_s3uri = shared.cmd_opts.embeddings_s3uri
        hypernetwork_s3uri = shared.cmd_opts.hypernetwork_s3uri

        try:
            username = req.username
            default_options = shared.opts.data

            if username != '':
                inputs = {
                    'action': 'get',
                    'username': username
                }
                api_endpoint = os.environ['api_endpoint']
                response = requests.post(url=f'{api_endpoint}/sd/user', json=inputs)
                if response.status_code == 200 and response.text != '':
                    try:
                        data = json.loads(response.text)
                        sd_model_checkpoint = shared.opts.sd_model_checkpoint
                        shared.opts.data = json.loads(data['options'])
                        modules.sd_vae.refresh_vae_list()
                        with self.queue_lock:
                            sd_models.reload_model_weights()
                            if sd_model_checkpoint == shared.opts.sd_model_checkpoint:
                                modules.sd_vae.reload_vae_weights()
                    except Exception as e:
                        print(e)

                self.download_s3files(hypernetwork_s3uri, shared.cmd_opts.hypernetwork_dir)
                hypernetworks.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)
                hypernetworks.hypernetwork.apply_strength()
            ##add sd model usage stats by River
            print(f'default_options:{shared.opts.data}')
            shared.sd_models_Ref.add_models_ref(shared.opts.data['sd_model_checkpoint'])
            ##end 
            if req.task == 'text-to-image':
                self.download_s3files(embeddings_s3uri, shared.cmd_opts.embeddings_dir)
                sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
                response = self.text2imgapi(req.txt2img_payload)
                self.post_invocations(username, response.images,req.task)
                shared.opts.data = default_options
                return response
            elif req.task == 'image-to-image':
                self.download_s3files(embeddings_s3uri, shared.cmd_opts.embeddings_dir)
                sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
                response = self.img2imgapi(req.img2img_payload)
                self.post_invocations(username, response.images,req.task)
                shared.opts.data = default_options
                return response
            elif req.task == 'extras-single-image':
                response = self.extras_single_image_api(req.extras_single_payload)
                self.post_invocations(username, [response.image],req.task)
                shared.opts.data = default_options
                return response
            elif req.task == 'extras-batch-images':
                response = self.extras_batch_images_api(req.extras_batch_payload)
                self.post_invocations(username, response.images,req.task)
                shared.opts.data = default_options
                return response                
            elif req.task == 'reload-all-models':
                return self.reload_all_models()
            elif req.task == 'set-models-bucket':
                bucket = req.models_bucket
                return self.set_models_bucket(bucket)
            else:
                raise NotImplementedError
        except Exception as e:
            traceback.print_exc()

    def ping(self):
        # print('-------ping------')
        return {'status': 'Healthy'}

    def reload_all_models(self):
        print('-------reload_all_models------')
        def remove_files(path):
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f'{file_path} has been removed')
                    if file_path.find('Stable-diffusion'):
                        de_register_model(file_name,'sd')
                    elif file_path.find('ControlNet'):
                        de_register_model(file_name,'cn')
                elif os.path.isdir(file_path):
                    remove_files(file_path)
                    os.rmdir(file_path)
        shared.syncLock.acquire()
        #remove all files in /tmp/models/ and /tmp/cache/
        remove_files(shared.tmp_models_dir)
        remove_files(shared.tmp_cache_dir)
        shared.syncLock.release()
        return {'simple_result':'success'}
    
    def set_models_bucket(self,bucket):
        shared.syncLock.acquire()
        if bucket.endswith('/'):
            bucket = bucket[:-1]
        url_parts = bucket.replace('s3://','').split('/')
        shared.models_s3_bucket = url_parts[0]
        lastfolder = url_parts[-1]
        if lastfolder == 'Stable-diffusion':
            shared.s3_folder_sd = '/'.join(url_parts[1:])
        elif lastfolder == 'ControlNet':
            shared.s3_folder_cn = '/'.join(url_parts[1:])
        else:
            shared.s3_folder_sd = '/'.join(url_parts[1:]+['Stable-diffusion'])
            shared.s3_folder_cn = '/'.join(url_parts[1:]+['ControlNet'])
        print(f'set_models_bucket to {shared.models_s3_bucket}')
        print(f'set_s3_folder_sd to {shared.s3_folder_sd}')
        print(f'set_s3_folder_cn to {shared.s3_folder_cn}')
        shared.syncLock.release()
        return {'simple_result':'success'}

    def launch(self, server_name, port):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port)

    def get_bucket_and_key(self, s3uri):
        pos = s3uri.find('/', 5)
        bucket = s3uri[5 : pos]
        key = s3uri[pos + 1 : ]
        return bucket, key
