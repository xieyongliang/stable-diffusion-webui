import html
import sys
import threading
import traceback
import time
from PIL import Image, ImageOps, ImageChops
import requests
import json
import time
from PIL import Image
import base64
import io
import traceback
from modules.shared import cmd_opts, opts
from modules import sd_samplers
import modules
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops

from modules import shared
import gradio as gr

queue_lock = threading.Lock()


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

    def sagemaker_inference(task, infer, username, sagemaker_endpoint, *args, **kwargs):
        infer = 'async'
        if task == 'text-to-image' or task == 'image-to-image':
            if task == 'text-to-image':
                script_args = []
                for i in range(23, len(args)):
                    if(isinstance(args[i], dict)):
                        script_arg = {}
                        for key in args[i]:
                            if key == 'image' or key == 'mask':
                                script_arg[key] = encode_image_to_base64(Image.fromarray(args[i][key]))
                            else:
                                script_arg[key] = args[i][key]
                        script_args.append(script_arg)
                    else:
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
                    "eta": opts.eta_ddim if sampler_index == 'DDIM' or sampler_index == 'PLMS' else opts.eta_ancestral,
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
                    'username': username
                }
            else:
                mode = args[0]
                prompt = args[1]
                negative_prompt = args[2]
                styles = [args[3], args[4]]
                init_img = args[5]
                init_img_with_mask = args[6]
                init_img_with_mask_orig = args[7]
                init_img_inpaint = args[8]
                init_mask_inpaint = args[9]
                mask_mode = args[10]
                steps = args[11]
                sampler_index = sd_samplers.samplers_for_img2img[args[12]].name
                mask_blur = args[13]
                mask_alpha = args[14]
                inpainting_fill = args[15]
                restore_faces = args[16]
                tiling = args[17]
                n_iter = args[18]
                batch_size = args[19]
                cfg_scale = args[20]
                denoising_strength = args[21]
                seed = args[22]
                subseed = args[23]
                subseed_strength = args[24]
                seed_resize_from_h = args[25]
                seed_resize_from_w = args[26]
                seed_enable_extras = args[27]
                height = args[28]
                width = args[29]
                resize_mode = args[30]
                inpaint_full_res = args[31]
                inpaint_full_res_padding = args[32]
                inpainting_mask_invert = args[33]
                img2img_batch_input_dir = args[34]
                img2img_batch_output_dir = args[35]

                script_args = []
                for i in range(36, len(args)):
                    if(isinstance(args[i], dict)):
                        script_arg = {}
                        for key in args[i]:
                            if key == 'image' or key == 'mask':
                                script_arg[key] = encode_image_to_base64(Image.fromarray(args[i][key]))
                            else:
                                script_arg[key] = args[i][key]
                        script_args.append(script_arg)
                    else:
                        script_args.append(args[i])

                if mode == 1:
                    # Drawn mask
                    if mask_mode == 0:
                        is_mask_sketch = isinstance(init_img_with_mask, dict)
                        if is_mask_sketch:
                            # Sketch: mask iff. not transparent
                            image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
                            alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
                            mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
                        else:
                            # Color-sketch: mask iff. painted over
                            image = init_img_with_mask
                            orig = init_img_with_mask_orig or init_img_with_mask
                            pred = np.any(np.array(image) != np.array(orig), axis=-1)
                            mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
                            mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
                            blur = ImageFilter.GaussianBlur(mask_blur)
                            image = Image.composite(image.filter(blur), orig, mask.filter(blur))

                        image = image.convert("RGB")

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
                    "cfg_scale": cfg_scale,
                    "width": width,
                    "height": height,
                    "restore_faces": restore_faces,
                    "tiling": tiling,
                    "negative_prompt": negative_prompt,
                    "eta": opts.eta_ddim if sampler_index == 'DDIM' or sampler_index == 'PLMS' else opts.eta_ancestral,
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
                    'username': username
                }

            params = {
                'endpoint_name': sagemaker_endpoint
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
                    'username': username
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
                    'username': username
                }

            params = {
                'endpoint_name': sagemaker_endpoint
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

    def f(username, *args, **kwargs):
        if cmd_opts.pureui and func == modules.txt2img.txt2img:
            sagemaker_endpoint = args[len(args) -1]
            args = args[:-2]
            res = sagemaker_inference('text-to-image', 'sync', username, sagemaker_endpoint, *args, **kwargs)
        elif cmd_opts.pureui and func == modules.img2img.img2img:
            sagemaker_endpoint = args[len(args) -1]
            args = args[:-2]
            res = sagemaker_inference('image-to-image', 'sync', username, sagemaker_endpoint, *args, **kwargs)
        elif cmd_opts.pureui and func == modules.extras.run_extras:
            sagemaker_endpoint = args[len(args) -1]
            args = args[:-2]
            res = sagemaker_inference('extras', 'sync', username, sagemaker_endpoint, *args, **kwargs)
        else:
            shared.state.begin()
            with queue_lock:
                res = func(*args, **kwargs)
            shared.state.end()

        return res

    return wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True)


def wrap_gradio_call(func, extra_outputs=None, add_stats=False):
    def f(request: gr.Request, *args, extra_outputs_array=extra_outputs, **kwargs):
        tokens = shared.demo.server_app.tokens
        cookies = request.headers['cookie'].split('; ')
        access_token = None
        for cookie in cookies:
            if cookie.startswith('access-token'):
                access_token = cookie[len('access-token=') : ]
                break
        username = tokens[access_token]

        run_memmon = shared.opts.memmon_poll_rate > 0 and not shared.mem_mon.disabled and add_stats
        if run_memmon:
            shared.mem_mon.monitor()
        t = time.perf_counter()

        try:
            if func.__name__ == 'f' or func.__name__ == 'run_settings':
                res = list(func(username, *args, **kwargs))
            else:
                res = list(func(*args, **kwargs))
        except Exception as e:
            # When printing out our debug argument list, do not print out more than a MB of text
            max_debug_str_len = 131072 # (1024*1024)/8

            print("Error completing request", file=sys.stderr)
            argStr = f"Arguments: {str(args)} {str(kwargs)}"
            print(argStr[:max_debug_str_len], file=sys.stderr)
            if len(argStr) > max_debug_str_len:
                print(f"(Argument list truncated at {max_debug_str_len}/{len(argStr)} characters)", file=sys.stderr)

            print(traceback.format_exc(), file=sys.stderr)

            shared.state.job = ""
            shared.state.job_count = 0

            if extra_outputs_array is None:
                extra_outputs_array = [None, '']

            res = extra_outputs_array + [f"<div class='error'>{html.escape(type(e).__name__+': '+str(e))}</div>"]

        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.job_count = 0

        if not add_stats:
            return tuple(res)

        elapsed = time.perf_counter() - t
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
        elapsed_text = f"{elapsed_s:.2f}s"
        if elapsed_m > 0:
            elapsed_text = f"{elapsed_m}m "+elapsed_text

        if run_memmon:
            mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
            active_peak = mem_stats['active_peak']
            reserved_peak = mem_stats['reserved_peak']
            sys_peak = mem_stats['system_peak']
            sys_total = mem_stats['total']
            sys_pct = round(sys_peak/max(sys_total, 1) * 100, 2)

            vram_html = f"<p class='vram'>Torch active/reserved: {active_peak}/{reserved_peak} MiB, <wbr>Sys VRAM: {sys_peak}/{sys_total} MiB ({sys_pct}%)</p>"
        else:
            vram_html = ''

        # last item is always HTML
        res[-1] += f"<div class='performance'><p class='time'>Time taken: <wbr>{elapsed_text}</p>{vram_html}</div>"

        return tuple(res)

    return f
