import html
import sys
import threading
import traceback
import time

from modules import shared, progress

import requests
import json
import base64
import io
import numpy as np
import os
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops
from modules.shared import cmd_opts, opts
import modules

import gradio as gr
import modules.sd_samplers

queue_lock = threading.Lock()

import modules.scripts

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
            image = image.convert('RGB')
            image.tobytes("hex", "rgb")
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG')
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
        if s3uri.startswith('s3://'):
            params = {'s3uri': s3uri}
            start = time.time()
            while True:
                if shared.state.interrupted or shared.state.skipped:
                    shared.job_count = 0
                    return None

                response = requests.get(url=f'{shared.api_endpoint}/s3', params = params)
                if response.status_code == 200:
                    try:
                        text = json.loads(response.text)
                        if text['count'] > 0:
                            break
                        else:
                            time.sleep(1)
                    except Exception as e:
                        processed = {}
                        processed['error'] = str(e)
                        print(response.text)
                        return processed
                else:
                    processed = {}
                    processed['error'] =  response.text
                    print(response.text)
                    return processed

            httpuri = text['payload'][0]['httpuri']
            response = requests.get(url=httpuri)
            try:
                processed = json.loads(response.text)
                print(f"Time taken: {time.time() - start}s")
                shared.job_count = 0
                return processed
            except Exception:
                print(response.text)
        else:
            processed = {}
            processed['error'] = response.text
            print(response.text)
            return processed

    def sagemaker_inference(task, infer_type, username, sagemaker_endpoint, *args, **kwargs):
        if task == 'text-to-image' or task == 'image-to-image':
            if task == 'text-to-image':
                script_args = []
                for i in range(29, len(args)):
                    if(isinstance(args[i], dict)):
                        script_arg = {}
                        for key in args[i]:
                            if key == 'image' or key == 'mask':
                                script_arg[key] = encode_image_to_base64(Image.fromarray(args[i][key]))
                            else:
                                script_arg[key] = args[i][key]
                        script_args.append(script_arg)
                    elif hasattr(args[i], '__dict__'):
                        script_arg = {}
                        for key in args[i].__dict__:
                            if key == 'image':
                                if args[i].__dict__[key]:
                                    script_arg[key] = {}
                                    if 'image' in args[i].__dict__[key]:
                                        script_arg[key]['image'] = encode_image_to_base64(Image.fromarray(args[i].__dict__[key]['image']))
                                    if 'mask' in args[i].__dict__[key]:
                                        script_arg[key]['mask'] = encode_image_to_base64(Image.fromarray(args[i].__dict__[key]['mask']))
                                else:
                                    script_arg[key] = None
                            else:
                                script_arg[key] = args[i].__dict__[key]
                        script_args.append(script_arg)
                    else:
                        script_args.append(args[i])

                txt2img_prompt = args[0]
                txt2img_negative_prompt = args[1]
                txt2img_prompt_styles = args[2]
                steps = args[3]
                sampler_index = modules.sd_samplers.samplers[args[4]].name
                restore_faces = args[5]
                tiling = args[6]
                batch_count = args[7]
                batch_size = args[8]
                cfg_scale = args[9]
                seed = args[10]
                subseed = args[11]
                subseed_strength = args[12]
                seed_resize_from_h = args[13]
                seed_resize_from_w = args[14]
                seed_enable_extras = args[15]
                height = args[16]
                width = args[17]
                enable_hr = args[18]
                denoising_strength = args[19]
                hr_scale = args[20]
                hr_upscaler = args[21]
                hr_second_pass_steps = args[22]
                hr_resize_x = args[23]
                hr_resize_y = args[24]
                hr_sampler_index = args[25]
                hr_prompt = args[26]
                hr_negative_prompt = args[27]
                override_settings = args[28]
                alwayson_scripts = {}
                script_name = None
                script_index = None

                for script in modules.scripts.scripts_txt2img.alwayson_scripts:
                    alwayson_script_arg = {
                        'args': []
                    }
                    for script_arg in script_args[script.args_from:script.args_to]:
                        alwayson_script_arg['args'].append(script_arg)
                    alwayson_scripts[script.name] = alwayson_script_arg
                script_index = script_args[0]
                if script_index != 0:
                    script_name = modules.scripts.scripts_txt2img.selectable_scripts[script_index - 1].name
                    args_from = modules.scripts.scripts_txt2img.selectable_scripts[script_index - 1].args_from
                    args_to = modules.scripts.scripts_txt2img.selectable_scripts[script_index - 1].args_to
                    script_args = script_args[args_from: args_to]

                payload = {
                    "enable_hr": enable_hr,
                    "denoising_strength": denoising_strength,
                    "hr_scale": hr_scale,
                    "hr_upscaler": hr_upscaler,
                    "hr_second_pass_steps": hr_second_pass_steps,
                    "hr_resize_x": hr_resize_x,
                    "hr_resize_y": hr_resize_y,
                    "hr_sampler_name": modules.sd_samplers.samplers_for_img2img[hr_sampler_index - 1].name if hr_sampler_index != 0 else None,
                    "hr_prompt": hr_prompt,
                    "hr_negative_prompt": hr_negative_prompt,
                    "prompt": txt2img_prompt,
                    "negative_prompt": txt2img_negative_prompt,
                    "prompt_styles": txt2img_prompt_styles,
                    "steps": steps,
                    "sampler_index": sampler_index,
                    "restore_faces": restore_faces,
                    "tiling": tiling,
                    "n_iter": batch_count,
                    "batch_size": batch_size,
                    "cfg_scale": cfg_scale,
                    "seed": seed,
                    "subseed": subseed,
                    "subseed_strength": subseed_strength,
                    "seed_resize_from_h": seed_resize_from_h,
                    "seed_resize_from_w": seed_resize_from_w,
                    "seed_enable_extras": seed_enable_extras,
                    "height": height,
                    "width": width,
                    "override_settings_texts": override_settings,
                    "eta": opts.eta_ddim if sampler_index == 'DDIM' or sampler_index == 'PLMS' else opts.eta_ancestral,
                    "s_churn": opts.s_churn,
                    "s_tmax": None,
                    "s_tmin": opts.s_tmin,
                    "s_noise": opts.s_noise,
                    "override_settings": override_settings,
                    "script_args": script_args,
                    "alwayson_scripts": alwayson_scripts
                }
                if script_name:
                    payload['script_name'] = script_name

                inputs = {
                    'task': task,
                    'txt2img_payload': payload,
                    'username': username
                }
            else:
                mode = args[0]
                prompt = args[1]
                negative_prompt = args[2]
                prompt_styles = args[3]
                init_img = args[4]
                sketch = args[5]
                init_img_with_mask = args[6]
                inpaint_color_sketch = args[7]
                inpaint_color_sketch_orig = args[8]
                init_img_inpaint = args[9]
                init_mask_inpaint = args[10]
                steps = args[11]
                sampler_index = modules.sd_samplers.samplers_for_img2img[args[12]].name
                mask_blur = args[13]
                mask_alpha = args[14]
                inpainting_fill = args[15]
                restore_faces = args[16]
                tiling = args[17]
                n_iter = args[18]
                batch_size = args[19]
                cfg_scale = args[20]
                image_cfg_scale = args[21]
                denoising_strength = args[22]
                seed = args[23]
                subseed = args[24]
                subseed_strength = args[25]
                seed_resize_from_h = args[26]
                seed_resize_from_w = args[27]
                seed_enable_extras = args[28]
                selected_scale_tab = args[29]
                height = args[30]
                width = args[31]
                scale_by = args[32]
                resize_mode = args[33]
                inpaint_full_res = args[34]
                inpaint_full_res_padding = args[35]
                inpainting_mask_invert = args[36]
                img2img_batch_input_dir = args[37]
                img2img_batch_output_dir = args[38]
                img2img_batch_inpaint_mask_dir = args[39]
                override_settings = args[40]
                alwayson_scripts = {}
                script_name = None
                script_index = None

                script_args = []
                for i in range(41, len(args)):
                    if(isinstance(args[i], dict)):
                        script_arg = {}
                        for key in args[i]:
                            if key == 'image' or key == 'mask':
                                script_arg[key] = encode_image_to_base64(Image.fromarray(args[i][key]))
                            else:
                                script_arg[key] = args[i][key]
                        script_args.append(script_arg)
                    elif hasattr(args[i], '__dict__'):
                        script_arg = {}
                        for key in args[i].__dict__:
                            if key == 'image':
                                script_arg[key] = {}
                                if args[i].__dict__[key]:
                                    if 'image' in args[i].__dict__[key]:
                                        script_arg[key]['image'] = encode_image_to_base64(Image.fromarray(args[i].__dict__[key]['image']))
                                    if 'mask' in args[i].__dict__[key]:
                                        script_arg[key]['mask'] = encode_image_to_base64(Image.fromarray(args[i].__dict__[key]['mask']))
                                else:
                                    script_arg[key] = None
                            else:
                                script_arg[key] = args[i].__dict__[key]
                        script_args.append(script_arg)
                    else:
                        script_args.append(args[i])

                if mode == 0:
                    image = init_img.convert("RGB")
                    mask = None
                elif mode == 1:  # img2img sketch
                    image = sketch.convert("RGB")
                    mask = None
                elif mode == 2:  # inpaint
                    image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
                    alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
                    mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
                    image = image.convert("RGB")
                elif mode == 3:  # inpaint sketch
                    image = inpaint_color_sketch
                    orig = inpaint_color_sketch_orig or inpaint_color_sketch
                    pred = np.any(np.array(image) != np.array(orig), axis=-1)
                    mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
                    mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
                    blur = ImageFilter.GaussianBlur(mask_blur)
                    image = Image.composite(image.filter(blur), orig, mask.filter(blur))
                    image = image.convert("RGB")
                elif mode == 4:  # inpaint upload mask
                    image = init_img_inpaint
                    mask = init_mask_inpaint
                else:
                    image = None
                    mask = None

                # Use the EXIF orientation of photos taken by smartphones.
                if image is not None:
                    image = ImageOps.exif_transpose(image)

                assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

                if selected_scale_tab == 1:
                    assert image, "Can't scale by because no image is selected"

                    width = int(image.width * scale_by)
                    height = int(image.height * scale_by)

                image_encoded_in_base64 = encode_image_to_base64(image) if image else None
                mask_encoded_in_base64 = encode_image_to_base64(mask) if mask else None

                for script in modules.scripts.scripts_img2img.alwayson_scripts:
                    alwayson_script_arg = {
                        'args': []
                    }
                    for script_arg in script_args[script.args_from:script.args_to]:
                        alwayson_script_arg['args'].append(script_arg)
                    alwayson_scripts[script.name] = alwayson_script_arg

                script_index = script_args[0]
                if script_index != 0:
                    script_name = modules.scripts.scripts_img2img.selectable_scripts[script_index - 1].name
                    args_from = modules.scripts.scripts_img2img.selectable_scripts[script_index - 1].args_from
                    args_to = modules.scripts.scripts_img2img.selectable_scripts[script_index - 1].args_to
                    script_args = script_args[args_from: args_to]

                script_index = script_args[0]
                if script_index != 0:
                    script_name = modules.scripts.scripts_img2img.selectable_scripts[script_index - 1].name

                payload = {
                    "init_images": [image_encoded_in_base64],
                    "resize_mode": resize_mode,
                    "denoising_strength": denoising_strength,
                    "image_cfg_scale": image_cfg_scale,
                    "mask": mask_encoded_in_base64,
                    "mask_blur": mask_blur,
                    "inpainting_fill": inpainting_fill,
                    "inpaint_full_res": inpaint_full_res,
                    "inpaint_full_res_padding": inpaint_full_res_padding,
                    "inpainting_mask_invert": inpainting_mask_invert,
                    "prompt": prompt,
                    "styles": prompt_styles,
                    "seed": seed,
                    "subseed": subseed,
                    "subseed_strength": subseed_strength,
                    "seed_resize_from_h": seed_resize_from_h,
                    "seed_resize_from_w": seed_resize_from_w,
                    "seed_enable_extras": seed_enable_extras,
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
                    "override_settings": override_settings,
                    "include_init_images": False,
                    "script_args": script_args,
                    "alwayson_scripts": alwayson_scripts
                }
                if mode == 5:
                    payload['img2img_batch_input_dir'] = img2img_batch_input_dir
                    payload['img2img_batch_output_dir'] = img2img_batch_output_dir
                    payload['img2img_batch_inpaint_mask_dir'] = img2img_batch_inpaint_mask_dir

                if script_name:
                    payload['script_name'] = script_name

                inputs = {
                    'task': task,
                    'img2img_payload': payload,
                    'username': username
                }

            params = {
                'endpoint_name': sagemaker_endpoint
            }
            response = requests.post(url=f'{shared.api_endpoint}/inference', params=params, json=inputs)
            if infer_type == 'async':
                processed = handle_sagemaker_inference_async(response)
            else:
                processed = json.loads(response.text)

            if processed == None:
                return [], "", ""

            images = []
            if 'error' not in processed:
                for image in processed['images']:
                    images.append(Image.open(io.BytesIO(base64.b64decode(image))))
                parameters = processed['parameters']
                info = json.loads(processed['info'])
                print(parameters, info)

                return images, json.dumps(info), modules.ui.plaintext_to_html('\n'.join(info['infotexts']))
            else:
                return images, "", processed['error']
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
            if infer_type == 'async':
                processed = handle_sagemaker_inference_async(response)
            else:
                processed = json.loads(response.text)

            images = []
            if 'error' not in processed:
                if task == 'extras-single-image':
                    images = [Image.open(io.BytesIO(base64.b64decode(processed['image'])))]
                else:
                    for image in processed['images']:
                        images.append(Image.open(io.BytesIO(base64.b64decode(image))))
                info = processed['html_info']
                print(info)
            else:
                info = processed['error']

            return images, "", info

    def f(username, *args, **kwargs):
        infer_type = os.environ.get('infer_type', 'async')
        # if the first argument is a string that says "task(...)", it is treated as a job id
        if len(args) > 0 and type(args[0]) == str and args[0][0:5] == "task(" and args[0][-1] == ")":
            id_task = args[0]
            progress.add_task_to_queue(id_task)
            args = args[1:]
        else:
            id_task = None

        if cmd_opts.pureui and func == modules.txt2img.txt2img:
            sagemaker_endpoint = args[len(args) -1]
            args = args[:-1]
            progress.start_task(id_task)
            res = sagemaker_inference('text-to-image', infer_type, username, sagemaker_endpoint, *args, **kwargs)
            progress.finish_task(id_task)
        elif cmd_opts.pureui and func == modules.img2img.img2img:
            sagemaker_endpoint = args[len(args) -1]
            args = args[:-1]
            progress.start_task(id_task)
            res = sagemaker_inference('image-to-image', infer_type, username, sagemaker_endpoint, *args, **kwargs)
            progress.finish_task(id_task)
        elif cmd_opts.pureui and func == modules.postprocessing.run_postprocessing:
            sagemaker_endpoint = args[len(args) -1]
            args = args[:-1]
            progress.start_task(id_task)
            res = sagemaker_inference('extras', infer_type, username, sagemaker_endpoint, *args, **kwargs)
            progress.finish_task(id_task)
        else:
            with queue_lock:
                shared.state.begin()
                progress.start_task(id_task)

                try:
                    res = func(*args, **kwargs)
                    progress.record_results(id_task, res)
                finally:
                    progress.finish_task(id_task)

                shared.state.end()

        return res

    return wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True)


def wrap_gradio_call(func, extra_outputs=None, add_stats=False):
    def f(request: gr.Request, *args, extra_outputs_array=extra_outputs, **kwargs):
        username = shared.get_webui_username(request)

        run_memmon = shared.opts.memmon_poll_rate > 0 and not shared.mem_mon.disabled and add_stats
        if run_memmon:
            shared.mem_mon.monitor()
        t = time.perf_counter()

        try:
            if func.__name__ == 'f' or func.__name__ == 'run_settings' or func.__name__ == 'save_files':
                res = list(func(username, *args, **kwargs))
            else:
                res = list(func(*args, **kwargs))
        except Exception as e:
            # When printing out our debug argument list, do not print out more than a MB of text
            max_debug_str_len = 131072 # (1024*1024)/8

            print("Error completing request", file=sys.stderr)
            argStr = f"Arguments: {args} {kwargs}"
            print(argStr[:max_debug_str_len], file=sys.stderr)
            if len(argStr) > max_debug_str_len:
                print(f"(Argument list truncated at {max_debug_str_len}/{len(argStr)} characters)", file=sys.stderr)

            print(traceback.format_exc(), file=sys.stderr)

            shared.state.job = ""
            shared.state.job_count = 0

            if extra_outputs_array is None:
                extra_outputs_array = [None, '']

            error_message = f'{type(e).__name__}: {e}'
            res = extra_outputs_array + [f"<div class='error'>{html.escape(error_message)}</div>"]

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

