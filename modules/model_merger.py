from __future__ import annotations
from datetime import datetime, timedelta
import pytz
import json
import gradio as gr
import os
import re
import requests
import sys
import threading

from modules import shared

input_chkpt_s3uri = ''
s3_checkpoints = []
s3_uri_pattern = re.compile(r"^s3://[\w\-\.]+/[\w\-\.\/]+$")

job_rwlock = threading.RLock()
processing_jobs = {}
last_processing_output_msg = ''

def get_processing_jobs():
    global job_rwlock
    global processing_jobs

    copy = {}
    with job_rwlock:
        copy = processing_jobs.copy()
    return copy

def add_processing_job(job_name, output_loc):
    global job_rwlock
    global processing_jobs

    with job_rwlock:
        processing_jobs[job_name] = output_loc

def delete_processing_job(job_name):
    global job_rwlock
    global processing_jobs

    with job_rwlock:
        if job_name in processing_jobs:
            del processing_jobs[job_name]

def get_last_processing_output_message():
    global job_rwlock
    global last_processing_output_msg
    
    last_msg = ''
    with job_rwlock:
        last_msg = last_processing_output_msg
    return last_msg

def set_last_processing_output_message(msg):
    global job_rwlock
    global last_processing_output_msg
    
    with job_rwlock:
        last_processing_output_msg = msg

time_fmt = '%Y-%m-%d-%H-%M-%S-UTC'
job_fmt = f'model-merge-{time_fmt}'

def uniq_job_name():
    # Valid job name must start with a letter or number ([a-zA-Z0-9]) and can contain up to 63 characters, including hyphens (-).
    global time_fmt
    global job_fmt
    import pytz

    now_utc = datetime.now(pytz.utc)
    current_time_str = now_utc.strftime(time_fmt)
    job_name = f'model-merge-{current_time_str}'
    return job_name

def get_job_elapsed_time(job_name):
    global job_fmt

    timestamp_utc = None
    try:
        timestamp_utc = datetime.strptime(job_name, job_fmt).replace(tzinfo=pytz.utc)
    except ValueError:
        print(f"Error: input string {job_name} does not match format: {job_fmt}.")
    
    if timestamp_utc is None:
        return None

    now_utc = datetime.now(pytz.utc)
    time_diff = now_utc - timestamp_utc
    return time_diff

def readable_time_diff(time_diff):
    total_seconds = int(time_diff.total_seconds())

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        time_str = f"{hours} hours, {minutes} minutes, {seconds} seconds"
    elif minutes > 0:
        time_str = f"{minutes} minutes, {seconds} seconds"
    else:
        time_str = f"{seconds} seconds"

    return time_str

def is_valid_s3_uri(s3_uri):
    global s3_uri_pattern
    match = s3_uri_pattern.match(s3_uri)
    return bool(match)

def load_checkpoints_from_s3_uri(s3_uri, primary_component,
                                 secondary_component, tertiary_component):
    global input_chkpt_s3uri
    global s3_checkpoints

    if not is_valid_s3_uri(s3_uri):
        return

    input_chkpt_s3uri = s3_uri.rstrip('/')

    s3_checkpoints.clear()

    params = {
        's3uri': input_chkpt_s3uri,
        'exclude_filters': 'yaml',
    }
    response = requests.get(url=f'{shared.api_endpoint}/s3', params = params)
    if response.status_code != 200:
        return

    text = json.loads(response.text)
    for obj in text['payload']:
        obj_key = obj['key']
        ckpt = obj_key.split('/')[-1]
        s3_checkpoints.append(ckpt)

    return [gr.Dropdown.update(choices=s3_checkpoints) for _ in range(3)]

def get_checkpoints_to_merge():
    global s3_checkpoints
    return s3_checkpoints

def get_chkpt_name(checkpoint_file):
    name = os.path.basename(checkpoint_file)
    if name.startswith("\\") or name.startswith("/"):
        name = name[1:]

    chkpt_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
    return chkpt_name

def get_merged_chkpt_name(primary_model_name, secondary_model_name,
                          tertiary_model_name, multiplier, interp_method,
                          checkpoint_format, custom_name):
    filename = get_chkpt_name(primary_model_name) + '_' + \
            str(round(1-multiplier, 2)) + '-' + \
             get_chkpt_name(secondary_model_name) + '_' + \
             str(round(multiplier, 2)) + '-'

    if isinstance(tertiary_model_name, str) and tertiary_model_name != '':
        filename += get_chkpt_name(tertiary_model_name) + '-'
        
    filename += interp_method.replace(" ", "_") + '-merged.' +  checkpoint_format
    filename = filename if custom_name == '' else (custom_name + '.' + checkpoint_format)
    return filename

def get_processing_job_status():
    job_dict = get_processing_jobs()
    if len(job_dict) == 0:
        print("No jobs running yet.")
        return get_last_processing_output_message()
    
    ret_message = ''
    for job_name, job_output_loc in job_dict.items():
        inputs = {'job_name': job_name}
        response = requests.get(url=f'{shared.api_endpoint}/process', json=inputs)
        
        if response.status_code != 200:
            ret_message += f"Processing job {job_name}:\tjob status unknown\n"
            continue

        job_elapsed_time = get_job_elapsed_time(job_name) 
        job_elapsed_timestr = f"Time elapsed: {readable_time_diff(job_elapsed_time)}" \
            if job_elapsed_time is not None else ''
    
        text = json.loads(response.text)
        job_status = text['job_status']
        shall_delete = False
        if job_status == 'Completed':
            msg = f"finished successfully. Output: {job_output_loc}. {job_elapsed_timestr}"
            shall_delete = True
        elif job_status == 'Failed': 
            msg = f"failed: {text['failure_reason']}. {job_elapsed_timestr}"
            shall_delete = True
        else:
            msg = f"still in progress. {job_elapsed_timestr}"
            
        ret_message += f"Processing job {job_name}:\t{msg}\n"
        print(f"Processing job {job_name}: {msg}")

        if shall_delete or (job_elapsed_time and job_elapsed_time > timedelta(hours=1)):
            print(f"Romving processing job '{job_name}', job_staus: {job_status}. {job_elapsed_timestr}")
            delete_processing_job(job_name)

    if ret_message == '': 
        ret_message = get_last_processing_output_message()
    else:
        set_last_processing_output_message(ret_message)

    return ret_message

def get_default_output_model_s3uri():
    s3uri = shared.get_default_sagemaker_bucket() + \
                '/stable-diffusion-webui/models/Stable-diffusion'
    return s3uri

def run_modelmerger_remote(primary_model_name, secondary_model_name,
                           tertiary_model_name, interp_method, multiplier,
                           save_as_half, custom_name, checkpoint_format,
                           output_chkpt_s3uri, submit_result):
    """ This is the same as run_modelmerger, but it calls a RESTful API to do the job """
    if isinstance(primary_model_name, list) or \
        isinstance(secondary_model_name, list):
        ret_msg = "At least primary_model_name and secondary_model_name must be set."
        set_last_processing_output_message(ret_msg)
        return reg_msg

    if output_chkpt_s3uri != '' and not is_valid_s3_uri(output_chkpt_s3uri):
        ret_msg = f"output_chkpt_s3uri is not valid: {output_chkpt_s3uri}"
        set_last_processing_output_message(ret_msg)
        return reg_msg

    input_srcs = f"{input_chkpt_s3uri}/{primary_model_name}," + \
                 f"{input_chkpt_s3uri}/{secondary_model_name}"
    input_dsts = f"/opt/ml/processing/input/primary," + \
                 f"/opt/ml/processing/input/secondary"

    if is_valid_s3_uri(output_chkpt_s3uri):
        output_dst = output_chkpt_s3uri 
    else:
        output_dst = get_default_output_model_s3uri()
    output_name = get_merged_chkpt_name(primary_model_name, secondary_model_name,
                          tertiary_model_name, multiplier, interp_method,
                          checkpoint_format, custom_name)
    # Make an argument dict to be accessible in the process script  
    args = {
        "primary_model": primary_model_name,
        "secondary_model": secondary_model_name,
        "interp_method": interp_method,
        "multiplier": multiplier,
        "save_as_half": save_as_half,
        "checkpoint_format": checkpoint_format,
        'output_destination': output_dst,
        'output_name': output_name,
    }

    if custom_name != '':
        args["custom_name"] = custom_name

    if isinstance(tertiary_model_name, str) and tertiary_model_name != '':
        input_srcs += f",{input_chkpt_s3uri}/{tertiary_model_name}"
        input_dsts += f",/opt/ml/processing/input/tertiary"
        args["tertiary_model"] = tertiary_model_name

    inputs = {
        'instance_type': 'ml.m5.4xlarge', # Memory intensive
        'instance_count': 1,
        'process_script': 'process_checkpoint_merge.py',
        'input_sources': input_srcs,
        'input_destination': input_dsts,
        'output_sources': '/opt/ml/processing/output',
        'output_destination': output_dst,
        'output_name': output_name, 
        'job_name': uniq_job_name(),
        'arguments': args
    }

    response = requests.post(url=f'{shared.api_endpoint}/process', json=inputs)
    if response.status_code != 200:
        ret_msg = f"Failed to run model merge process job: {response.text}"
        set_last_processing_output_message(ret_msg)
        return ret_msg 

    text = json.loads(response.text)
    job_name = text['job_name']

    # Add the job to the list for later status poll
    add_processing_job(job_name, f"{output_dst}/{output_name}")

    ret_msg = f"Merging models in Sagemaker Processing Job...\nJob Name: {job_name}"
    set_last_processing_output_message(ret_msg)

    return ret_msg
