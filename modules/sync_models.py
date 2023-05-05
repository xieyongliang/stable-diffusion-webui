import os,threading,psutil,json,time
import boto3
import modules.shared as shared
import modules.sd_models as sd_models
import modules.script_callbacks as script_callbacks
from modules.shared import syncLock

FREESPACE = 20
def check_space_s3_download(s3_client, bucket_name, s3_folder, local_folder, file, size, mode):
    print(f"bucket_name:{bucket_name},s3_folder:{s3_folder},file:{file}")
    if file == '' or None:
        print('Debug log:file is empty, return')
        return True
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
            hash = sd_models.model_hash(dist)
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

def free_local_disk(local_folder, size,mode):
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

def list_s3_objects(s3_client, bucket_name, prefix=''):
    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    # iterate over pages
    for page in page_iterator:
        # loop through objects in page
        if 'Contents' in page:
            for obj in page['Contents']:
                _, ext = os.path.splitext(obj['Key'].lstrip('/'))
                if ext in ['.pt', '.pth', '.ckpt', '.safetensors','.yaml']:
                    objects.append(obj)
        # if there are more pages to fetch, continue
        if 'NextContinuationToken' in page:
            page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix,
                                                ContinuationToken=page['NextContinuationToken'])
    return objects


def initial_s3_download(s3_client, s3_folder, local_folder,cache_dir,mode):
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
    # List all objects in the S3 folder
    s3_objects = list_s3_objects(s3_client=s3_client, bucket_name=shared.models_s3_bucket, prefix=s3_folder)
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
    for obj in s3_objects:
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
                check_space_s3_download(s3_client,shared.models_s3_bucket, s3_folder,local_folder, fname, tmp_s3_files.get(fname)[1], mode)
        except Exception as e:
            print(e)

    print(f'-----s3_files---{s3_files}')
    # save the lastest one
    with open(s3_file_name, "w") as f:
        json.dump(s3_files, f)
    
def sync_s3_folder(local_folder, cache_dir,mode):
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
            if mode == 'sd':
                #Refreshing Model List
                sd_models.list_models()
            # cn models sync not supported temporally due to an unfixed bug
            elif mode == 'cn':
                script_callbacks.update_cn_models_callback()
            elif mode == 'lora':
                print('Nothing To do')

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