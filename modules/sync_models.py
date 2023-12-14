import os
import threading
import psutil
import json
import time
import hashlib

class ModelSync:
    def __init__(self, s3_client, s3_bucket, s3_folder, local_folder, queue_lock, model_hash, sync_lock, cache_dir, refresh_callback=None, available_freespace=20):
        self.models_ref = {}
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.s3_folder = s3_folder
        self.local_folder = local_folder
        self.queue_lock = queue_lock
        self.cache_dir = cache_dir
        self.sync_lock = sync_lock
        self.model_hash = model_hash
        self.refresh_callback = refresh_callback
        self.available_freespace = available_freespace

        thread = threading.Thread(target=self.sync_thread)
        thread.start()
        print (f's3://{self.s3_bucket}/{self.s3_folder} sync thread start')
        return thread

    def get_models_ref_dict(self):
        return self.models_ref

    def add_models_ref(self, model_name):
        if model_name in self.models_ref:
            self.models_ref[model_name] += 1
        else:
            self.models_ref[model_name] = 0

    def remove_model_ref(self,model_name):
        if self.models_ref.get(model_name):
            del self.models_ref[model_name]

    def get_models_ref(self, model_name):
        return self.models_ref.get(model_name)

    def get_least_ref_model(self):
        sorted_models = sorted(self.models_ref.items(), key=lambda item: item[1])
        if sorted_models:
            least_ref_model, least_counter = sorted_models[0]
            return least_ref_model,least_counter
        else:
            return None,None

    def pop_least_ref_model(self):
        sorted_models = sorted(self.models_ref.items(), key=lambda item: item[1])
        if sorted_models:
            least_ref_model, least_counter = sorted_models[0]
            del self.models_ref[least_ref_model]
            return least_ref_model,least_counter
        else:
            return None,None

    def free_local_disk(self, size):
        disk_usage = psutil.disk_usage('/tmp')
        freespace = disk_usage.free/(1024**3)
        if freespace - size >= self.available_freespace:
            return
        models_Ref = None
        model_name,ref_cnt  = models_Ref.get_least_ref_model()
        if model_name and ref_cnt:
            filename = model_name[:model_name.rfind("[")]
            os.remove(os.path.join(self.local_folder, filename))
            disk_usage = psutil.disk_usage('/tmp')
            freespace = disk_usage.free/(1024**3)
            print(f"Remove file: {os.path.join(self.local_folder, filename)} now left space:{freespace}")
        else:
            ## if ref_cnt == 0, then delete the oldest zero_ref one
            zero_ref_models = set([model[:model.rfind(" [")] for model, count in models_Ref.get_models_ref_dict().items() if count == 0])
            local_files = set(os.listdir(self.local_folder))
            # join with local
            files = [(os.path.join(self.local_folder, file), os.path.getctime(os.path.join(self.local_folder, file))) for file in zero_ref_models.intersection(local_files)]
            if len(files) == 0:
                print(f"No files to remove in folder: {self.local_folder}, please remove some files in S3 bucket")
                return
            files.sort(key=lambda x: x[1])
            oldest_file = files[0][0]
            os.remove(oldest_file)
            disk_usage = psutil.disk_usage('/tmp')
            freespace = disk_usage.free/(1024**3)
            print(f"Remove file: {oldest_file} now left space:{freespace}")
            filename = os.path.basename(oldest_file)

    def check_space_s3_download(self, file, size):
        if file == '' or None:
            print('Debug log:file is empty, return')
            return True
        src = self.s3_folder + '/' + file
        dist =  os.path.join(self.local_folder, file)
        os.makedirs(os.path.dirname(dist), exist_ok=True)
        # Get disk usage statistics
        disk_usage = psutil.disk_usage('/tmp')
        freespace = disk_usage.free/(1024**3)
        print(f"Total space: {disk_usage.total/(1024**3)}, Used space: {disk_usage.used/(1024**3)}, Free space: {freespace}")
        if freespace - size >= self.available_freespace:
            try:
                self.s3_client.download_file(self.s3_bucket, src, dist)
                #init ref cnt to 0, when the model file first time download
                hash = self.model_hash(dist)
                self.model_ref.add_models_ref('{0} [{1}]'.format(file, hash))
                print(f'download_file success:from {self.s3_bucket}/{src} to {dist}')
            except Exception as e:
                print(f'download_file error: from {self.s3_bucket}/{src} to {dist}')
                print(f"An error occurred: {e}")
                return False
            return True
        else:
            return False

    def list_s3_objects(self):
        objects = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_folder)
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
                page_iterator = paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_folder,
                                                    ContinuationToken=page['NextContinuationToken'])
        return objects

    def sync(self):
        # Check and Create tmp folders
        md = hashlib.md5()
        md.update(f's3://{self.s3_bucket}/{self.s3_folder}'.encode('utf-8'))
        etag_cache = md.hexdigest()
        os.makedirs(os.path.dirname(self.local_folder), exist_ok=True)
        os.makedirs(os.path.dirname(self.cache_dir), exist_ok=True)
        s3_file_name = os.path.join(self.cache_dir, etag_cache)
        # Create an empty file if not exist
        if os.path.isfile(s3_file_name) == False:
            s3_files = {}
            with open(s3_file_name, "w") as f:
                json.dump(s3_files, f)

        # List all objects in the S3 folder
        s3_objects = self.list_s3_objects(s3_client=self.s3_client, bucket_name=self.s3_bucket, prefix=self.get_models_ref_dicts3_folder)
        # Check if there are any new or deleted files
        s3_files = {}
        for obj in s3_objects:
            etag = obj['ETag'].strip('"').strip("'")   
            size = obj['Size']/(1024**3)
            key = obj['Key'].replace(self.s3_folder, '').lstrip('/')
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
            if os.path.isfile(os.path.join(self.local_folder, file)):
                os.remove(os.path.join(self.local_folder, file))
                print(f'remove file {os.path.join(self.local_folder, file)}')
        # Add new files 
        for file in new_files.union(mod_files):
            registerflag = True
            retry = 3 ##retry limit times to prevent dead loop in case other folders is empty
            while retry:
                ret = self.check_space_s3_download(self, file, s3_files[file][1])
                #if the space is not enough free
                if ret:
                    retry = 0
                else:
                    self.free_local_disk(self.local_folder,s3_files[file][1])
                    retry = retry - 1
        if registerflag:
            if self.callback:
                #Refreshing Model List
                with self.queue_lock:
                    self.callback()

    # Create a thread function to keep syncing with the S3 folder
    def sync_thread(self):
        while True:
            self.sync_lock.acquire()
            self.sync()
            self.sync_lock.release()
            time.sleep(30)
