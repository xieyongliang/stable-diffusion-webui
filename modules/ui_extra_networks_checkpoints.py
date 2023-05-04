import html
import json
import os
import gradio as gr

from modules import shared, ui_extra_networks, sd_models


class ExtraNetworksPageCheckpoints(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Checkpoints')

    def refresh(self, sagemaker_endpoint, request: gr.Request):
        username = shared.get_webui_username(request)
        shared.refresh_checkpoints(sagemaker_endpoint, username)

    def list_items(self):
        checkpoint: sd_models.CheckpointInfo
        for name, checkpoint in sd_models.checkpoints_list.items():
            path, ext = os.path.splitext(checkpoint.filename)
            yield {
                "name": os.path.splitext(os.path.basename(checkpoint.filename))[0],
                "filename": path,
                "preview": self.find_preview(path),
                "description": self.find_description(path),
                "search_term": self.search_terms_from_path(checkpoint.filename) + " " + (checkpoint.hash or ""),
                "onclick": '"' + html.escape(f"""return selectCheckpoint({json.dumps(name)})""") + '"',
                "local_preview": f"{path}.{shared.opts.samples_format}",
            }

    def allowed_directories_for_previews(self):
        return [v for v in [shared.cmd_opts.ckpt_dir, sd_models.model_path] if v is not None]
