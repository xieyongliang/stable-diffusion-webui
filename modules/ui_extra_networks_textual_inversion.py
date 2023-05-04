import json
import os
import gradio as gr
from modules import paths

from modules import ui_extra_networks, sd_hijack, shared


class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Textual Inversion')
        self.allow_negative_prompt = True

    def refresh(self, sagemaker_endpoint, request: gr.Request):
        shared.reload_embeddings(request)

    def list_items(self):
        for name, filename in shared.embeddings.items():
            path, ext = os.path.splitext(filename)
            yield {
                "name": name,
                "filename": filename,
                "preview": self.find_preview(path),
                "description": self.find_description(path),
                "search_term": self.search_terms_from_path(filename),
                "prompt": json.dumps(name),
                "local_preview": f"{path}.preview.{shared.opts.samples_format}",
            }

    def allowed_directories_for_previews(self):
        return [os.path.join(paths.models_path, 'embeddings')]
