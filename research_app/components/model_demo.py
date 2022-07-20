from loguru import logger
import os
import gradio as gr
from lightning.app.components.serve import ServeGradio

from lightning.app import BuildConfig


class CustomBuildConfig(BuildConfig):
    def build_commands(self):
        return [
            "cd research_app/components && git clone https://github.com/aniketmaurya/OFA.git",
            "cd research_app/components && pip install ."
        ]

class ModelDemo(ServeGradio):
    """Serve model with Gradio UI.

    You need to define i. `build_model` and ii. `predict` method and Lightning `ServeGradio` component will
    automatically launch the Gradio interface.
    """
    inputs=[gr.inputs.Image(type='pil'), "textbox"]
    outputs=[gr.outputs.Image(type='numpy'), 'text']
    enable_queue = True
    examples = [['test.jpeg', 'what color is the left car?'],
                ['test.jpeg', 'which region does the text " a grey car " describe?']]

    def __init__(self):
        super().__init__(parallel=True, cloud_build_config=CustomBuildConfig())

    def build_model(self):
        import os
        os.chdir("research_app/components/OFA")
        from .OFA.gradio_app import general_interface
        
        return general_interface
        
    def predict(self, image, instruction):
        return self.model(image, instruction)
