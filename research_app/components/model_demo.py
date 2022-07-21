import gradio as gr
import lightning as L
from lightning.app import BuildConfig
from lightning.app.components.serve import ServeGradio


class CustomBuildConfig(BuildConfig):
    def build_commands(self):
        return [
            "cd OFA && pip install -r requirements.txt && cd ../",
        ]


class ModelDemo(ServeGradio):
    """Serve model with Gradio UI.

    You need to define i. `build_model` and ii. `predict` method and Lightning `ServeGradio` component will
    automatically launch the Gradio interface.
    """

    inputs = [gr.inputs.Image(type="pil"), "textbox"]
    outputs = [gr.outputs.Image(type="numpy"), "text"]
    enable_queue = False
    examples = [
        ["test.jpeg", "what color is the left car?"],
        ["test.jpeg", 'which region does the text " a grey car " describe?'],
    ]

    def __init__(self):
        super().__init__(
            parallel=True, cloud_build_config=CustomBuildConfig(), cloud_compute=L.CloudCompute("cpu-medium"),
        )

    def build_model(self):
        import sys, os
        sys.path.append("OFA")
        os.system("export PYTHONPATH=OFA")
        from OFA.gradio_app import general_interface

        return general_interface

    def predict(self, image, instruction):
        print(instruction)
        return self.model(image, instruction)
