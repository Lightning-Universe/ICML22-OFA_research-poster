# ⚡️ OFA: Research Poster Template 🔬

Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework

This app is a research poster demo of [OFA paper](https://arxiv.org/abs/2202.03052). It showcasese the paper, blog, notebook, and model demo where
you can upload an image and do prediction based on text query. To create a research poster for your work use the Lightning Research
Template app.


## Getting started

To create a Research Poster you can install this app via the [Lightning CLI](https://lightning.ai/lightning-docs/) or
[use the template](https://docs.github.com/en/articles/creating-a-repository-from-a-template) from GitHub and
manually install the app as mentioned below.

### Installation

#### With Lightning CLI

`lightning install app lightning/icml22-ofa`

#### Use GitHub

You can clone the forked app repo and follow the steps below to install the app.

```
git clone https://github.com/lightning-AI/LAI-icml22-ofa-research-poster.git
cd LAI-icml22-ofa-research-poster
pip install -r requirements.txt
pip install -e .

# install OFA dependencies
cd OFA
pip install -r requirements.txt
```

Once you have installed the app, you can goto the `LAI-icml22-ofa-research-poster` folder and
run `lightning run app app.py --cloud` from terminal.
This will launch the template app in your default browser with tabs containing research paper, blog, Training
logs, and Model Demo.

You should see something like this in your browser:

> ![image](./assets/demo.png)

You can modify the content of this app and customize it to your research.
At the root of this template, you will find [app.py](./app.py) that contains the `ResearchApp` class. This class
provides arguments like a link to a paper, a blog, and whether to launch a Gradio demo. You can read more about what
each of the arguments does in the docstrings.

### Highlights

- Provide the link for paper, blog, or training logger like WandB as an argument, and `ResearchApp` will create a tab
  for each.
- Make a poster for your research by editing the markdown file in the [resources](./resources/poster.md) folder.
- Add interactive model demo with Gradio app, update the gradio component present in the \[research_app (
  ./research_app/components/model_demo.py) folder.
- View a Jupyter Notebook or launch a fully-fledged notebook instance (Sharing a Jupyter Notebook instance can expose
  the cloud instance to security vulnerability.)
- Reorder the tab layout using the `tab_order` argument.

### Example

```python
# update app.py at the root of the repo
import lightning as L

poster_dir = "resources"
paper = "https://arxiv.org/abs/2202.03052"
github = "https://github.com/OFA-Sys/OFA"
notebook_path = "resources/OFA.ipynb"
tabs = ["Poster", "model demo", "Notebook viewer", "Paper"]

app = L.LightningApp(
    ResearchApp(
        paper=paper,
        poster_dir=poster_dir,
        notebook_path=notebook_path,
        launch_gradio=True,
        launch_jupyter_lab=False,  # don't launch for public app, can expose to security vulnerability
    )
)
```

## FAQs

1. How to pull from the latest template
   code? [Answer](https://stackoverflow.com/questions/56577184/github-pull-changes-from-a-template-repository)
