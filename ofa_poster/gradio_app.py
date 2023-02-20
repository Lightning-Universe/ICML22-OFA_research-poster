import os
import urllib

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from fairseq import checkpoint_utils
from fairseq import options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from tasks.mm_tasks.refcoco import RefcocoTask

# Register
tasks.register_task("refcoco", RefcocoTask)

# turn on cuda if GPU is available
use_cuda = False  # torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = True if use_cuda else False

# specify some options for evaluation
CKPT_PATH = "checkpoints/ofa_large_clean.pt"
parser = options.get_generation_parser()
input_args = ["", "--task=refcoco", "--beam=10", f"--path={CKPT_PATH}", "--bpe-dir=OFA/utils/BPE"]
args = options.parse_args_and_arch(parser, input_args)
cfg = convert_namespace_to_omegaconf(args)

# Download checkpoints
if not os.path.exists(CKPT_PATH):
    print("Downloading model, this might take some time!")
    os.makedirs("checkpoints", exist_ok=True)
    urllib.request.urlretrieve(
        "https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/ofa_large_clean.pt",
        CKPT_PATH,
    )

# Load pretrained ckpt & config
task = tasks.setup_task(cfg.task)
models, cfg = checkpoint_utils.load_model_ensemble(utils.split_paths(cfg.common_eval.path), task=task)

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose(
    [
        lambda image: image.convert("RGB"),
        transforms.Resize((task.cfg.patch_image_size, task.cfg.patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    token_result = []
    bin_result = []
    img_result = []
    for token in x.strip().split():
        if token.startswith("<bin_"):
            bin_result.append(token)
        elif token.startswith("<code_"):
            img_result.append(token)
        else:
            if bpe is not None:
                token = bpe.decode(f"{token}")
            if tokenizer is not None:
                token = tokenizer.decode(token)
            if token.startswith(" ") or len(token_result) == 0:
                token_result.append(token.strip())
            else:
                token_result[-1] += token

    return " ".join(token_result), " ".join(bin_result), " ".join(img_result)


def coord2bin(coords, w_resize_ratio, h_resize_ratio):
    coord_list = [float(coord) for coord in coords.strip().split()]
    bin_list = []
    bin_list += [f"<bin_{int((coord_list[0] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1)))}>"]
    bin_list += [f"<bin_{int((coord_list[1] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1)))}>"]
    bin_list += [f"<bin_{int((coord_list[2] * w_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1)))}>"]
    bin_list += [f"<bin_{int((coord_list[3] * h_resize_ratio / task.cfg.max_image_size * (task.cfg.num_bins - 1)))}>"]
    return " ".join(bin_list)


def bin2coord(bins, w_resize_ratio, h_resize_ratio):
    bin_list = [int(bin[5:-1]) for bin in bins.strip().split()]
    coord_list = []
    coord_list += [bin_list[0] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[1] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    coord_list += [bin_list[2] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / w_resize_ratio]
    coord_list += [bin_list[3] / (task.cfg.num_bins - 1) * task.cfg.max_image_size / h_resize_ratio]
    return coord_list


def encode_text(text, length=None, append_bos=False, append_eos=False):
    line = [
        task.bpe.encode(f" {word.strip()}") if not word.startswith("<code_") and not word.startswith("<bin_") else word
        for word in text.strip().split()
    ]
    line = " ".join(line)
    s = task.tgt_dict.encode_line(line=line, add_if_not_exist=False, append_eos=False).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


def construct_sample(image: Image, instruction: str):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])

    instruction = encode_text(f" {instruction.lower().strip()}", append_bos=True, append_eos=True).unsqueeze(0)
    instruction_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in instruction])
    sample = {
        "id": np.array(["42"]),
        "net_input": {
            "src_tokens": instruction,
            "src_lengths": instruction_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        },
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def general_interface(image, instruction):
    # Construct input sample & preprocess for GPU if cuda available
    sample = construct_sample(image, instruction)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    # Generate result
    with torch.no_grad():
        hypos = task.inference_step(generator, models, sample)
        tokens, bins, imgs = decode_fn(hypos[0][0]["tokens"], task.tgt_dict, task.bpe, generator)

    print(tokens)
    print(bins)
    print(imgs)
    if bins.strip() != "":
        w, h = image.size
        w_resize_ratio = task.cfg.patch_image_size / w
        h_resize_ratio = task.cfg.patch_image_size / h
        img = np.asarray(image)
        coord_list = bin2coord(bins, w_resize_ratio, h_resize_ratio)
        cv2.rectangle(
            img, (int(coord_list[0]), int(coord_list[1])), (int(coord_list[2]), int(coord_list[3])), (0, 255, 0), 3
        )
        return img, None
    else:
        return None, tokens


if __name__ == "__main__":
    title = "OFA-Generic_Interface"
    description = (
        "Gradio Demo for OFA-Generic_Interface."
        " You can use different instructions to perform various tasks (i.e., image captioning, visual grounding,"
        " VQA and grounded captioning) with just one model."
        " Upload your own image or click any one of the examples, and write a proper instruction."
        ' Then click "Submit" and wait for the result.'
    )
    article = (
        "<p style='text-align: center'><a href='https://github.com/OFA-Sys/OFA' target='_blank'>OFA Github "
        "Repo</a></p> "
    )
    examples = [
        ["test.jpeg", "what color is the left car?"],
        ["test.jpeg", 'which region does the text " a grey car " describe?'],
    ]
    io = gr.Interface(
        fn=general_interface,
        inputs=[gr.inputs.Image(type="pil"), "textbox"],
        outputs=[gr.outputs.Image(type="numpy"), "text"],
        title=title,
        description=description,
        article=article,
        examples=examples,
        cache_examples=False,
    )
    io.launch(debug=True)
