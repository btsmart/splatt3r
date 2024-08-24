#!/usr/bin/env python3
# The MASt3R Gradio demo, modified for predicting 3D Gaussian Splats

# --- Original License ---
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import functools
import os
import sys
import tempfile

import gradio
import torch
from huggingface_hub import hf_hub_download

sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')
sys.path.append('src/pixelsplat_src')
from dust3r.utils.image import load_images
from mast3r.utils.misc import hash_md5
import main
import utils.export as export


def get_reconstructed_scene(outdir, model, device, silent, image_size, ios_mode, filelist):

    assert len(filelist) == 1 or len(filelist) == 2, "Please provide one or two images"
    if ios_mode:
        filelist = [f[0] for f in filelist]
    if len(filelist) == 1:
        filelist = [filelist[0], filelist[0]]
    imgs = load_images(filelist, size=image_size, verbose=not silent)

    for img in imgs:
        img['img'] = img['img'].to(device)
        img['original_img'] = img['original_img'].to(device)
        img['true_shape'] = torch.from_numpy(img['true_shape'])

    output = model(imgs[0], imgs[1])

    pred1, pred2 = output
    plyfile = os.path.join(outdir, 'gaussians.ply')
    export.save_as_ply(pred1, pred2, plyfile)
    return plyfile

if __name__ == '__main__':

    image_size = 512
    server_name = '127.0.0.1'
    server_port = None
    share = True
    silent = False
    ios_mode = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = "brandonsmart/splatt3r_v1.0"
    filename = "epoch=19-step=1200.ckpt"
    weights_path = hf_hub_download(repo_id=model_name, filename=filename)
    model = main.MAST3RGaussians.load_from_checkpoint(weights_path, device)
    chkpt_tag = hash_md5(weights_path)

    # Define example inputs and their corresponding precalculated outputs
    examples = [
        ["demo_examples/scannet++_1_img_1.jpg", "demo_examples/scannet++_1_img_2.jpg", "demo_examples/scannet++_1.ply"],
        ["demo_examples/scannet++_2_img_1.jpg", "demo_examples/scannet++_2_img_2.jpg", "demo_examples/scannet++_2.ply"],
        ["demo_examples/scannet++_3_img_1.jpg", "demo_examples/scannet++_3_img_2.jpg", "demo_examples/scannet++_3.ply"],
        ["demo_examples/scannet++_4_img_1.jpg", "demo_examples/scannet++_4_img_2.jpg", "demo_examples/scannet++_4.ply"],
        ["demo_examples/scannet++_5_img_1.jpg", "demo_examples/scannet++_5_img_2.jpg", "demo_examples/scannet++_5.ply"],
        ["demo_examples/scannet++_6_img_1.jpg", "demo_examples/scannet++_6_img_2.jpg", "demo_examples/scannet++_6.ply"],
        ["demo_examples/scannet++_7_img_1.jpg", "demo_examples/scannet++_7_img_2.jpg", "demo_examples/scannet++_7.ply"],
        ["demo_examples/scannet++_8_img_1.jpg", "demo_examples/scannet++_8_img_2.jpg", "demo_examples/scannet++_8.ply"],
        ["demo_examples/in_the_wild_1_img_1.jpg", "demo_examples/in_the_wild_1_img_2.jpg", "demo_examples/in_the_wild_1.ply"],
        ["demo_examples/in_the_wild_2_img_1.jpg", "demo_examples/in_the_wild_2_img_2.jpg", "demo_examples/in_the_wild_2.ply"],
        ["demo_examples/in_the_wild_3_img_1.jpg", "demo_examples/in_the_wild_3_img_2.jpg", "demo_examples/in_the_wild_3.ply"],
        ["demo_examples/in_the_wild_4_img_1.jpg", "demo_examples/in_the_wild_4_img_2.jpg", "demo_examples/in_the_wild_4.ply"],
        ["demo_examples/in_the_wild_5_img_1.jpg", "demo_examples/in_the_wild_5_img_2.jpg", "demo_examples/in_the_wild_5.ply"],
        ["demo_examples/in_the_wild_6_img_1.jpg", "demo_examples/in_the_wild_6_img_2.jpg", "demo_examples/in_the_wild_6.ply"],
        ["demo_examples/in_the_wild_7_img_1.jpg", "demo_examples/in_the_wild_7_img_2.jpg", "demo_examples/in_the_wild_7.ply"],
        ["demo_examples/in_the_wild_8_img_1.jpg", "demo_examples/in_the_wild_8_img_2.jpg", "demo_examples/in_the_wild_8.ply"],
    ]

    for i in range(len(examples)):
        for j in range(len(examples[i])):
            examples[i][j] = hf_hub_download(repo_id=model_name, filename=examples[i][j])

    with tempfile.TemporaryDirectory(suffix='_mast3r_gradio_demo') as tmpdirname:

        cache_path = os.path.join(tmpdirname, chkpt_tag)
        os.makedirs(cache_path, exist_ok=True)

        recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, model, device, silent, image_size, ios_mode)

        if not ios_mode:
            for i in range(len(examples)):
                examples[i].insert(2, (examples[i][0], examples[i][1]))
                                         
        css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
        with gradio.Blocks(css=css, title="Splatt3R Demo") as demo:

            gradio.HTML('<h2 style="text-align: center;">Splatt3R Demo</h2>')

            with gradio.Column():
                gradio.Markdown('''
                    Please upload exactly one or two images below to be used for reconstruction.
                    If non-square images are uploaded, they will be cropped to squares for reconstruction.
                ''')
                if ios_mode:
                    inputfiles = gradio.Gallery(type="filepath")
                else:
                    inputfiles = gradio.File(file_count="multiple")
                run_btn = gradio.Button("Run")
                gradio.Markdown('''
                    ## Output
                    Below we show the generated 3D Gaussian Splat.
                    The generated splats are 30-40MB, so please allow up to 30 seconds for them to be downloaded from Hugging Face before rendering.
                    As it downloads your previous generations may be visible.
                    The arrow in the top right of the window below can be used to download the .ply for rendering with other viewers,
                    such as [here](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php?art=1&cu=0,-1,0&cp=0,1,0&cla=1,0,0&aa=false&2d=false&sh=0) or [here](https://playcanvas.com/supersplat/editor).
                ''')
                outmodel = gradio.Model3D(
                    clear_color=[1.0, 1.0, 1.0, 0.0],
                )
                run_btn.click(fn=recon_fun, inputs=[inputfiles], outputs=[outmodel])

                gradio.Markdown('''
                    ## Examples
                    A gallery of examples generated from ScanNet++ and from 'in the wild' images taken with a mobile phone.
                    These examples are 30-40MB, so please allow up to 30 seconds for them to be downloaded from Hugging Face before rendering.
                    As it downloads your previous generations may be visible.
                ''')
                
                snapshot_1 = gradio.Image(None, visible=False)
                snapshot_2 = gradio.Image(None, visible=False)
                if ios_mode:
                    gradio.Examples(
                        examples=examples,
                        inputs=[snapshot_1, snapshot_2, outmodel],
                        examples_per_page=5
                    )
                else:
                    gradio.Examples(
                        examples=examples,
                        inputs=[snapshot_1, snapshot_2, inputfiles, outmodel],
                        examples_per_page=5
                    )

        demo.launch(share=share, server_name=server_name, server_port=server_port)
