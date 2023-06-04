#!/usr/bin/env python3

# this runs in parallel via accelerate, loads audio from webdataset
# encodes it vith a torchscript file of an autoencoder, and does 
# VICReg on those embeddings. 
# e.g it fine-tunes OOBLECK to have a more 'semantic' latent space
# ....though OOBLECK training is done via Lightning and we're not doing that.  



# Sample launch:
"""
 accelerate launch ./wds_audio_accelerate.py --model_ckpt=~/checkpoints/autoencoder.ts --data_sources="s3://s-laion/stuff" --profiles='' --batch_size=128 --sample_size=32768 --sample_rate=48000 --num_workers=12 --recursive
"""

# much of this code is plagiarized from fad_pytorch because I'm writing these at the same time ;-)  --drscotthawley

import os
import argparse
from accelerate import Accelerator
import warnings
import torch

from aeiou.core import get_device, load_audio, get_audio_filenames, makedir
from aeiou.datasets import get_wds_loader
from aeiou.hpc import HostPrinter
from pathlib import Path
#from audio_algebra.given_models import StackedDiffAEWrapper
import ast
import torchaudio
from tqdm.auto import tqdm
import math



def do_the_thing(args):
    "the thing that does the thing"

    # HPC / parallel setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddps = f"[{local_rank}/{world_size}]"  # string for distributed computing info, e.g. "[1/8]" 
    accelerator = Accelerator()
    hprint = HostPrinter(accelerator)  # hprint only prints on head node
    device = accelerator.device  # get_device()
    hprint(f"lfg: args = {args}")
    hprint(f'{ddps} Using device: {device}')
    
    # let's load up that (auto)encoder (we don't care about the decoder part in this demo though)
    model_ckpt, data_sources, profiles, n = args.model_ckpt, args.data_sources, args.profiles,  args.n
    names = data_sources.split(' ')
    hprint(f"names = {names}")
    profiles = ast.literal_eval(profiles)
    hprint(f"profiles = {profiles}")
    
    # dataloader
    dl = get_wds_loader(
        batch_size=args.batch_size,
        s3_url_prefix=None,
        sample_size=int(args.sample_size*1.2),  # we're gonna grab audio that's slightly too long and then use a datasplitter on it
        names=names,
        sample_rate=args.sample_rate,
        num_workers=args.num_workers,
        recursive=True,
        random_crop=True,
        epoch_steps=10000,
        profiles=profiles,
    )
   
    print(f"loading {model_ckpt}....")
    if model_ckpt.endswith('.ts'):
        model = torch.jit.load(model_ckpt)
    #else:  # default is stacked diffae
    #    model = StackedDiffAEWrapper(ckpt_info={'ckpt_path':model_ckpt})
    try:
        model.setup()  # if it needs setup call
    except: 
        pass 
    model.eval()  # normally we might let this model train, but for this example we'll just train the Projector 
    model = model.to(device)

    vl_loss = VICRegLoss()
    
    model, dl, vl_loss = accelerator.prepare( model, dl, vl_loss )  # prepare distributes things among GPUs

    # VICReg Training loop
    progress_bar = tqdm(range(math.ceil(args.n/args.batch_size)), disable=not accelerator.is_local_main_process)

    for i, data in enumerate(dl): # lets get some data, batches
        reals = data[0][0]  # we only care about the first batch, and the first item in that batch
        early, late = get_shifted_batches(reals, output_length=args.sample_size)
        hprint("hey, early.shape, late.shape = ", early.shape, late.shape)
        progress_bar.update(1)

    # TODO: add more! 
   

def main(): 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('name', help='Name prefix for output directories: <name>_reals/ & <name>_fakes/')
    parser.add_argument('model_ckpt', help='TorchScript (.ts) Autoencoder checkpoint file')

    parser.add_argument('data_sources', help='Space-separated string listing S3 resources for data')
    parser.add_argument('-b',"--batch_size", default=2, help='batch size')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of Pytorch workers to use in DataLoader')
    parser.add_argument('-p',"--profiles", default='', help='String representation of dict {resource:profile} of AWS credentials')
    parser.add_argument('--sample_rate', type=int, default=48000, help='sample rate (will resample inputs at this rate)')
    parser.add_argument('-s','--sample_size', type=int, default=2**18, help='Number of samples per clip')

    args = parser.parse_args()
    do_the_thing( args )


if __name__ == '__main__' and "get_ipython" not in dir():
    main()
