#!/usr/bin/env python3

"""
This is a pretty bare-bones testing script.  It deliberately doesn't make use 
of any accelerate or Lighthing functions. 
...and I don't actually know how to launch a job with *just* torch.dist, so 
parallel testing will will occur in "examples/"
"""

import torch
from vicregaddon import *
import argparse
from functools import partial, update_wrapper

TensorDict = Dict[str, torch.Tensor]


def wrapped_partial(func, *args, **kwargs):
    "source: http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/"
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def test_get_shifted_batches(*args, **kwargs):
    output_size=kwargs.get('output_size', 256) 
    combine=kwargs.get('combine', False)
    batch_size = kwargs.get('batch_size', 5)
    audio_long = torch.randn(batch_size, 2, int(output_size*1.5))
    early, late = get_shifted_batches(audio_long, output_size, combine=combine)
    assert torch.equal(audio_long[:, :, :output_size], early)
    assert torch.equal(audio_long[:, :, -output_size:], late)
    print(f"Success!")
    return True


def test_VICRegLoss(*args, **kwargs):
    batch_size = kwargs.get('batch_size', 5)
    kwargs.pop('batch_size')
    loss_class = VICRegLoss(*args, **kwargs)
    test_shape = (batch_size, 32, 256)
    # TODO: lock down the RNG for reproducibility
    inputs = {'latent': torch.randn(test_shape), 'latent2': torch.randn(test_shape)}
    new_inputs = loss_class.forward(inputs)
    vicreg_losses = {key:val for key, val in new_inputs.items() if 'vicreg' in key}
    print("  vicreg_losses:\n", vicreg_losses)
    print(f"Success!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test vicregaddon package',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--parallel', action='store_true', help='use multiple GPUs')
    args = parser.parse_args()

    tests = [test_get_shifted_batches, wrapped_partial(test_VICRegLoss, parallel=args.parallel)]
    score = 0
    for i, test in enumerate(tests):
        print(f"test {i+1}: {test.__name__}: ", end="")
        score += test(batch_size=args.batch_size)

    print(f"{score} out of {len(tests)} tests passed")