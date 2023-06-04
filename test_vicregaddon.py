#!/usr/bin/env python3
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

def test_get_shifted_batches(output_size=4096, combine=False):
    audio_long = torch.randn(5, 2, int(output_size*1.5))
    early, late = get_shifted_batches(audio_long, output_size, combine=combine)
    assert torch.equal(audio_long[:, :, :output_size], early)
    assert torch.equal(audio_long[:, :, -output_size:], late)
    print(f"Success!")
    return True


def test_VICRegLoss(*args, **kwargs):
    loss_class = VICRegLoss(*args, **kwargs)
    test_shape = (5, 32, 256)
    inputs = {'latent': torch.randn(test_shape), 'latent2': torch.randn(test_shape)}
    new_inputs = loss_class.forward(inputs)
    vicreg_losses = {key:val for key, val in new_inputs.items() if 'vicreg' in key}
    print("  vicreg_losses:\n", vicreg_losses)
    print(f"Success!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test vicregaddon package',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--parallel', action='store_true', help='use multiple GPUs')
    args = parser.parse_args()

    tests = [test_get_shifted_batches, wrapped_partial(test_VICRegLoss, parallel=args.parallel)]
    score = 0
    for i, test in enumerate(tests):
        print(f"test {i+1}: {test.__name__}: ", end="")
        score += test()

    print(f"{score} out of {len(tests)} tests passed")