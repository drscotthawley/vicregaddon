# vicregaddon

> *"vicreg add-on".  As in "Yeah, but are you getting it? vicregaddon it"*

A lightweight and modular parallel PyTorch implementation of [VICReg](https://github.com/facebookresearch/vicreg). 

(I intend this for audio, but could be used for other things)

## Usage:
```
from vicregaddon import *
```

## Dependencies: 

- torch
- einops

In the library itself, parallelism is handled directly via `torch.distributed` and thus is "agnostic" to whether PyTorch Lightning or Accelerate (or some other thing) is used.

The `examples/` however, have several more dependencies. You will need to install Lightning and/or Accelerate and/or other things to run them, depending on each example.
