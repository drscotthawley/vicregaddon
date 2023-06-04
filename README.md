# vicregaddon

> *"vicreg add-on" (can be jokingly pronounced like "Armageddon" for those so inclined)*

A lightweight and modular parallel PyTorch implementation of VICReg (intended for audio)

## Usage:
```
from vicregaddon import *
```

## Dependencies: 

- torch
- einops

In the library itself, parallelism is handled directly via `torch.distributed` and thus is "agnostic" to whether PyTorch Lightning or Accelerate (or some other thing) is used.

The `examples/` however, have several more dependencies. You will need to install Lightning and/or Accelerate and/or other things to run them, depending on each example.
