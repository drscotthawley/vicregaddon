import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Optional, Sequence
from einops import rearrange, pack, unpack

# 4 parts to this file: 
#   1. "data splitter" routines
#   2. the Projector model
#   3. losses -- implemented both as standalone functions and/or as nn.Module class
#   4. Copy of Facebook's MIT license, to cover FullGatherLayer and other code snippets


#---------------------------
# 1. "data splitter" options/routines
#---------------------------
def get_shifted_batches(audio_in:torch.Tensor, output_length=None, combine=False):
    """
    Very simple standalone data splitter function, based on time.
    Splits a long audio clip into two shorter clips, taken from the "start" and "end" of the long clip.
    Returns the two versions of the audio clip as a single tensor, with the "early" clip(s) first.
    Default is a no-op (i.e. returns the original audio_in)

    audio_in is expected to have dimensions [batch, channels, time]
    """
    if (output_length is None) or (output_length >= audio_in.shape[-1]) or (output_length <= 0): 
        return audio_in   # no op
    early = audio_in[..., :output_length]
    late = audio_in[..., -output_length:]
    if combine:
        return torch.cat([early, late], dim=0) # can undo via torch.chunk for 2 chunks
    else: 
        return early, late


class DataSplitter(nn.Module):
    "This is a more sophisticated data splitter than *can* split by time but can also apply effects, etc."
    def __init__(self, 
                 output_length=None, # if not None, will try to split by time
                 effects_string='',  # string of list of names of valid aeiou.datasets augmentation routines
                 **kwargs):
        super().__init__()
        self.output_length, self.effects_string, self.kwargs = output_length, effects_string, kwargs

    def forward(self, audio_in:torch.Tensor):
        if self.output_length is not None:
            audio_in = get_shifted_batches(audio_in, output_length=self.output_length)
        if self.effects_string != '':
            print("WARNING: DataSplitter is not yet implemented to apply effects.  No effects will be applied.")
            #audio_in = apply_effects(audio_in, self.effects_string, **self.kwargs)
        return audio_in



#---------------------------
# 2. Projector model
# This is a "fallback" model: users can supply their own to VICRegLoss() instead of this, 
#   e.g., if they'd prefer one based on convolutions instead of Linear layers
#---------------------------
def VICRegProjector(
        flat_latent_dim=32*256,  # flattened dimension of the latent encoding space of autoencoder. (VICReg on images uses 2048 from 320x320 images)
        mlp_str="8192-8192-8192",  # default spec of MLP layers as per VICReg code
        ):
    """Fallback model if none specified by user. 
    (Slightly) Adapted from FacebookResearch's original VICReg code. Their LICENSE is below"""
    mlp_spec = f"{flat_latent_dim}-{mlp_str}" # string of sizes of MLP layers
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)




#---------------------------
# 3. losses -- implemented both as standalone functions and/or as nn.Module class 
#---------------------------

# first, just a few helper functions

# utility layer needed for parallel runs, called by VICRegLoss class below
class FullGatherLayer(torch.autograd.Function):
    """
    Source: original VICreg code from FacebookResearch: https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py#L307 MIT LICENCE below
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


# couple more utilities used by oobleck 
TensorDict = Dict[str, torch.Tensor]  # not sure if this is the same as in tensordict package

def accumulate_value(inputs: TensorDict, update: TensorDict):
    for k, v in update.items():
        if k in inputs:
            inputs[k] += v
        else:
            inputs[k] = v
    return inputs



# next, standalone versions of vicreg losses e.g. static methods

# *WARNING:* these standalone functions assume you've already done a "gather" operation
# when running in parallel. See the VICRegLoss class below for a more complete
# implementation that does the gather for you.
def vicreg_var_loss(z:torch.Tensor, gamma=1.0, eps=1e-4):
    "variance loss for VICReg"
    std_z = torch.sqrt(z.var(dim=0) + eps) 
    return torch.mean(F.relu(gamma - std_z))   # the relu performs the max(0, ...)

def vicreg_inv_loss(z1:torch.Tensor, z2:torch.Tensor):
    "invariance / similarity loss for VICReg: just MSE"
    return F.mse_loss(z1, z2)

def off_diagonal(x:torch.Tensor):
    "gets off-diagnonal elements of a square (covariance) matrix; used by vicreg_cov_loss "
    n, m = x.shape
    assert n == m, f"off_diagonal: matrix must be square but got shape: {x.shape}"  
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_cov_loss(z:torch.Tensor):
    "Covariance loss for VICReg. the sum of the off-diagaonal terms of the covariance matrix"
    num_features = z.shape[1]*z.shape[2]  
    cov_z = torch.cov(rearrange(z, 'b c t -> ( c t ) b'))   
    return off_diagonal(cov_z).pow_(2).sum().div(num_features)


# now a full class that does a lot of things for you
class VICRegLoss(nn.Module):
    """
    Follows the loss APIs used in Harmonai's OOBLECK repo 
    VICReg, https://arxiv.org/abs/2105.04906, https://github.com/facebookresearch/vicreg/
    VICReg operates on the projections z of the autoencoder's latent encodings y.
    """

    def __init__(self, gamma: float = 1., eps: float = 1.0e-4, weight: float = 1.,
                var_weight: float = 1., inv_weight: float = 1., cov_weight: float = 1.,
                parallel=True, 
                projector=None,  # projector model, e.g. multi-layer MLP. will fall back to default setup (see VICRegProjector, above) if None
                # these next two are only applicable if projector is None;
                flat_latent_dim=32768, #32*256, # = 8192, but doesn't have to match projector_mlp
                projector_mlp="8192-8192-8192", # dims of projector layers. Using string API b/c that's what Meta did, and if it's good enough for Yann,...
                debug=False,
                 ) -> None:
        super().__init__()
        self.gamma, self.eps, self.weight = gamma, eps, weight
        self.var_weight, self.inv_weight, self.cov_weight =  var_weight, inv_weight, cov_weight
        self.parallel = parallel
        self.flat_latent_dim = flat_latent_dim
        self.projector = projector if projector is not None else VICRegProjector(flat_latent_dim=flat_latent_dim, mlp_str=projector_mlp)
        self.debug = debug
        self.show_trunc_notice = True

    def forward(self, inputs:TensorDict) -> TensorDict:
        "this does not operate on audio, but rather on the latent encodings y"
        y1, y2 = inputs['latent'], inputs['latent2'] # TODO: not married to these names

        # prep for Projector.  If it's using Linear layers, it needs a flat input
        y1f, ps = pack( [y1], 'b *' )
        y2f, ps = pack( [y2], 'b *' )
        if y1f.shape[-1] != self.flat_latent_dim and self.show_trunc_notice :
                print(f"Expected y1.flatten().shape[-1] to be {self.flat_latent_dim}, but got y1f.shape = {y1f.shape}.  Truncating to fit. Suppressing further notices.")  
                self.show_trunc_notice = False 
        maxdim = max( y1f.shape[-1] , self.flat_latent_dim )

        # run Projector on encodings y to get embeddings z 
        z1 = self.projector(y1f[..., :maxdim])
        z2 = self.projector(y2f[..., :maxdim])

        # If we had flattened for the Projector, we need to unflatten. TODO: make this more general
        #[z1] = unpack(z1, ps, 'b *')  # this assumes we're in the same dimensionality as the y space
        #[z2] = unpack(z2, ps, 'b *')
        z1 = z1.view( (y1.shape[0],y1.shape[1],-1) )
        z2 = z2.view( (y2.shape[0],y2.shape[1],-1) )
        # or...
        #z1 = z1.view( (y1.shape[0],-1, y1.shape[2]) )
        #z2 = z2.view( (y2.shape[0],-1, y2.shape[2] )        
        
        if self.parallel: # TODO: could detect this automatically, e.g. using os.getenv['WORLD_SIZE']
            if not dist.is_available():
                raise ValueError(
                    "Error: parallel=True but torch.distributed is not available (yet)."
                    "Either set parallel=False or make sure VICRegLoss is initialized late enough in the PyTorch Lightning execution that torch.distributed is enabled."
                )
            if dist.is_initialized(): # distributed computing
                if dist.get_world_size() > 1: # gather from all processes for batch statistics
                    z1 = torch.cat(FullGatherLayer.apply(z1), dim=0)
                    z2 = torch.cat(FullGatherLayer.apply(z2), dim=0)
            else:
                print("WARNING: self.parallel=True but dist.is_initialized=False. No distributed gather will occur")

        z1 = z1 - z1.mean(dim=0) # zero-mean across batches. b/c Facebook does it
        z2 = z2 - z2.mean(dim=0)

        var_loss = self.var_weight * 0.5 * ( vicreg_var_loss(z1, gamma=self.gamma, eps=self.eps) + vicreg_var_loss(z1, gamma=self.gamma, eps=self.eps) ) 
        inv_loss = self.inv_weight * vicreg_inv_loss(z1, z2)
        cov_loss = self.cov_weight * 0.5 * ( vicreg_cov_loss(z1) + vicreg_cov_loss(z2) )
        loss = self.weight * ( var_loss + inv_loss + cov_loss )
        inputs = accumulate_value(
            inputs,
            {
                "vicreg_var_loss": var_loss,
                "vicreg_inv_loss": inv_loss,
                "vicreg_cov_loss": cov_loss,
                "vicreg_loss": loss
            },
        )
        #inputs.update({"vicreg_loss": loss})
        return inputs
    


#------------------------------------
# 4. for Facebook's code, here's their MIT license:
#------------------------------------
# MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
