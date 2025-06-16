import torch.nn as nn

from .flows.coupling import ResNetCouplingLayer
from .flows.lu import LULinear
from .flows.permutation import RandomPermutation

def create_flow_seq(dim, depth):
    chain = []
    for i in range(depth):
        chain.append(ResNetCouplingLayer(dim))
        chain.append(RandomPermutation(dim))
        chain.append(LULinear(dim))
    chain.append(ResNetCouplingLayer(dim))
    return SequentialFlow(chain)


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """
    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, context=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None and context is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        elif context is None:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, reverse=reverse)
            return x, logpx
        elif logpx is None:
            for i in inds:
                x = self.chain[i](x, context=context, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, context=context, reverse=reverse)
            return x, logpx
