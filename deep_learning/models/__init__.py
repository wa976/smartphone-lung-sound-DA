
from .ast import ASTModel
import torch
from .ast_ftms import ASTFTMSModel

_backbone_class_map = {
    'ast': ASTModel,
    'ast_ftms' : ASTFTMSModel,
}


def get_backbone_class(key):
    if key in _backbone_class_map:
        return _backbone_class_map[key]
    else:
        raise ValueError('Invalid backbone: {}'.format(key))