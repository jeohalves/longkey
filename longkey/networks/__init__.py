from .TagKPE import TagKPE
from .ChunkKPE import ChunkKPE
from .RankKPE import RankKPE
from .HyperMatch import HyperMatch
from .JointKPE import JointKPE
from .LongKey import LongKey
from .SpanKPE import SpanKPE
from ..constant import Idx2Tag
from . import hyperbolic

NUM_CLASSES = 2


def get_class(cfg):
    models = {
        "span": (SpanKPE, NUM_CLASSES),
        "tag": (TagKPE, len(Idx2Tag)),
        "chunk": (ChunkKPE, NUM_CLASSES),
        "rank": (RankKPE, NUM_CLASSES),
        "hypermatch": (HyperMatch, NUM_CLASSES),
        "joint": (JointKPE, NUM_CLASSES),
        "longkey": (LongKey, NUM_CLASSES),
    }

    if cfg.model.method in models:
        return models[cfg.model.method]

    else:
        raise RuntimeError(
            f"Invalid retriever class and model type {cfg.model.method}. Choices: {models.keys()}."
        )
