from .DataJoint import DataJoint
from .DataChunk import DataChunk
from .DataLong import DataLong
from .DataRank import DataRank
from .DataSpan import DataSpan
from .DataTag import DataTag

method_dataloader_classes = {
    "span": DataSpan,
    "tag": DataTag,
    "chunk": DataChunk,
    "rank": DataRank,
    "hypermatch": DataRank,
    "joint": DataJoint,
    "longkey": DataLong,
}


def get_class(name):
    if name in method_dataloader_classes:
        return method_dataloader_classes[name]
    else:
        raise RuntimeError("Invalid retriever class: %s" % name)
