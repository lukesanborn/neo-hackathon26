class Vocab:  # placeholder base class for scgpt.tokenizer.GeneVocab
    def __init__(self, *args, **kwargs):
        raise RuntimeError("torchtext shim: Vocab is not available in the Blackwell image")


def vocab(*args, **kwargs):
    raise RuntimeError("torchtext shim: torchtext.vocab.vocab is not available")


def build_vocab_from_iterator(*args, **kwargs):
    raise RuntimeError("torchtext shim: build_vocab_from_iterator is not available")
