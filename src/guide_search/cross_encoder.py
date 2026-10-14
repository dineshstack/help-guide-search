"""The real reranker.

This file loads a cross-encoder with the sentence-transformers library and
offers it through the `Reranker` interface from `reranking.py`.

Three models were candidates. All three have a licence that allows commercial
use and were trained on many languages, Arabic among them. They were compared
on the `dev` questions, and the one named by MODEL_NAME and REVISION was
chosen. The numbers are in evaluation/reranker_choice.json and in the
report, under "The reranker".

It is large: about 2.3 GB on disk and in memory, and several seconds per
question on two CPUs, because it reads every candidate passage in full.

Importing this file loads torch, which takes a few seconds. Creating a
`CrossEncoderReranker` loads the model itself, which downloads it the first
time. So this file is imported only where the real model is needed, never by
the unit tests.
"""

import torch
from sentence_transformers import CrossEncoder

# Each candidate, pinned to one revision: the exact commit of the model files.
# Without it we would get whatever the model's page holds on the day of the
# download.
CANDIDATES = {
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": "1427fd652930e4ba29e8149678df786c240d8825",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "Alibaba-NLP/gte-multilingual-reranker-base": "8215cf04918ba6f7b6a62bb44238ce2953d8831c",
}

# The chosen model.
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
REVISION = CANDIDATES[MODEL_NAME]

# The longest input, in tokens, for one question and passage together. A
# passage is at most about 1,200 characters plus its title and heading, which
# stays inside this. Longer input would be cut off at the end.
MAX_LENGTH = 512


class CrossEncoderReranker:
    """The real model, behind the same names as the fake one."""

    def __init__(self, name: str = MODEL_NAME, revision: str | None = None) -> None:
        if revision is None:
            revision = CANDIDATES[name]
        self.model_id = f"{name}@{revision}"
        # device: left alone, the library uses the GPU on an Apple laptop and
        # the CPU elsewhere, and the scores would differ slightly.
        # activation_fn: the sigmoid turns each raw score into a number
        # between 0 and 1. Set here so that every candidate is read the same
        # way, whatever its own files say.
        # trust_remote_code stays False: a model that needs to run code from
        # its own repository is not used.
        self.model = CrossEncoder(
            name,
            revision=revision,
            device="cpu",
            max_length=MAX_LENGTH,
            activation_fn=torch.nn.Sigmoid(),
            trust_remote_code=False,
        )

    def score(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(question, text) for text in texts], show_progress_bar=False)
        # predict returns a numpy array; the rest of the code wants plain floats.
        return [float(score) for score in scores]
