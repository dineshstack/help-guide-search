"""The real embedding model.

This file loads the model "intfloat/multilingual-e5-small" with the
sentence-transformers library and offers it through the `Embedder` interface
from `embedding.py`. The model was trained on about 100 languages, so an
Arabic question can match an English passage.

Importing this file loads torch, which takes a few seconds. Creating an
`E5Embedder` loads the model itself; the first time ever, that downloads about
0.5 GB into the Hugging Face cache. So this file is imported only where the
real model is needed, never by the unit tests.
"""

from sentence_transformers import SentenceTransformer

from guide_search.embedding import passage_input, query_input

MODEL_NAME = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class E5Embedder:
    """The real model, behind the same three names as the fake one."""

    # Stored with every vector, so vectors made by another model or another
    # revision are never mixed up with these.
    model_id = f"{MODEL_NAME}@{REVISION}"

    def __init__(self) -> None:
        # revision: the exact commit of the model files. Without it we would
        # get whatever the model's page holds on the day of the download.
        # device: left alone, the library uses the GPU on an Apple laptop and
        # the CPU elsewhere, and the numbers would differ slightly.
        self.model = SentenceTransformer(MODEL_NAME, revision=REVISION, device="cpu")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        inputs = [passage_input(text) for text in texts]
        # normalize_embeddings scales every vector to length 1.
        vectors = self.model.encode(inputs, normalize_embeddings=True)
        # encode returns a numpy array; the rest of the code wants plain lists.
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(query_input(text), normalize_embeddings=True)
        return vector.tolist()

    def token_count(self, text: str) -> int:
        """Count the tokens of this exact text, start and end tokens included.

        The model reads at most 512 tokens and silently ignores the rest. This
        count is not cut off there, so a result above 512 means the end of the
        text would be lost.
        """
        token_ids = self.model.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        return len(token_ids)
