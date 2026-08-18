import numpy as np
import pandas as pd

from ml.embeddings import build_event_text, embed_texts


def test_build_event_text_composes_fields():
    events = pd.DataFrame({
        "id": [1],
        "name": ["Lisboa Trail"],
        "description": ["A nice trail run in Lisbon hills."],
    })
    event_categories = {1: {1, 2}}
    slug_by_category_id = {1: "trail-running", 2: "road-running"}
    texts = build_event_text(events, event_categories, slug_by_category_id)
    assert len(texts) == 1
    assert "Lisboa Trail" in texts[0]
    assert "trail-running" not in texts[0]
    assert "road-running" not in texts[0]
    assert "nice trail run" in texts[0]


def test_embed_texts_shape_and_determinism():
    texts = ["hello world", "trail running in Portugal"]
    a = embed_texts(texts)
    b = embed_texts(texts)
    assert a.shape == (2, 384)
    assert a.dtype == np.float32
    np.testing.assert_allclose(a, b, atol=1e-5)
