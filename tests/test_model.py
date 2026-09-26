"""Tests for feature extraction and the classifier."""

import numpy as np
import pytest

from src.dataset import FAKE, REAL
from src.features import STYLE_FEATURES, StyleFeatureExtractor, build_vectorizer, combine_features
from src.model import NewsClassifier
from src.preprocessing import clean_text


def test_style_features_shape_and_keys():
    extractor = StyleFeatureExtractor()
    matrix = extractor.transform(["SHOCKING!!! You won't BELIEVE this!!"])
    assert matrix.shape == (1, len(STYLE_FEATURES))
    assert np.isfinite(matrix).all()


def test_style_features_capture_sensationalism():
    extractor = StyleFeatureExtractor()
    shouty = extractor.transform(["BREAKING!!! SHOCKING exclusive secret revealed NOW!!!"])[0]
    calm = extractor.transform(
        ["The committee published its report on Thursday in the capital."]
    )[0]
    style = dict(zip(STYLE_FEATURES, shouty))
    calm_style = dict(zip(STYLE_FEATURES, calm))
    assert style["exclaim_ratio"] > calm_style["exclaim_ratio"]


def test_vectorizer_uses_cleaning_pipeline():
    vectorizer = build_vectorizer(max_features=50)
    vectorizer.fit(["Check https://a.com now", "Check this report now"])
    transformed = vectorizer.transform(["check url"])
    assert transformed.shape[0] == 1
    assert transformed.nnz >= 1


def test_combine_features_shapes():
    vectorizer = build_vectorizer(max_features=50)
    texts = ["alpha beta gamma", "beta gamma delta", "gamma delta epsilon"]
    tfidf = vectorizer.fit_transform(texts)
    style = StyleFeatureExtractor().fit_transform(texts)
    combined = combine_features(tfidf, style)
    assert combined.shape[0] == 3
    assert combined.shape[1] == tfidf.shape[1] + style.shape[1]


def test_classifier_learns_the_task(sample_data):
    texts, labels = sample_data
    model = NewsClassifier(max_features=300)
    model.fit(texts, labels)

    assert model.is_fitted
    assert set(model.predict(texts)) <= {FAKE, REAL}

    fake_result = model.predict_one(FAKE_ARTICLE)
    real_result = model.predict_one(REAL_ARTICLE)
    assert fake_result["label"] == FAKE
    assert real_result["label"] == REAL
    assert 0.0 <= fake_result["confidence"] <= 1.0
    assert abs(sum(fake_result["probabilities"].values()) - 1.0) < 1e-6


def test_predict_before_training_raises():
    with pytest.raises(RuntimeError):
        NewsClassifier().predict(["anything"])


def test_evaluate_returns_expected_metrics(sample_data):
    texts, labels = sample_data
    model = NewsClassifier(max_features=300)
    model.fit(texts, labels)
    scores = model.evaluate(texts, labels)

    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in scores
        assert 0.0 <= scores[key] <= 1.0
    assert scores["accuracy"] == pytest.approx(1.0)
    assert len(scores["confusion_matrix"]) == 2


def test_save_and_load_roundtrip(tmp_path, trained_model):
    model, path = trained_model
    reloaded = NewsClassifier.load(path)

    assert reloaded.is_fitted
    assert reloaded.predict_one(REAL_ARTICLE)["label"] == model.predict_one(REAL_ARTICLE)["label"]
    assert reloaded.summary()["model_version"] == model.summary()["model_version"]


def test_summary_is_serialisable(trained_model):
    import json

    model, _ = trained_model
    payload = model.summary()
    assert json.loads(json.dumps(payload)) == payload


FAKE_ARTICLE = (
    "SHOCKING!!! You WON'T BELIEVE this one weird trick!! Doctors HATE it and big pharma "
    "is BANNING this miracle cure. Share before it is deleted forever, the truth they hide!!!"
)

REAL_ARTICLE = (
    "WASHINGTON (Reuters) - The Senate approved a bipartisan infrastructure bill on Tuesday "
    "by a vote of 69-30, according to a tally released by the clerk's office."
)


def test_clean_text_is_used_consistently():
    assert clean_text(REAL_ARTICLE) not in (None, "")
