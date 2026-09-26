"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import NewsClassifier  # noqa: E402

REAL_ARTICLES = [
    "WASHINGTON (Reuters) - The Senate passed a bipartisan infrastructure bill on Tuesday "
    "by a vote of 69-30, sending the measure to the House of Representatives for consideration.",
    "The central bank raised interest rates by a quarter of a percentage point on Wednesday "
    "and signalled that officials expect one further increase before the end of the year.",
    "Officials at the health ministry said vaccination coverage reached 84 percent this quarter, "
    "according to data released on Monday by the national statistics office.",
    "Lawmakers voted 240-180 to approve the appropriations package, which now heads to the "
    "president's desk. The bill funds the government through the end of the fiscal year.",
    "The unemployment rate fell to 4.1 percent last month, the labour department reported, "
    "as employers added 216,000 jobs across construction, manufacturing and healthcare.",
    "Researchers published their findings in a peer reviewed journal on Thursday, saying the "
    "trial followed 3,000 participants over five years under regulated conditions.",
    "The foreign ministry confirmed in a written statement that negotiations will resume next "
    "month, according to reporters travelling with the delegation on Sunday.",
    "Company executives said quarterly revenue rose 6 percent to 12.4 billion dollars, in line "
    "with analyst estimates published before the market opened on Tuesday morning.",
    "The supreme court agreed to hear the appeal next term, according to an order listed on "
    "the court's docket. Oral arguments are expected to be scheduled early next year.",
    "Voters in the district elected a new representative on Tuesday, election officials said, "
    "after all precincts reported and the remaining mail ballots were counted by Wednesday.",
]

FAKE_ARTICLES = [
    "SHOCKING!! Doctors are FURIOUS about this one weird trick that cures diabetes in 3 days!! "
    "Big pharma is BANNING this miracle cure — share this before it gets deleted forever!!!",
    "You won't believe what this politician did — the secret elite conspiracy they covered up "
    "with the help of the dishonest fake stream media. Wake up before it is too late!!!",
    "BREAKING!!! A secret insider reveals the truth they don't want you to know about vaccines. "
    "Miracle doctors hate this instant cure, banned by the government and hidden from YOU!!!",
    "EXPOSED!! The illuminati plan to control your mind has been revealed by a whistleblower "
    "who was immediately silenced. Share this urgently before censors delete the evidence!!!",
    "Miracle weight loss melt destroys 40 pounds in one week without diet or exercise!!! "
    "Nutritionists are stunned by this instant secret cure that pharmaceutical firms banned!!!",
    "You do NOT want to miss this — a hidden government document proves the whole thing was a "
    "hoax engineered by elites. Wake up, do your own research and share before this is censored!!!",
    "REVEALED: the truth about the secret cure they healed patients instantly and then buried. "
    "The fake news media refuses to cover this shocking exclusive story. Share before deleted!!!",
    "This celebrity died and came back to life to expose the conspiracy — the doctors hate the "
    "miracle secret he revealed. Exclusive shocking footage, must see before it is banned!!!",
    "The government is HIDING this miracle cure from you!! Doctors are terrified of this secret "
    "and the fake stream media will never report it. Share now before this gets banned forever!!!",
    "ANONYMOUS insiders claim the whole election was rigged by elites using secret machines. "
    "You won't believe the shocking proof they tried to bury — wake up and share this NOW!!!",
]

FAKE_ARTICLE = FAKE_ARTICLES[0]
REAL_ARTICLE = REAL_ARTICLES[0]


@pytest.fixture(scope="session")
def sample_data():
    texts = REAL_ARTICLES + FAKE_ARTICLES
    labels = ["REAL"] * len(REAL_ARTICLES) + ["FAKE"] * len(FAKE_ARTICLES)
    return texts, labels


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory, sample_data):
    """Train a tiny model once and keep it on disk for app tests."""
    texts, labels = sample_data
    model = NewsClassifier(max_features=500)
    model.fit(texts, labels)
    path = tmp_path_factory.mktemp("model") / "news_classifier.joblib"
    model.save(str(path))
    return model, str(path)


@pytest.fixture()
def db_app():
    """A bare app with an empty in-memory database (no model required)."""
    from app import create_app

    application = create_app("testing")
    with application.app_context():
        from src.storage import db

        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def app(db_app, trained_model):
    from app import registry

    _model, path = trained_model
    registry.path = path
    registry.load(force=True)
    yield db_app
    registry.unload()


@pytest.fixture()
def client(app):
    return app.test_client()
