import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
APP_NAME = "AI-Powered Fake News Detection System"

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fake_news.db")
BLOCKCHAIN_RPC_URL = os.getenv("BLOCKCHAIN_RPC_URL", "")
BLOCKCHAIN_CONTRACT_ADDRESS = os.getenv("BLOCKCHAIN_CONTRACT_ADDRESS", "")
BLOCKCHAIN_PRIVATE_KEY = os.getenv("BLOCKCHAIN_PRIVATE_KEY", "")

# Flask-SQLAlchemy expects a full URL; a bare driver name means "relative to instance/".
if DATABASE_URL and not DATABASE_URL.startswith(("sqlite:///", "postgresql://", "mysql://", "mysql+pymysql://")):
    DATABASE_URL = f"sqlite:///{DATABASE_URL}"


class Config:
    """Base configuration."""

    SECRET_KEY = SECRET_KEY
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB per request
    APP_NAME = APP_NAME
    JSON_SORT_KEYS = False


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config(name: str | None = None) -> type:
    name = (name or os.getenv("FLASK_ENV", "development")).lower()
    return config_by_name.get(name, DevelopmentConfig)
