"""
Pytest Configuration and Fixtures

Author: Zhmuryk Andrii
Copyright (c) 2024-2026 Andrii Zhmuryk. Released under the MIT License (see LICENSE)
"""

import os
import tempfile

# Set testing environment before importing app
os.environ["TESTING"] = "true"
# The app reads DATABASE_URL when it is imported. Unless one is provided (the CI
# integration job runs these tests against PostgreSQL), point it at a throwaway
# SQLite file so tests that go through the real engine never depend on a database
# left over from an earlier run, and never touch the developer's own one.
if "DATABASE_URL" not in os.environ:
    _TEST_DB_DIR = tempfile.mkdtemp(prefix="fraud-tests-")
    os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_DIR}/test.db".replace("\\", "/")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import Base, get_db
from app.db.models import User, Prediction


@pytest.fixture(scope="session", autouse=True)
def app_database():
    """Create the tables in the app's own (temporary) database once per session."""
    from app.db.database import init_db

    init_db()
    yield


@pytest.fixture(scope="session", autouse=True)
def fraud_model_loaded(tmp_path_factory):
    """Load a small model trained on synthetic data when no trained model exists.

    The real model is trained on the Kaggle dataset, which is not in the repository,
    so the model endpoints would answer 503 in a clean checkout. This trains a tiny
    forest on the project's own synthetic generator, which is enough to exercise the
    endpoints, and leaves a real model alone when one is present.
    """
    from pathlib import Path

    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    from app.core.config import settings
    from app.models.ml_model import fraud_model
    from ml.generate_sample_data import generate_synthetic_data

    backend_dir = Path(__file__).resolve().parent.parent
    if (backend_dir / settings.model_path).exists() and (backend_dir / settings.scaler_path).exists():
        yield
        return

    df = generate_synthetic_data(n_samples=3000, fraud_ratio=0.02)
    features, labels = df.drop(columns="Class"), df["Class"]
    scaler = StandardScaler().fit(features)
    model = RandomForestClassifier(
        n_estimators=25, max_depth=8, random_state=42, class_weight="balanced"
    ).fit(scaler.transform(features), labels)

    out = tmp_path_factory.mktemp("model")
    joblib.dump(model, out / "fraud_detector.pkl")
    joblib.dump(scaler, out / "scaler.pkl")
    assert fraud_model.load(str(out / "fraud_detector.pkl"), str(out / "scaler.pkl"))
    yield


# Create in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create test client with overridden database"""
    app.dependency_overrides[get_db] = lambda: db_session

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db_session):
    """Create a test user"""
    from app.services.auth_service import get_password_hash
    
    user = User(
        username="testuser",
        email="test@example.com",
        full_name="Test User",
        hashed_password=get_password_hash("Password123!"),
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(client, test_user):
    """Get authentication headers for test user"""
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "testuser", "password": "Password123!"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_transaction():
    """Sample transaction data for testing"""
    return {
        "time": 0.0,
        "v1": -1.359807,
        "v2": -0.072781,
        "v3": 2.536347,
        "v4": 1.378155,
        "v5": -0.338321,
        "v6": 0.462388,
        "v7": 0.239599,
        "v8": 0.098698,
        "v9": 0.363787,
        "v10": 0.090794,
        "v11": -0.551600,
        "v12": -0.617801,
        "v13": -0.991390,
        "v14": -0.311169,
        "v15": 1.468177,
        "v16": -0.470401,
        "v17": 0.207971,
        "v18": 0.025791,
        "v19": 0.403993,
        "v20": 0.251412,
        "v21": -0.018307,
        "v22": 0.277838,
        "v23": -0.110474,
        "v24": 0.066928,
        "v25": 0.128539,
        "v26": -0.189115,
        "v27": 0.133558,
        "v28": -0.021053,
        "amount": 149.62
    }
