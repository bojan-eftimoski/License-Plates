"""Smoke tests for the FastAPI backend (web/ — not graded). Guarded on data presence."""
import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
    if not os.path.isdir("data/templates") or not os.listdir("data/templates"):
        pytest.skip("templates not generated (run tools/dataset/synth_glyphs.py)")
    from web.app import app
    return TestClient(app)


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_analyze_image_reads_clear_plate(client):
    if not os.path.isfile("Images/registracii1.jpg"):
        pytest.skip("sample image not present")
    with open("Images/registracii1.jpg", "rb") as f:
        j = client.post("/api/analyze", files={"file": ("registracii1.jpg", f, "image/jpeg")}).json()
    assert j["type"] == "image"
    assert j["annotated"]
    assert "SK9507BT" in {p["plate_text"] for p in j["plates"]}   # the clear plate must read


def test_csv_export(client):
    assert "plate_text" in client.get("/api/results.csv").text
