from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from mnemo.api.app import create_app
from mnemo.config import Settings


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(
        db_path=tmp_path / "test.db",
        work_dir=tmp_path / "work",
        cookies_path=tmp_path / "cookies.txt",
    )
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.text == "OK"


def test_cookies_endpoint_writes_file(client, tmp_path):
    r = client.post("/api/v1/auth/cookies", json={"cookies": "# Netscape\n.youtube.com\tTRUE\t/\tTRUE\t0\ttest\tval\n"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    saved = (tmp_path / "cookies.txt").read_text()
    assert "test" in saved


def test_video_process_enqueues(client):
    r = client.post("/api/v1/video/process",
                    json={"video_url": "https://youtube.com/watch?v=test"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert body["video_id"].startswith("video_")


def test_video_process_rejects_empty_url(client):
    r = client.post("/api/v1/video/process", json={"video_url": ""})
    assert r.status_code == 400


def test_video_status_404_for_unknown(client):
    r = client.get("/api/v1/video/status", params={"video_id": "nope"})
    assert r.status_code == 404


def test_video_status_returns_pending_after_enqueue(client):
    r1 = client.post("/api/v1/video/process",
                     json={"video_url": "https://x.com/v.mp4"})
    vid = r1.json()["video_id"]
    r2 = client.get("/api/v1/video/status", params={"video_id": vid})
    assert r2.status_code == 200
    body = r2.json()
    assert body["video_id"] == vid
    assert body["status"] == "pending"
    assert body["motion_status"] == "pending"
    assert body["gapper_status"] == "pending"


def test_memory_query_empty_by_default(client):
    r = client.post("/api/v1/video/process",
                    json={"video_url": "https://x.com/v.mp4"})
    vid = r.json()["video_id"]
    q = client.get(f"/api/v1/memory/{vid}/query", params={"q": "anything"})
    assert q.status_code == 200
    assert q.json() == {"video_id": vid, "results": []}
