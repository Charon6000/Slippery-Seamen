import json

from main import app


def test_upload_without_file_returns_400():
    client = app.test_client()
    resp = client.post("/", data={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert data.get("error") == "no file uploaded"
