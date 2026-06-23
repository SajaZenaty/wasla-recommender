from datetime import datetime, timezone


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "Wasla Recommender"
    assert body["ready"] is True
    assert body["endpoints"]["health"] == "/health"


def test_ready(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["can_serve_recommendations"] is True
    assert body["posts"] > 0
    assert body["users"] > 0
    assert body["issue"] is None
    assert body["data_source"] is not None


def test_recommend_known_user(client):
    # Mock data always assigns user_id 0..n-1.
    resp = client.post("/recommend", json={"user_id": 0, "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == 0
    assert len(body["recommendations"]) <= 5


def test_recommend_unknown_user_returns_404(client):
    resp = client.post("/recommend", json={"user_id": 10_000_001})
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "user_not_found"


def test_sync_bootstrap_with_empty_interactions(client):
    resp = client.post(
        "/sync/bootstrap",
        json={
            "users": [
                {
                    "user_id": 1,
                    "skills": ["برمجة"],
                    "needs": ["تصميم"],
                    "location": "غزة",
                    "time_balance": 10,
                    "trust_score": 3,
                }
            ],
            "posts": [
                {
                    "post_id": 10,
                    "user_id": 2,
                    "post_type": "عرض",
                    "category": "برمجة",
                    "title": "خدمة برمجة",
                    "description": "أقدم خدمات برمجة",
                    "location": "غزة",
                    "time_credits": 1,
                }
            ],
            "interactions": [],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["posts"] == 1

    rec = client.post("/recommend", json={"user_id": 1, "top_k": 5})
    assert rec.status_code == 200


def test_recommend_accepts_string_user_id(client):
    resp = client.post("/recommend", json={"user_id": "0", "top_k": 3})
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "0"


def test_sync_bootstrap_invalid_payload_returns_400(client):
    resp = client.post(
        "/sync/bootstrap",
        json={
            "users": [{"user_id": 1, "skills": [], "needs": []}],
            "posts": [{"post_id": 1, "user_id": 2, "category": "x"}],
            "interactions": [],
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "invalid_payload"


def test_sync_post_then_recommend_includes_it(client):
    # Create a controlled user and a matching offer, independent of mock RNG.
    user_id = 999_001
    post_id = 999_002

    client.post(
        "/sync/users",
        json={
            "users": [
                {
                    "user_id": user_id,
                    "skills": [],
                    "needs": ["برمجة"],
                    "location": "غزة",
                    "time_balance": 10,
                    "trust_score": 3,
                }
            ]
        },
    )

    client.post(
        "/sync/post",
        json={
            "post_id": post_id,
            "user_id": 999_003,
            "post_type": "عرض",
            "category": "برمجة",
            "title": "خدمة برمجة",
            "description": "أقدم خدمات برمجة",
            "service_mode": "الكتروني",
            "location": "غزة",
            "time_credits": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    resp = client.post("/recommend", json={"user_id": user_id, "top_k": 50})
    assert resp.status_code == 200
    post_ids = [r["post_id"] for r in resp.json()["recommendations"]]
    assert post_id in post_ids
