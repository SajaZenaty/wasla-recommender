"""Static test fixtures shaped like Express export payloads."""

from datetime import datetime, timezone

from src.data.express_loader import frames_from_payload


def sample_bootstrap_payload():
    now = datetime.now(timezone.utc).isoformat()
    return {
        "users": [
            {
                "user_id": 0,
                "skills": ["تصميم"],
                "needs": ["برمجة"],
                "location": "غزة",
                "time_balance": 10,
                "trust_score": 3,
            },
            {
                "user_id": 1,
                "skills": ["برمجة"],
                "needs": ["تصميم"],
                "location": "خانيونس",
                "time_balance": 8,
                "trust_score": 4,
            },
            {
                "user_id": 2,
                "skills": ["تعليم"],
                "needs": ["ترجمة"],
                "location": "رفح",
                "time_balance": 12,
                "trust_score": 2,
            },
            {
                "user_id": 3,
                "skills": ["صيانة"],
                "needs": ["طبخ"],
                "location": "غزة",
                "time_balance": 5,
                "trust_score": 3,
            },
            {
                "user_id": 4,
                "skills": ["ترجمة"],
                "needs": ["تصوير"],
                "location": "جباليا",
                "time_balance": 15,
                "trust_score": 5,
            },
            {
                "user_id": 5,
                "skills": ["طبخ"],
                "needs": ["تنظيف"],
                "location": "دير البلح",
                "time_balance": 6,
                "trust_score": 2,
            },
            {
                "user_id": 6,
                "skills": ["تصوير"],
                "needs": ["تعليم"],
                "location": "المغازي",
                "time_balance": 9,
                "trust_score": 4,
            },
            {
                "user_id": 7,
                "skills": ["تنظيف"],
                "needs": ["صيانة"],
                "location": "البريج",
                "time_balance": 7,
                "trust_score": 3,
            },
        ],
        "posts": [
            {
                "post_id": 100,
                "user_id": 1,
                "post_type": "عرض",
                "category": "برمجة",
                "title": "خدمة برمجة",
                "description": "أقدم خدمات برمجة وتطوير مواقع",
                "service_mode": "الكتروني",
                "location": "خانيونس",
                "time_credits": 2,
                "timestamp": now,
            },
            {
                "post_id": 101,
                "user_id": 2,
                "post_type": "عرض",
                "category": "تعليم",
                "title": "دروس خصوصية",
                "description": "أقدم دروس في الرياضيات والعلوم",
                "service_mode": "وجاهي",
                "location": "رفح",
                "time_credits": 1,
                "timestamp": now,
            },
            {
                "post_id": 102,
                "user_id": 3,
                "post_type": "طلب",
                "category": "طبخ",
                "title": "أحتاج مساعدة في الطبخ",
                "description": "بدور على شخص يساعدني في تحضير وجبات",
                "service_mode": "وجاهي",
                "location": "غزة",
                "time_credits": 1,
                "timestamp": now,
            },
            {
                "post_id": 103,
                "user_id": 4,
                "post_type": "عرض",
                "category": "ترجمة",
                "title": "خدمات ترجمة",
                "description": "ترجمة من العربية إلى الإنجليزية",
                "service_mode": "الكتروني",
                "location": "جباليا",
                "time_credits": 1,
                "timestamp": now,
            },
            {
                "post_id": 104,
                "user_id": 5,
                "post_type": "عرض",
                "category": "تصميم",
                "title": "تصميم شعارات",
                "description": "أصمم شعارات وهويات بصرية",
                "service_mode": "الكتروني",
                "location": "دير البلح",
                "time_credits": 2,
                "timestamp": now,
            },
            {
                "post_id": 105,
                "user_id": 6,
                "post_type": "عرض",
                "category": "تصوير",
                "title": "تصوير فعاليات",
                "description": "تصوير احترافي للمناسبات",
                "service_mode": "وجاهي",
                "location": "المغازي",
                "time_credits": 3,
                "timestamp": now,
            },
        ],
        "interactions": [
            {
                "user_id": 0,
                "post_id": 100,
                "action": "click",
                "timestamp": now,
            },
            {
                "user_id": 0,
                "post_id": 101,
                "action": "save",
                "timestamp": now,
            },
            {
                "user_id": 1,
                "post_id": 104,
                "action": "apply",
                "timestamp": now,
            },
        ],
    }


def load_sample_frames():
    return frames_from_payload(sample_bootstrap_payload())
