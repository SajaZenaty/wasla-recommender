"""Pydantic request/response models for the recommender API."""
from pydantic import BaseModel, ConfigDict, Field


class RecommendRequest(BaseModel):
    user_id: int | str
    top_k: int | None = Field(default=None, ge=1)


class RecommendItem(BaseModel):
    post_id: int | str
    final_score: float
    post_type: str | None = None
    breakdown: dict[str, float]


class RecommendResponse(BaseModel):
    user_id: int | str
    count: int
    recommendations: list[RecommendItem]


class PostIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    post_id: int | str
    user_id: int | str
    post_type: str
    category: str
    title: str
    description: str
    service_mode: str | None = None
    location: str | None = None
    time_credits: float = 0
    timestamp: str | None = None


class InteractionIn(BaseModel):
    user_id: int | str
    post_id: int | str
    action: str
    timestamp: str | None = None


class UserIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_id: int | str
    skills: list[str] = []
    needs: list[str] = []
    location: str | None = None
    time_balance: float = 0
    trust_score: float = 0


class UsersSyncRequest(BaseModel):
    users: list[UserIn]


class BootstrapRequest(BaseModel):
    """Optional inline snapshot. If omitted, data is pulled from Express."""

    model_config = ConfigDict(extra="allow")

    users: list[dict] | None = None
    posts: list[dict] | None = None
    interactions: list[dict] | None = None


class StatusResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    ready: bool
    can_serve_recommendations: bool = False
    model_loaded: bool
    posts: int
    users: int
    interactions: int
    last_bootstrap_at: str | None = None
    pending_rebuild: bool
