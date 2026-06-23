# Express Integration Contract

This document is for the Node.js / Express backend team. The recommender is a
separate HTTP service; Express remains the system of record and orchestrates
all calls. No recommendation logic is ported to Node.

## 1. Field contract

The recommender expects these exact field names. Map your DB models to them
(for example, stringify Mongo `_id`s consistently across users, posts, and
interactions so they join correctly).

### User

| Field | Type | Notes |
|-------|------|-------|
| `user_id` | int or string | Stable unique id. |
| `skills` | string[] | Categories the user can provide. |
| `needs` | string[] | Categories the user wants. |
| `location` | string | City name. |
| `time_balance` | number | Time credits available. |
| `trust_score` | number | 0–5. |

### Post

| Field | Type | Notes |
|-------|------|-------|
| `post_id` | int or string | Stable unique id. |
| `user_id` | int or string | Author id. |
| `post_type` | string | `عرض` (offer) or `طلب` (request). |
| `category` | string | One category. |
| `title` | string | |
| `description` | string | |
| `service_mode` | string | `الكتروني` (online) or `وجاهي` (in person). |
| `location` | string | City name. |
| `time_credits` | number | Cost of the post. |
| `timestamp` | ISO 8601 string | Creation time. |

### Interaction

| Field | Type | Notes |
|-------|------|-------|
| `user_id` | int or string | Who acted. |
| `post_id` | int or string | Target post. |
| `action` | string | `click`, `save`, or `apply`. |
| `timestamp` | ISO 8601 string | When it happened. |

## 2. Export endpoint (Express implements)

Used by the recommender for the initial bootstrap and the nightly full rebuild.

```
GET /internal/recommender-export
Header: X-Internal-Token: <RECOMMENDER_API_KEY>
```

Response:

```json
{
  "users": [ /* User objects */ ],
  "posts": [ /* Post objects */ ],
  "interactions": [ /* Interaction objects */ ]
}
```

Protect it with the same shared secret the recommender uses. It should not be
publicly reachable.

## 3. Event push hooks (Express calls the recommender)

Push changes as they happen so the next recommendation reflects them. All
requests send `X-Internal-Token`.

```js
const axios = require("axios");

const rec = axios.create({
  baseURL: process.env.RECOMMENDER_URL,            // e.g. http://recommender:8000
  timeout: Number(process.env.RECOMMENDER_TIMEOUT_MS || 5000),
  headers: { "X-Internal-Token": process.env.RECOMMENDER_API_KEY },
});

// After creating or updating a post
await rec.post("/sync/post", mapPost(post));

// After a user clicks / saves / applies
await rec.post("/sync/interaction", {
  user_id: String(userId),
  post_id: String(postId),
  action,                                          // "click" | "save" | "apply"
  timestamp: new Date().toISOString(),
});

// After a user updates their profile
await rec.post("/sync/users", { users: [mapUser(user)] });
```

Pushes should be fire-and-forget relative to the user request (await in a
background job or catch and log errors) so recommender hiccups never block your
API.

## 4. Feed proxy (Express implements)

Express asks the recommender for an ordered list of post IDs, then hydrates the
full posts from its own DB, preserving the recommender's order.

```js
router.get("/feed/:userId", auth, async (req, res) => {
  let order = [];
  try {
    const { data } = await rec.post("/recommend", {
      user_id: String(req.params.userId),
      top_k: 20,
    });
    order = data.recommendations.map((r) => r.post_id);
  } catch (err) {
    // Fallback: recommender unavailable -> chronological feed.
    const posts = await Post.find().sort({ createdAt: -1 }).limit(20);
    return res.json(posts);
  }

  const posts = await Post.find({ _id: { $in: order } });
  const byId = new Map(posts.map((p) => [String(p._id), p]));
  res.json(order.map((id) => byId.get(String(id))).filter(Boolean));
});
```

## 5. Fallback contract

If the recommender returns `503` (index not ready / rebuilding) or times out,
Express must serve a fallback feed (chronological or category-based). The
recommender always keeps serving its last good index during a failed rebuild.

If the recommender returns `404` with `detail.error === "user_not_found"`, the
user exists in Express but was never synced to the recommender index. Options:

1. **Preferred:** push the profile immediately, then retry once:

```js
if (err.response?.status === 404 && err.response?.data?.detail?.error === "user_not_found") {
  await rec.post("/sync/users", { users: [mapUser(user)] }).catch(log);
  const { data } = await rec.post("/recommend", { user_id: String(userId), top_k: 20 });
  // ...
}
```

2. **Root cause:** ensure `GET /internal/recommender-export` includes all users
   and that `user_id` matches what you send to `/recommend` (same string/int
   representation). After fixing export, call `POST /sync/bootstrap` on the
   recommender.

Check recommender readiness with `GET /ready`: `can_serve_recommendations` must
be `true` and `users` must be greater than zero.

## 6. Environment (Express side)

```
RECOMMENDER_URL=http://recommender:8000
RECOMMENDER_API_KEY=<same value as the recommender service>
RECOMMENDER_TIMEOUT_MS=5000
```
