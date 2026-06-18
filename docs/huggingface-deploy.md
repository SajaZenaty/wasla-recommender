# Deploying on Hugging Face Spaces (free, 16 GB RAM)

The free CPU tier on Hugging Face Spaces provides 2 vCPU and 16 GB RAM, which
is more than enough for the PyTorch + sentence-transformers stack. No changes to
the recommendation/model code are required — only deployment configuration.

The repo is already prepared:

- `README.md` has the Space config frontmatter (`sdk: docker`, `app_port: 7860`).
- The `Dockerfile` runs as UID 1000 (the Spaces runtime user), bakes the model
  into a writable cache, and listens on port 7860.

## Option A — Manual push (simplest)

1. Create a new Space at https://huggingface.co/new-space
   - SDK: Docker
   - Hardware: CPU basic (free)
2. Add the Space as a git remote and push:

```bash
# Replace <user> and <space-name> with your values
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space Ai-integration:main
```

3. The Space builds the Dockerfile automatically. First build takes a few
   minutes (it downloads the model into the image).

## Option B — Auto-sync from GitHub

Use the included workflow `.github/workflows/hf-sync.yml`. It mirrors the
branch to the Space on every push.

Setup:

1. Create the Space (as in Option A, step 1).
2. Create a Hugging Face access token with write scope:
   https://huggingface.co/settings/tokens
3. In the GitHub repo: Settings -> Secrets and variables -> Actions, add:
   - `HF_TOKEN` = the token
   - `HF_SPACE` = `<user>/<space-name>`

## Environment variables (Space settings -> Variables and secrets)

| Variable | Value | Notes |
|----------|-------|-------|
| `USE_MOCK_DATA` | `true` | Start with mock data until Express is wired. |
| `ENABLE_SCHEDULER` | `false` | No persistent storage on the free tier. |
| `INDEX_SNAPSHOT_PATH` | `/home/user/data/index_snapshot.pkl` | Writable path (already the default). |
| `RECOMMENDER_API_KEY` | your secret | Mark as a secret. Required for `/sync/*`. |
| `EXPRESS_INTERNAL_URL` | your Express URL | Set when connecting the backend. |

Do not set `PORT`; the Dockerfile already targets 7860, which matches
`app_port` in the README frontmatter.

## After deploy

The Space exposes a public URL like:

```
https://<user>-<space-name>.hf.space/health
https://<user>-<space-name>.hf.space/ready
https://<user>-<space-name>.hf.space/docs
```

`/health` should return `{"status": "ok", "model_loaded": true}` and `/ready`
should report `ready: true` once bootstrap completes.

## Notes

- Free Spaces sleep after inactivity; the first request after sleep is slow
  while the container wakes. This is fine for development and the scheduler is
  disabled accordingly.
- Storage is ephemeral on the free tier, so the snapshot is rebuilt on each
  cold start (from mock data or an Express pull).
