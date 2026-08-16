---
name: vectordb-check
description: Ensures Qdrant (the vector DB backend) is reachable before running the RAG pipeline's build or eval scripts. Use this BEFORE running run/build.sh, run/eval.sh, or run/local_run.sh — not needed for run/dist_run.sh, which never touches Qdrant.
---

# Vector DB check

`run/build.sh` always tries to create/update Qdrant collections (the "Vector DB (qdrant)"
build step), and `run/eval.sh` needs Qdrant reachable whenever `RETRIEVAL_TYPE="vectordb"`.
Both fail loudly if Qdrant isn't reachable — `build.sh` exits with `sys.exit(1)` right
after printing `VectorDB failed, exiting`, *after* chunking/embedding/indexing have
already run (wasting all of that time for nothing). Run this check first, every time,
before invoking any of those three scripts — no need to ask the user first, this is
read-only/idempotent until step 3.

## 1. Check if Qdrant is already up

```bash
curl -sf -m 3 http://localhost:6333 >/dev/null && echo UP || echo DOWN
```

If `UP`, stop here — nothing to do, go run the build/eval script.

## 2. Start Docker Desktop if it isn't running

This machine runs **Docker Desktop** as a user-level systemd service, not the plain
`docker.io`/`docker-ce` system service — so no `sudo` is needed:

```bash
systemctl --user start docker-desktop
```

Safe to call even if it's already running (no-op). Then wait for the daemon socket to
come up — it takes a few seconds after a cold start of the Desktop VM:

```bash
for i in $(seq 1 24); do
  docker ps >/dev/null 2>&1 && break
  sleep 5
done
```

If `docker ps` still fails after ~2 minutes, stop and report it to the user rather than
retrying indefinitely.

## 3. Start the Qdrant container

Qdrant's data is bind-mounted at `/home/tanzima/Documents/AI/qdrant` (see `config.py`'s
`DOCKER_PORT` comment and `qdrant_setup.md`), so reuse the existing container instead of
starting a fresh one — that preserves every collection already built (5K/20K/50K docs,
all three chunking strategies) instead of starting from empty.

```bash
docker ps -a --filter ancestor=qdrant/qdrant --format '{{.Names}}\t{{.Status}}\t{{.CreatedAt}}'
```

- **A container is listed and already `Up`**: nothing to do, skip to step 4.
- **A container is listed but stopped/exited**: start the most recently created one (by
  `CreatedAt`) — this reuses the persisted collections:
  ```bash
  docker start <container_name>
  ```
  If more than one qdrant container is listed, only start the most recent one; leave any
  older stopped ones alone rather than removing them — they may be intentional backups,
  not yours to delete without asking.
- **No qdrant container exists at all** (fresh machine, or the volume was never
  initialized): run one fresh — the bind mount means the data still persists:
  ```bash
  docker run -d -p 6333:6333 -p 6334:6334 \
    -v /home/tanzima/Documents/AI/qdrant:/qdrant/storage qdrant/qdrant
  ```

## 4. Confirm it's reachable

```bash
for i in $(seq 1 10); do
  curl -sf -m 3 http://localhost:6333 >/dev/null && break
  sleep 2
done
curl -s http://localhost:6333
```

Expect `{"title":"qdrant - vector search engine", ...}`. If it never comes up, stop and
show the container logs instead of proceeding:

```bash
docker logs --tail 30 <container_name>
```

## 5. Proceed

Once Qdrant answers, go ahead and run the build/eval script the user actually asked for.
