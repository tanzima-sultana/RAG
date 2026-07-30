# Qdrant Vector DB Setup (Docker, Ubuntu)

## 1. Install Docker Engine

Set up Docker's official apt repository:

```bash
# Remove any conflicting packages
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

# Add Docker's official GPG key
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
```

Download the Docker Desktop `.deb` package from docker.com, then install:

```bash
sudo apt install ./docker-desktop-amd64.deb
```

Launch Docker Desktop from the applications menu, or:

```bash
systemctl --user start docker-desktop
```

Sign-in is optional — skip it if the button doesn't work; local Docker still works without an account.

Verify:

```bash
docker --version
docker ps
```

## 2. Run Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v <qdrant-dir>/qdrant:/qdrant/storage qdrant/qdrant
```

- `-d` runs it in the background
- `-p 6333:6333` exposes the REST API
- `-p 6334:6334` exposes gRPC
- `-v` bind-mounts local storage so data persists across container restarts/recreations

Verify it's up:

```bash
curl http://localhost:6333
```

Expected response: `{"title":"qdrant - vector search engine", ...}`

## 3. Install the Python client

Activate your pyenv environment first, then:

```bash
pip install qdrant-client
```

Verify from Python:

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
print(client.get_collections())
```

Expected output: `collections=[]`

## 4. Daily startup

Docker Desktop must be running first. Then either:

**Option A — restart the existing container:**

```bash
docker ps -a
docker start <container_name_or_id>
```

**Option B — run a fresh container (data persists via the mounted volume):**

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v <qdrant-dir>/qdrant:/qdrant/storage qdrant/qdrant
```
