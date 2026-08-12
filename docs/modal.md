# Running the OpenPI policy server on Modal

This setup runs the full OpenPI model and checkpoint on a Modal NVIDIA GPU. The robot or simulator runs locally and connects through the lightweight `openpi-client` package over an authenticated WebSocket.

## 1. Create the local environment file

From the repository root:

```bash
cp .env.example .env
```

The real `.env` file is ignored by Git. Put all local credentials and endpoint settings in this file; never add secrets to `.env.example`.

There are two different Modal credential pairs:

- `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` authorize `modal deploy`. Create or configure them with `modal token new`, or copy a service-user token from the Modal dashboard.
- `MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET` protect the public WebSocket endpoint. Create them with `modal workspace proxy-tokens create`.

Fill in these four values in the repository-root `.env` file:

```dotenv
MODAL_TOKEN_ID=ak-...
MODAL_TOKEN_SECRET=as-...
MODAL_PROXY_TOKEN_ID=wk-...
MODAL_PROXY_TOKEN_SECRET=ws-...
```

If the Modal CLI is already authenticated through `~/.modal.toml`, the first pair may remain empty. The proxy-token pair is still required by the Mac client.

## 2. Choose the OpenPI revision and policy

The Modal image clones the fork over HTTPS and checks out the exact revision in `OPENPI_GIT_REF`. The default is the commit from which this branch was created. After pushing new server-side changes, replace it with the pushed commit SHA:

```dotenv
OPENPI_REPOSITORY=https://github.com/itsperini/openpi.git
OPENPI_GIT_REF=<pushed-commit-sha>
```

The default policy is π0.5 DROID:

```dotenv
OPENPI_SERVER_ARGS="--env=DROID"
```

Other built-in environments are `ALOHA`, `ALOHA_SIM`, and `LIBERO`. To serve an explicit config and checkpoint, use arguments such as:

```dotenv
OPENPI_SERVER_ARGS="policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid"
```

## 3. Deploy

Install the Modal CLI in a small local virtual environment:

```bash
python3 -m venv .venv-modal
source .venv-modal/bin/activate
pip install modal
```

Deploy with the wrapper that loads the repository-root `.env` file:

```bash
./scripts/deploy_modal.sh
```

The first deployment builds the CUDA/OpenPI image. The first server start also downloads the selected checkpoint into the persistent `openpi-checkpoints` Modal Volume.

Modal prints an `https://...modal.run` endpoint after deployment. Copy it into `.env`, changing only the scheme to `wss://`:

```dotenv
OPENPI_MODAL_ENDPOINT=wss://YOUR-WORKSPACE--openpi-openpi-server.modal.run
```

## 4. Install and run the Mac client

The Mac only needs the client package:

```bash
python3 -m venv .venv-client
source .venv-client/bin/activate
pip install -e packages/openpi-client

set -a
source .env
set +a
```

The client reads the endpoint and proxy credentials from the exported environment:

```python
from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy.from_modal()
result = policy.infer(observation)
actions = result["actions"]
```

Resize camera images to 224×224 before sending them. Modal limits individual WebSocket messages to 2 MiB.

## Capacity and lifecycle

The defaults use an L40S GPU, a single container, and a 20-minute idle window:

```dotenv
OPENPI_GPU=L40S
OPENPI_MIN_CONTAINERS=0
```

Keeping `OPENPI_MIN_CONTAINERS=0` allows the GPU to shut down between experiments. Set it to `1` only when avoiding cold-start latency is worth continuously reserving and billing the GPU.

Each WebSocket connection is allowed to live for up to 24 hours. The client automatically reconnects and retries one inference if the connection closes, but robot code should still enter a safe state whenever communication is lost or an inference call fails.
