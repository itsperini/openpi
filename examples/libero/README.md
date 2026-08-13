# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

To connect the simulator to an authenticated Modal deployment, export the proxy token variables and pass the full WebSocket URL:

```bash
python examples/libero/main.py \
  --args.modal-endpoint "$OPENPI_LIBERO_MODAL_ENDPOINT" \
  --args.task-id 0 \
  --args.num-trials-per-task 1 \
  --args.max-steps 80 \
  --args.display
```

This runs one short Franka Panda task, opens an agent-camera preview, and saves an MP4. Press `q` or Escape to stop early.

### Apple Silicon Mac

The stock LIBERO container targets an NVIDIA Linux host. On an Apple Silicon Mac, use the CPU-only headless runtime while keeping inference on Modal:

```bash
docker build -t openpi-libero-macos -f examples/libero/Dockerfile.macos .

docker run --rm \
  --env-file .env \
  -v "$PWD:/app" \
  openpi-libero-macos \
  python examples/libero/main.py \
  --args.modal-endpoint "$OPENPI_LIBERO_MODAL_ENDPOINT" \
  --args.task-id 0 \
  --args.num-trials-per-task 1 \
  --args.max-steps 80
```

This writes the Franka camera rollout beneath `data/libero/videos`. The container is headless because Docker Desktop does not expose a native MuJoCo GUI from its Linux VM; play the resulting MP4 on the Mac.

If Docker Desktop is unavailable, the simulator can run natively on Apple Silicon:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv venv --python 3.11 .venv-libero-mac
UV_CACHE_DIR="$PWD/.uv-cache" uv pip install \
  --python .venv-libero-mac/bin/python \
  torch==2.9.0 \
  -r examples/libero/requirements.macos.txt

set -a
source .env
set +a

.venv-libero-mac/bin/python examples/libero/run_macos.py \
  --args.modal-endpoint "$OPENPI_LIBERO_MODAL_ENDPOINT" \
  --args.task-id 0 \
  --args.num-trials-per-task 1 \
  --args.max-steps 80 \
  --args.display
```

The Mac launcher configures LIBERO, CGL rendering, and the compatibility setting needed to load LIBERO's trusted initial-state assets.

To verify MuJoCo and the Franka locally before starting Modal, run a mock trajectory that requires no policy server:

```bash
.venv-libero-mac/bin/python examples/libero/mock_macos.py
```

This opens a live agent-camera window and saves `data/libero/videos/franka_mock_trajectory.mp4`.

### Synchronized episode inspector

Policy rollouts also write a synchronized episode beneath `data/libero/episodes`. Each episode contains the MuJoCo overview, both policy camera videos, `metadata.json`, and per-step telemetry in `trajectory.jsonl`.

Start the Vite dashboard after recording an episode:

```bash
cd examples/libero/visualizer
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. A single timeline drives all three videos, joint state, end-effector position, executed actions, predicted action chunks, and inference metadata.

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
