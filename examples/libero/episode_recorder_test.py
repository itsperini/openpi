import json

from examples.libero.episode_recorder import EpisodeRecorder


def test_finalize_records_vm_serving_provenance(tmp_path):
    recorder = EpisodeRecorder(
        tmp_path,
        task_suite="libero_spatial",
        task_id=0,
        trial_id=0,
        instruction="pick up the bowl",
        control_hz=20,
        replan_steps=5,
        server_metadata={},
        inference_backend="vm",
        inference_transport="ssh_tunnel",
        inference_endpoint="ws://127.0.0.1:8000",
    )

    episode_path = recorder.finalize(success=False)
    metadata = json.loads((episode_path / "metadata.json").read_text())
    index = json.loads((tmp_path / "index.json").read_text())

    assert metadata["schema_version"] == 3
    assert metadata["serving"] == {
        "backend": "vm",
        "transport": "ssh_tunnel",
        "endpoint": "ws://127.0.0.1:8000",
    }
    assert index["schema_version"] == 2
    assert index["episodes"][0]["inference_backend"] == "vm"
    assert index["episodes"][0]["inference_transport"] == "ssh_tunnel"
