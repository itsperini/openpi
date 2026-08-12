import pytest
import websockets.exceptions

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


class _FakeConnection:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.sent = []

    def send(self, data):
        self.sent.append(data)

    def recv(self):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_forwards_additional_headers(monkeypatch):
    calls = []
    connection = _FakeConnection(
        [
            msgpack_numpy.packb({"model": "pi05_droid"}),
            msgpack_numpy.packb({"actions": [1, 2, 3]}),
        ]
    )

    def connect(uri, **kwargs):
        calls.append((uri, kwargs))
        return connection

    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", connect)

    policy = websocket_client_policy.WebsocketClientPolicy(
        "wss://example.modal.run",
        additional_headers={"Modal-Key": "wk-test", "Modal-Secret": "ws-test"},
    )

    assert policy.get_server_metadata() == {"model": "pi05_droid"}
    assert policy.infer({"prompt": "test"}) == {"actions": [1, 2, 3]}
    assert calls[0][0] == "wss://example.modal.run"
    assert calls[0][1]["additional_headers"] == {
        "Modal-Key": "wk-test",
        "Modal-Secret": "ws-test",
    }


def test_from_modal_reads_environment(monkeypatch):
    calls = []
    connection = _FakeConnection([msgpack_numpy.packb({})])

    def connect(uri, **kwargs):
        calls.append((uri, kwargs))
        return connection

    monkeypatch.setenv("OPENPI_MODAL_ENDPOINT", "wss://example.modal.run")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-test")
    monkeypatch.setattr(websocket_client_policy.websockets.sync.client, "connect", connect)

    websocket_client_policy.WebsocketClientPolicy.from_modal()

    assert calls[0][0] == "wss://example.modal.run"
    assert calls[0][1]["additional_headers"] == {
        "Modal-Key": "wk-test",
        "Modal-Secret": "ws-test",
    }


def test_from_modal_reports_missing_environment(monkeypatch):
    monkeypatch.delenv("OPENPI_MODAL_ENDPOINT", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET", raising=False)

    with pytest.raises(ValueError, match="OPENPI_MODAL_ENDPOINT"):
        websocket_client_policy.WebsocketClientPolicy.from_modal()


def test_reconnects_and_retries_inference_once(monkeypatch):
    closed = websockets.exceptions.ConnectionClosedOK(None, None)
    connections = iter(
        [
            _FakeConnection([msgpack_numpy.packb({"generation": 1}), closed]),
            _FakeConnection(
                [
                    msgpack_numpy.packb({"generation": 2}),
                    msgpack_numpy.packb({"actions": [4, 5, 6]}),
                ]
            ),
        ]
    )

    monkeypatch.setattr(
        websocket_client_policy.websockets.sync.client,
        "connect",
        lambda *args, **kwargs: next(connections),
    )

    policy = websocket_client_policy.WebsocketClientPolicy("wss://example.modal.run")

    assert policy.infer({"prompt": "test"}) == {"actions": [4, 5, 6]}
    assert policy.get_server_metadata() == {"generation": 2}
