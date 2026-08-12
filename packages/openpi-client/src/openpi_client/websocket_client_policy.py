import logging
import os
import time
from typing import Any, Dict, Mapping, Optional, Tuple

from typing_extensions import override
import websockets.exceptions
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        additional_headers: Optional[Mapping[str, str]] = None,
        reconnect: bool = True,
        retry_delay: float = 5.0,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._additional_headers = dict(additional_headers or {})
        if self._api_key is not None and "Authorization" in self._additional_headers:
            raise ValueError("api_key and an Authorization entry in additional_headers cannot be used together")
        self._reconnect = reconnect
        self._retry_delay = retry_delay
        self._ws, self._server_metadata = self._wait_for_server()

    @classmethod
    def from_modal(
        cls,
        endpoint: Optional[str] = None,
        token_id: Optional[str] = None,
        token_secret: Optional[str] = None,
        **kwargs: Any,
    ) -> "WebsocketClientPolicy":
        """Connect to a Modal proxy-authenticated WebSocket endpoint.

        Values can be passed explicitly or supplied through OPENPI_MODAL_ENDPOINT,
        MODAL_PROXY_TOKEN_ID, and MODAL_PROXY_TOKEN_SECRET.
        """
        endpoint = endpoint or os.getenv("OPENPI_MODAL_ENDPOINT")
        token_id = token_id or os.getenv("MODAL_PROXY_TOKEN_ID")
        token_secret = token_secret or os.getenv("MODAL_PROXY_TOKEN_SECRET")

        missing = [
            name
            for name, value in (
                ("OPENPI_MODAL_ENDPOINT", endpoint),
                ("MODAL_PROXY_TOKEN_ID", token_id),
                ("MODAL_PROXY_TOKEN_SECRET", token_secret),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing Modal connection settings: {', '.join(missing)}")
        if "additional_headers" in kwargs:
            raise ValueError("from_modal manages additional_headers; pass other client options only")

        return cls(
            endpoint,
            additional_headers={
                "Modal-Key": token_id,
                "Modal-Secret": token_secret,
            },
            **kwargs,
        )

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _headers(self) -> Optional[Dict[str, str]]:
        headers = dict(self._additional_headers)
        if self._api_key:
            headers["Authorization"] = f"Api-Key {self._api_key}"
        return headers or None

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=self._headers(),
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except (OSError, TimeoutError) as error:
                logging.info(f"Still waiting for server ({error})...")
                time.sleep(self._retry_delay)

    def _infer_once(self, data: bytes) -> Dict:
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # We're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        try:
            return self._infer_once(data)
        except websockets.exceptions.ConnectionClosed:
            if not self._reconnect:
                raise
            logging.warning("WebSocket connection closed; reconnecting and retrying inference once")
            self._ws, self._server_metadata = self._wait_for_server()
            return self._infer_once(data)

    @override
    def reset(self) -> None:
        pass
