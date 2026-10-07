"""Tests for ai_diffusion.connection.Connection using MockClient."""

from __future__ import annotations

import asyncio

import pytest

from ai_diffusion.backend.client import ClientEvent, ClientMessage, MissingResources
from ai_diffusion.backend.network import NetworkError
from ai_diffusion.backend.resources import Arch
from ai_diffusion.model.connection import Connection, ConnectionState

from .conftest import qtapp
from .mock.client import MockClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_for_state(conn: Connection, *exclude: ConnectionState, timeout: int = 100):
    """Yield to the event loop until conn.state is not in *exclude*."""
    for _ in range(timeout):
        await asyncio.sleep(0)
        if conn.state not in exclude:
            return
    raise TimeoutError(f"State stuck at {conn.state!r} after {timeout} iterations")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@qtapp
async def test_connect_success_and_disconnect():
    """Connection.connect transitions to connected; explicit disconnect resets state."""
    conn = Connection()
    client = MockClient()

    states: list[ConnectionState] = []
    conn.state_changed.connect(states.append)

    conn.connect(client)
    await _wait_for_state(conn, ConnectionState.connecting, ConnectionState.disconnected)

    assert conn.state is ConnectionState.connected
    assert client.connected is True
    assert ConnectionState.connecting in states
    assert ConnectionState.connected in states

    await asyncio.sleep(0)
    await conn.disconnect()

    assert conn.state is ConnectionState.disconnected
    assert client.connected is False
    assert ConnectionState.disconnected in states


@pytest.mark.parametrize(
    "state", [ConnectionState.disconnected, ConnectionState.auth_pending, ConnectionState.error]
)
@qtapp
async def test_reconnect_replaces_listener_even_when_ui_state_changed(state):
    conn = Connection()
    old, new = MockClient(), MockClient()
    received = []
    conn.message_received.connect(received.append)
    try:
        await conn._connect(old)
        await asyncio.sleep(0)
        conn.state = state
        await conn._connect(new)
        new.push(ClientMessage(ClientEvent.finished, "new-job", 1))
        for _ in range(20):
            await asyncio.sleep(0)
        assert old.disconnect_count == 1
        assert [msg.job_id for msg in received] == ["new-job"]
        assert conn.client is new
    finally:
        await conn.disconnect()


@qtapp
async def test_failed_listener_can_be_replaced():
    class FailingClient(MockClient):
        async def listen(self):
            yield ClientMessage(ClientEvent.progress, "old-job", 0.5)
            raise RuntimeError("Listener failed")

    conn = Connection()
    old, new = FailingClient(), MockClient()
    received = []
    conn.message_received.connect(received.append)
    try:
        await conn._connect(old)
        for _ in range(10):
            await asyncio.sleep(0)
        assert conn.state is ConnectionState.error
        assert "Listener failed" in conn.error
        await conn._connect(new)
        new.push(ClientMessage(ClientEvent.finished, "new-job", 1))
        for _ in range(20):
            await asyncio.sleep(0)
        assert conn.client is new
        assert received[-1].job_id == "new-job"
        assert old.disconnect_count == 1
    finally:
        await conn.disconnect()


@qtapp
async def test_connect_network_error():
    """Connection.connect with NetworkError ends up in error state."""
    conn = Connection()
    client = MockClient()
    client.connect_error = NetworkError(0, "Connection refused", "ws://mock")

    states: list[ConnectionState] = []
    conn.state_changed.connect(states.append)

    conn.connect(client)
    await _wait_for_state(conn, ConnectionState.connecting, ConnectionState.disconnected)

    assert conn.state is ConnectionState.error
    assert conn.error_kind == "network"
    assert conn.error != ""
    assert ConnectionState.connecting in states
    assert ConnectionState.error in states


@qtapp
async def test_connect_missing_resources():
    """Connection.connect with MissingResources ends up in error state."""
    conn = Connection()
    client = MockClient()
    missing = MissingResources({Arch.sd15: []})
    client.connect_error = missing

    states: list[ConnectionState] = []
    conn.state_changed.connect(states.append)

    conn.connect(client)
    await _wait_for_state(conn, ConnectionState.connecting, ConnectionState.disconnected)

    assert conn.state is ConnectionState.error
    assert conn.error_kind == "missing_resources"
    assert conn.missing_resources is missing
    assert ConnectionState.error in states


@qtapp
async def test_listen_messages_and_reconnect():
    """Client messages are forwarded; a disconnect followed by reconnect clears the error."""
    conn = Connection()
    client = MockClient()

    # Seed the messages that listen() will deliver immediately after connect
    client.messages = [
        ClientMessage(ClientEvent.progress, "job-1", 0.3),
        ClientMessage(ClientEvent.disconnected, ""),
        ClientMessage(ClientEvent.connected, ""),
        ClientMessage(ClientEvent.finished, "job-1", 1.0),
    ]

    received: list[ClientMessage] = []
    conn.message_received.connect(received.append)

    errors: list[str] = []
    conn.error_changed.connect(errors.append)

    conn.connect(client)
    await _wait_for_state(conn, ConnectionState.connecting, ConnectionState.disconnected)

    # Wait until all pre-loaded messages have been consumed and listen() is
    # blocking on the queue (messages list emptied by then).
    for _ in range(200):
        await asyncio.sleep(0)
        if not client.messages:
            break

    # Give the _handle_messages loop time to process everything in the queue
    for _ in range(20):
        await asyncio.sleep(0)

    # progress and finished events are forwarded via message_received
    assert any(m.event is ClientEvent.progress for m in received), (
        f"expected progress message, got: {received}"
    )
    assert any(m.event is ClientEvent.finished for m in received), (
        f"expected finished message, got: {received}"
    )

    # A non-empty error should have been set during the disconnect phase …
    assert any(e != "" for e in errors), "expected a non-empty error during disconnect"

    # … and then cleared once the reconnect message arrived
    assert conn.error == "", f"error should be cleared after reconnect, got: {conn.error!r}"

    await conn.disconnect()
    assert conn.state is ConnectionState.disconnected
