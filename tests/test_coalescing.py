"""
Tests for per-turn delta-emission ("coalescing") in the status-context hook.

The hook should inject its <system-reminder> block on the first
provider:request of each turn (iteration=0 or after a prompt:submit reset)
and thereafter only when the block content changed. With coalesce=False,
or when event data has no 'iteration' key, the legacy inject-every-call
behavior applies.
"""

import pytest
from unittest.mock import MagicMock, patch

from amplifier_module_hooks_status_context import StatusContextHook


def make_hook(**config_overrides) -> StatusContextHook:
    """Create a hook with a MagicMock coordinator and deterministic config."""
    coordinator = MagicMock()
    coordinator.get_capability.return_value = None
    coordinator.session_id = "test-session-id"
    coordinator.parent_id = None
    config = {
        "working_dir": ".",
        "include_git": False,
        "include_datetime": False,
        "include_session": False,
    }
    config.update(config_overrides)
    return StatusContextHook(coordinator, config)


def env_info(content: str) -> dict:
    """Deterministic _gather_env_info return value."""
    return {"formatted": content, "is_git_repo": False}


class TestCoalescing:
    """Test suite for per-turn delta-emission."""

    @pytest.mark.asyncio
    async def test_iteration_zero_injects(self):
        """Case 1: iteration=0 injects."""
        hook = make_hook()
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            result = await hook.on_provider_request(
                "provider:request", {"iteration": 0}
            )
        assert result.action == "inject_context"
        assert "ENV-A" in result.context_injection

    @pytest.mark.asyncio
    async def test_iteration_one_unchanged_continues(self):
        """Case 2: iteration=1 with unchanged content -> action continue."""
        hook = make_hook()
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            first = await hook.on_provider_request(
                "provider:request", {"iteration": 0}
            )
            second = await hook.on_provider_request(
                "provider:request", {"iteration": 1}
            )
        assert first.action == "inject_context"
        assert second.action == "continue"

    @pytest.mark.asyncio
    async def test_content_change_mid_turn_injects(self):
        """Case 3: content change at iteration>0 -> injects."""
        hook = make_hook()
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            await hook.on_provider_request("provider:request", {"iteration": 0})
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-B")):
            result = await hook.on_provider_request(
                "provider:request", {"iteration": 1}
            )
        assert result.action == "inject_context"
        assert "ENV-B" in result.context_injection

    @pytest.mark.asyncio
    async def test_new_turn_iteration_zero_reinjects_unchanged(self):
        """Case 4: iteration=0 again with unchanged content -> injects (new turn)."""
        hook = make_hook()
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            await hook.on_provider_request("provider:request", {"iteration": 0})
            await hook.on_provider_request("provider:request", {"iteration": 1})
            result = await hook.on_provider_request(
                "provider:request", {"iteration": 0}
            )
        assert result.action == "inject_context"

    @pytest.mark.asyncio
    async def test_prompt_submit_reset_reinjects(self):
        """Case 5: on_prompt_submit reset, then iteration>0 same content -> injects."""
        hook = make_hook()
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            await hook.on_provider_request("provider:request", {"iteration": 0})
            reset = await hook.on_prompt_submit("prompt:submit", {})
            result = await hook.on_provider_request(
                "provider:request", {"iteration": 1}
            )
        assert reset.action == "continue"
        assert hook._last_injected_hash is not None
        assert result.action == "inject_context"

    @pytest.mark.asyncio
    async def test_missing_iteration_key_always_injects(self):
        """Case 6: data without 'iteration' key -> injects on consecutive calls."""
        hook = make_hook()
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            first = await hook.on_provider_request("provider:request", {})
            second = await hook.on_provider_request("provider:request", {})
        assert first.action == "inject_context"
        assert second.action == "inject_context"

    @pytest.mark.asyncio
    async def test_coalesce_false_always_injects(self):
        """Case 7: coalesce=False -> injects on consecutive calls (iteration 0, 1)."""
        hook = make_hook(coalesce=False)
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            first = await hook.on_provider_request(
                "provider:request", {"iteration": 0}
            )
            second = await hook.on_provider_request(
                "provider:request", {"iteration": 1}
            )
        assert first.action == "inject_context"
        assert second.action == "inject_context"
