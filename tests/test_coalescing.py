"""
Tests for once-per-turn snapshot gating ("coalescing") in the status-context hook.

The hook computes and injects its <system-reminder> block on the first
provider:request of each turn (iteration <= 1, or after a prompt:submit
reset) and NEVER re-injects within the same turn -- skipped calls build no
content at all (no _gather_env_info, no git subprocesses). With
coalesce=False, or when event data has no 'iteration' key, the legacy
inject-every-call behavior applies.
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


class TestSnapshotCoalescing:
    """Test suite for once-per-turn snapshot gating."""

    @pytest.mark.asyncio
    async def test_iteration_one_injects(self):
        """Case 1: iteration=1 (turn start, live 1-based) injects."""
        hook = make_hook()
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            result = await hook.on_provider_request(
                "provider:request", {"iteration": 1}
            )
        assert result.action == "inject_context"
        assert "ENV-A" in result.context_injection

    @pytest.mark.asyncio
    async def test_iteration_two_same_turn_continues_even_if_content_changed(self):
        """Case 2: iteration=2 same turn -> continue EVEN IF content changed."""
        hook = make_hook()
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            first = await hook.on_provider_request(
                "provider:request", {"iteration": 1}
            )
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-B")):
            second = await hook.on_provider_request(
                "provider:request", {"iteration": 2}
            )
        assert first.action == "inject_context"
        assert second.action == "continue"

    @pytest.mark.asyncio
    async def test_skipped_call_builds_no_content(self):
        """Case 3: on a skipped call, _gather_env_info is NOT called."""
        hook = make_hook()
        calls = {"count": 0}

        def counting_env_info():
            calls["count"] += 1
            return env_info("ENV-A")

        with patch.object(hook, "_gather_env_info", side_effect=counting_env_info):
            await hook.on_provider_request("provider:request", {"iteration": 1})
            assert calls["count"] == 1
            skipped = await hook.on_provider_request(
                "provider:request", {"iteration": 2}
            )
        assert skipped.action == "continue"
        assert calls["count"] == 1  # not called again on the skipped call

    @pytest.mark.asyncio
    async def test_prompt_submit_reset_then_iteration_two_injects(self):
        """Case 4: prompt:submit reset, then iteration=2 -> injects."""
        hook = make_hook()
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            await hook.on_provider_request("provider:request", {"iteration": 1})
            reset = await hook.on_prompt_submit("prompt:submit", {})
            result = await hook.on_provider_request(
                "provider:request", {"iteration": 2}
            )
        assert reset.action == "continue"
        assert result.action == "inject_context"

    @pytest.mark.asyncio
    async def test_new_turn_iteration_one_again_injects(self):
        """Case 5: new turn via iteration=1 again (no prompt:submit) -> injects."""
        hook = make_hook()
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            await hook.on_provider_request("provider:request", {"iteration": 1})
            await hook.on_provider_request("provider:request", {"iteration": 2})
            result = await hook.on_provider_request(
                "provider:request", {"iteration": 1}
            )
        assert result.action == "inject_context"

    @pytest.mark.asyncio
    async def test_iteration_zero_treated_as_turn_start(self):
        """Case 6: iteration=0 (0-based orchestrators) treated as turn start -> injects."""
        hook = make_hook()
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            result = await hook.on_provider_request(
                "provider:request", {"iteration": 0}
            )
        assert result.action == "inject_context"
        assert "ENV-A" in result.context_injection

    @pytest.mark.asyncio
    async def test_missing_iteration_key_always_injects(self):
        """Case 7: data without 'iteration' key -> injects on consecutive calls."""
        hook = make_hook()
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            first = await hook.on_provider_request("provider:request", {})
            second = await hook.on_provider_request("provider:request", {})
        assert first.action == "inject_context"
        assert second.action == "inject_context"

    @pytest.mark.asyncio
    async def test_coalesce_false_always_injects(self):
        """Case 8: coalesce=False -> injects on consecutive calls (iterations 1, 2)."""
        hook = make_hook(coalesce=False)
        with patch.object(hook, "_gather_env_info", return_value=env_info("ENV-A")):
            first = await hook.on_provider_request(
                "provider:request", {"iteration": 1}
            )
            second = await hook.on_provider_request(
                "provider:request", {"iteration": 2}
            )
        assert first.action == "inject_context"
        assert second.action == "inject_context"
