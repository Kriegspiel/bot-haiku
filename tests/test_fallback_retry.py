from __future__ import annotations

from contextlib import ExitStack
import unittest
from unittest import mock

import bot


class FallbackRetryTests(unittest.TestCase):
    def state(self):
        return {"state": "active", "turn": "white", "your_color": "white",
                "move_number": 3, "possible_actions": ["move"],
                "allowed_moves": ["a2a3", "e2e4"],
                "scoresheet": {"viewer_color": "white", "turns": []}}

    def test_fallback_avoids_rejected_preferred_move(self):
        state = self.state()
        decision = bot.fallback_decision(
            state, exclude_actions=[{"action": "move", "uci": "e2e4"}])
        self.assertEqual(decision, {"action": "move", "uci": "a2a3"})
        self.assertEqual(state["allowed_moves"], ["a2a3", "e2e4"])

    def test_fallback_stops_after_all_moves_rejected(self):
        self.assertEqual(bot.fallback_ranked_actions(
            self.state(), exclude_actions=[
                {"action": "move", "uci": "e2e4"},
                {"action": "move", "uci": "a2a3"}]), [])

    def test_fallback_can_ask_after_moves_exhausted(self):
        state = self.state()
        state["possible_actions"].append("ask_any")
        self.assertEqual(bot.fallback_ranked_actions(
            state, exclude_actions=[
                {"action": "move", "uci": "e2e4"},
                {"action": "move", "uci": "a2a3"}]),
            [{"action": "ask_any", "uci": None}])

    def test_fallback_does_not_repeat_rejected_question(self):
        state = {"allowed_moves": [], "possible_actions": ["ask_any"]}
        self.assertIsNone(bot.fallback_decision(
            state, exclude_actions=[{"action": "ask_any", "uci": None}]))

    def test_no_key_fallback_receives_retry_exclusions(self):
        with mock.patch.object(bot, "anthropic_enabled", return_value=False):
            actions, source, response = bot.choose_ranked_actions(
                self.state(), game_id="test-game",
                exclude_actions=[{"action": "move", "uci": "e2e4"}])
        self.assertEqual(actions, [{"action": "move", "uci": "a2a3"}])
        self.assertTrue(source.startswith("fallback_no_"))
        self.assertIsNone(response)

    def test_model_failure_fallback_receives_retry_exclusions(self):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(bot, "anthropic_enabled", return_value=True))
            stack.enter_context(mock.patch.object(
                bot, "call_anthropic_messages", side_effect=ValueError("response exhausted its token limit")))
            stack.enter_context(self.assertLogs(bot.logger, level="WARNING"))
            actions, source, _ = bot.choose_ranked_actions(
                self.state(), game_id="test-game",
                exclude_actions=[{"action": "move", "uci": "e2e4"}])
        self.assertEqual(actions, [{"action": "move", "uci": "a2a3"}])
        self.assertEqual(source, "fallback")

    def test_game_runner_progresses_after_rejected_fallback(self):
        state = self.state()
        submissions = []

        def get(path):
            return {"rule_variant": "berkeley_any"} if path == "/game/test-game" else dict(state)

        def post(path, payload):
            self.assertEqual(path, "/game/test-game/move")
            submissions.append(payload["uci"])
            done = payload["uci"] == "a2a3"
            if done:
                state["turn"] = "black"
            return {"move_done": done, "announcement": "complete" if done else "illegal"}

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(bot, "anthropic_enabled", return_value=False))
            stack.enter_context(mock.patch.object(bot, "get_json", side_effect=get))
            stack.enter_context(mock.patch.object(bot, "post_json", side_effect=post))
            stack.enter_context(mock.patch.object(bot, "get_conversation_state", return_value={}))
            stack.enter_context(mock.patch.object(bot, "save_conversation_state"))
            stack.enter_context(mock.patch.object(bot, "max_model_batches_per_turn", return_value=3))
            self.assertTrue(bot.maybe_play_game("test-game"))
        self.assertEqual(submissions, ["e2e4", "a2a3"])
        # A new turn can try the move again; exclusions are scoped to the caller's turn.
        self.assertEqual(bot.fallback_decision(self.state()), {"action": "move", "uci": "e2e4"})
