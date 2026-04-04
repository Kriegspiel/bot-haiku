from __future__ import annotations

import unittest
from unittest import mock

import bot


class BotTests(unittest.TestCase):
    def test_normalize_ranked_decisions_filters_invalid_and_duplicates(self) -> None:
        state = {"possible_actions": ["move", "ask_any"], "allowed_moves": ["e2e4", "d2d4"]}
        decisions = bot.normalize_ranked_decisions(
            {
                "candidates": [
                    {"action": "move", "uci": "E2E4", "reason": "center"},
                    {"action": "move", "uci": "e2e4", "reason": "duplicate"},
                    {"action": "move", "uci": "a2a4", "reason": "illegal"},
                    {"action": "ask_any", "uci": None, "reason": "question"},
                ]
            },
            state,
        )
        self.assertEqual(decisions, [{"action": "move", "uci": "e2e4"}, {"action": "ask_any", "uci": None}])

    def test_normalize_decision_accepts_legal_move(self) -> None:
        state = {"possible_actions": ["move", "ask_any"], "allowed_moves": ["e2e4", "d2d4"]}
        decision = bot.normalize_decision({"action": "move", "uci": "E2E4"}, state)
        self.assertEqual(decision, {"action": "move", "uci": "e2e4"})

    def test_normalize_decision_rejects_illegal_move(self) -> None:
        state = {"possible_actions": ["move"], "allowed_moves": ["e2e4"]}
        decision = bot.normalize_decision({"action": "move", "uci": "a2a4"}, state)
        self.assertIsNone(decision)

    def test_normalize_decision_accepts_ask_any_when_available(self) -> None:
        state = {"possible_actions": ["ask_any"], "allowed_moves": []}
        decision = bot.normalize_decision({"action": "ask_any", "uci": None}, state)
        self.assertEqual(decision, {"action": "ask_any", "uci": None})

    def test_extract_response_text_reads_nested_output(self) -> None:
        payload = {"content": [{"type": "text", "text": "{\"candidates\":[{\"action\":\"move\",\"uci\":\"e2e4\",\"reason\":\"center\"}]}"}]}
        self.assertEqual(
            bot.extract_response_text(payload),
            "{\"candidates\":[{\"action\":\"move\",\"uci\":\"e2e4\",\"reason\":\"center\"}]}",
        )

    def test_fallback_prefers_center_moves(self) -> None:
        state = {"possible_actions": ["move"], "allowed_moves": ["a2a3", "e2e4", "h2h3"]}
        decision = bot.fallback_decision(state)
        self.assertEqual(decision, {"action": "move", "uci": "e2e4"})

    def test_fallback_ranked_actions_wraps_single_decision(self) -> None:
        state = {"possible_actions": ["move"], "allowed_moves": ["a2a3", "e2e4"]}
        decisions = bot.fallback_ranked_actions(state)
        self.assertEqual(decisions, [{"action": "move", "uci": "e2e4"}])

    def test_build_prompts_split_rules_and_state(self) -> None:
        state = {
            "rule_variant": "berkeley_any",
            "your_color": "white",
            "state": "active",
            "turn": "white",
            "move_number": 3,
            "your_fen": "fen",
            "possible_actions": ["move"],
            "allowed_moves": ["e2e4"],
            "scoresheet": {"viewer_color": "white", "turns": []},
        }
        system_prompt = bot.build_system_prompt("berkeley_any")
        user_prompt = bot.build_initial_user_prompt(state)
        followup_prompt = bot.build_followup_user_prompt(
            state,
            feedback=["Rejected move e2e4: Illegal move"],
            exclude_actions=[{"action": "move", "uci": "e2e4"}],
            recent_updates=["Turn 3 black: Illegal move"],
        )
        self.assertIn("Rules and setting", system_prompt)
        self.assertIn("\"private_board_fen\":\"fen\"", user_prompt)
        self.assertIn("\"phase\":\"initial_turn_snapshot\"", user_prompt)
        self.assertIn("Rejected move e2e4: Illegal move", followup_prompt)
        self.assertIn("Turn 3 black: Illegal move", followup_prompt)
        self.assertIn("\"already_tried_this_turn\":[\"e2e4\"]", followup_prompt)
        self.assertIn("ordered from best to worse priority", followup_prompt)

    def test_new_recent_items_returns_only_suffix_delta(self) -> None:
        previous = ["Turn 1 white: Move complete", "Turn 1 black: Illegal move"]
        current = [
            "Turn 1 white: Move complete",
            "Turn 1 black: Illegal move",
            "Turn 2 white: Has pawn captures",
        ]
        self.assertEqual(bot.new_recent_items(previous, current), ["Turn 2 white: Has pawn captures"])

    def test_scoresheet_digest_changes_with_content(self) -> None:
        digest_a = bot.scoresheet_digest({"viewer_color": "white", "turns": []})
        digest_b = bot.scoresheet_digest({"viewer_color": "white", "turns": [{"turn": 1, "white": [], "black": []}]})
        self.assertNotEqual(digest_a, digest_b)

    def test_summarize_scoresheet_turns_returns_lines(self) -> None:
        lines = bot.summarize_scoresheet_turns(
            {
                "viewer_color": "white",
                "turns": [{"turn": 1, "white": [{"message": "Move attempt — Move complete"}], "black": []}],
            },
            max_turns=12,
        )
        self.assertTrue(isinstance(lines, list))
        self.assertTrue(any("Move attempt" in line for line in lines))

    def test_should_create_lobby_game_when_under_cap_and_no_waiting_game(self) -> None:
        self.assertTrue(bot.should_create_lobby_game([]))
        self.assertFalse(bot.should_create_lobby_game([{"state": "active"}]))

    def test_should_not_create_lobby_game_when_waiting_game_exists(self) -> None:
        games = [{"state": "active"}, {"state": "waiting"}]
        self.assertFalse(bot.should_create_lobby_game(games))

    def test_should_not_create_lobby_game_when_active_cap_is_reached(self) -> None:
        games = [{"state": "active"} for _ in range(5)]
        self.assertFalse(bot.should_create_lobby_game(games))

    def test_open_bot_lobby_candidates_only_include_other_bot_waiting_games(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_BOT_USERNAME": "gptnano"}):
            candidates = bot.open_bot_lobby_candidates(
                [
                    {
                        "game_code": "BOT123",
                        "created_by": "randobot",
                        "rule_variant": "berkeley_any",
                    },
                    {
                        "game_code": "SELF12",
                        "created_by": "gptnano",
                        "rule_variant": "berkeley_any",
                    },
                    {
                        "game_code": "HUM123",
                        "created_by": "fil",
                        "rule_variant": "berkeley_any",
                    },
                ],
                profile_lookup=lambda username: {"role": "bot" if username == "randobot" else "user"},
            )

        self.assertEqual([game["game_code"] for game in candidates], ["BOT123"])

    def test_open_bot_lobby_candidates_respect_supported_rule_variants(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_BOT_USERNAME": "gptnano", "KRIEGSPIEL_SUPPORTED_RULE_VARIANTS": "berkeley,berkeley_any"}):
            candidates = bot.open_bot_lobby_candidates(
                [
                    {"game_code": "BER123", "created_by": "randobot", "rule_variant": "berkeley"},
                    {"game_code": "ANY123", "created_by": "randobot", "rule_variant": "berkeley_any"},
                ],
                profile_lookup=lambda username: {"role": "bot"},
            )

        self.assertEqual([game["game_code"] for game in candidates], ["BER123", "ANY123"])

    def test_choose_bot_game_to_join_respects_probability(self) -> None:
        games = [{"game_code": "BOT123", "created_by": "randobot", "rule_variant": "berkeley_any"}]

        with mock.patch.dict("os.environ", {"KRIEGSPIEL_BOT_USERNAME": "gptnano"}):
            with mock.patch.object(bot.random, "random", return_value=0.9):
                self.assertIsNone(bot.choose_bot_game_to_join(games, rng=bot.random))
            with mock.patch.object(bot.random, "random", return_value=0.05):
                with mock.patch.object(bot.random, "choice", side_effect=lambda items: items[0]):
                    with mock.patch.object(bot, "get_public_user", return_value={"role": "bot"}):
                        self.assertEqual(bot.choose_bot_game_to_join(games, rng=bot.random)["game_code"], "BOT123")

    def test_should_not_join_bot_lobby_game_when_active_cap_reached(self) -> None:
        games = [{"state": "active"} for _ in range(5)]
        self.assertFalse(bot.should_join_bot_lobby_game(games))

    def test_has_own_waiting_game_detects_existing_lobby(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_BOT_USERNAME": "gptnano"}):
            self.assertTrue(bot.has_own_waiting_game([{"game_code": "ABC123", "created_by": "gptnano"}]))
            self.assertFalse(bot.has_own_waiting_game([{"game_code": "XYZ789", "created_by": "randobot"}]))


if __name__ == "__main__":
    unittest.main()
