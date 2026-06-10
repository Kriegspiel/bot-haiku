from __future__ import annotations

import json
import unittest
from unittest import mock

import bot


class BotTests(unittest.TestCase):
    def setUp(self) -> None:
        bot._ANTHROPIC_PREFLIGHT_CACHE.update({"ready": None, "expires_at": 0.0, "reason": "unchecked"})
        bot._MODEL_AVAILABILITY_REPORT_CACHE.update({"ready": None, "reason": "", "reported_at": 0.0})

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
            "material_summary": {
                "white": {"pieces_remaining": 16, "pawns_captured": None},
                "black": {"pieces_remaining": 15, "pawns_captured": None},
            },
            "scoresheet": {
                "viewer_color": "white",
                "turns": [
                    {"turn": 1, "white": [{"move_uci": "e2e4", "message": "Move complete"}], "black": []},
                ],
            },
        }
        system_prompt = bot.build_system_prompt("berkeley_any")
        user_prompt = bot.build_initial_user_prompt(state)
        followup_prompt = bot.build_followup_user_prompt(
            state,
            feedback=["Rejected move e2e4: Illegal move"],
            exclude_actions=[{"action": "move", "uci": "e2e4"}],
            recent_updates=["Turn 3 black: Illegal move"],
        )
        self.assertIn("Berkeley + Any", system_prompt)
        self.assertNotIn("I. Introduction", system_prompt)
        self.assertIn("\"private_board_fen\":\"fen\"", user_prompt)
        self.assertNotIn("rule_variant", user_prompt)
        self.assertNotIn("pawns_captured", user_prompt)
        self.assertIn("\"material\":{\"black\":{\"pieces_remaining\":15},\"white\":{\"pieces_remaining\":16}}", user_prompt)
        self.assertIn("\"recent_turns\":[{\"black\":[],\"turn\":1,\"white\":[\"[e2e4] Move complete\"]}]", user_prompt)
        self.assertIn("Rejected move e2e4: Illegal move", followup_prompt)
        self.assertNotIn("Turn 3 black: Illegal move", followup_prompt)
        self.assertIn("\"rejected_this_turn\":[\"e2e4\"]", followup_prompt)

    def test_ruleset_summary_files_cover_supported_variants(self) -> None:
        summary_files = {path.stem for path in bot.RULESET_SUMMARY_DIR.glob("*.md")}
        self.assertEqual(summary_files, set(bot.SUPPORTED_RULE_VARIANTS))
        required_concepts = ["illegal", "capture", "check", "pawn", "promotion", "stalemate"]
        checklist_labels = [
            "Referee response to illegal tries:",
            "Capture announcements:",
            "Check announcements:",
            "Pawn-capture / Any? handling:",
            "Promotion announcements:",
            "Stalemate:",
        ]
        for variant in bot.SUPPORTED_RULE_VARIANTS:
            summary = bot.load_ruleset_summary(variant)
            normalized = summary.lower()
            self.assertGreaterEqual(len(summary.split()), 90)
            self.assertLessEqual(len(summary.split()), 190)
            self.assertFalse(any(line.startswith("- ") for line in summary.splitlines()))
            for label in checklist_labels:
                self.assertNotIn(label, summary)
            for concept in required_concepts:
                self.assertIn(concept, normalized)

        rand_summary = bot.load_ruleset_summary("rand")
        self.assertIn("promotions are announced in rand, but not the promoted piece type", rand_summary.lower())
        self.assertIn("the stalemated player loses in rand", rand_summary.lower())

    def test_build_system_prompt_uses_ruleset_summary_file(self) -> None:
        summary = bot.load_ruleset_summary("wild16")
        system_prompt = bot.build_system_prompt("wild16")
        self.assertIn("illegal tries private", summary)
        self.assertIn(summary, system_prompt)
        self.assertNotIn("Cincinnati", system_prompt)

    def test_turn_snapshot_requests_exact_batch_when_enough_actions_exist(self) -> None:
        allowed_moves = ["a2a3", "b2b3", "c2c3", "d2d3", "e2e3", "f2f3", "g2g3", "h2h3", "b1c3"]
        state = {
            "rule_variant": "berkeley_any",
            "your_color": "white",
            "turn": "white",
            "move_number": 1,
            "your_fen": "fen",
            "possible_actions": ["move", "ask_any"],
            "allowed_moves": allowed_moves,
            "scoresheet": {"viewer_color": "white", "turns": []},
        }
        payload = bot.build_turn_snapshot_payload(state)
        system_prompt = bot.build_system_prompt("berkeley_any")

        self.assertEqual(payload["target_count"], 10)
        self.assertIn("Return exactly target_count", system_prompt)
        self.assertNotIn("Return up to target_count", system_prompt)

    def test_anthropic_max_prompt_turns_has_ten_turn_floor(self) -> None:
        with mock.patch.dict("os.environ", {"ANTHROPIC_MAX_PROMPT_TURNS": "5"}):
            self.assertEqual(bot.anthropic_max_prompt_turns(), 10)
        with mock.patch.dict("os.environ", {"ANTHROPIC_MAX_PROMPT_TURNS": "12"}):
            self.assertEqual(bot.anthropic_max_prompt_turns(), 12)
        with mock.patch.dict("os.environ", {"ANTHROPIC_MAX_PROMPT_TURNS": "invalid"}):
            self.assertEqual(bot.anthropic_max_prompt_turns(), 10)

    def test_turn_snapshot_includes_at_least_ten_recent_turns_when_available(self) -> None:
        turns = [
            {
                "turn": turn_number,
                "white": [{"move_uci": f"a{turn_number}a{turn_number + 1}", "message": "Move complete"}],
                "black": [],
            }
            for turn_number in range(1, 13)
        ]
        state = {
            "rule_variant": "berkeley_any",
            "your_color": "white",
            "turn": "white",
            "move_number": 12,
            "your_fen": "fen",
            "possible_actions": ["move"],
            "allowed_moves": ["e2e4"],
            "scoresheet": {"viewer_color": "white", "turns": turns},
        }

        with mock.patch.dict("os.environ", {"ANTHROPIC_MAX_PROMPT_TURNS": "5"}):
            payload = bot.build_turn_snapshot_payload(state)

        self.assertEqual(len(payload["recent_turns"]), 10)
        self.assertEqual(payload["recent_turns"][0]["turn"], 3)
        self.assertEqual(payload["recent_turns"][-1]["turn"], 12)

    def test_turn_snapshot_payload_includes_rule_specific_public_context(self) -> None:
        cincinnati_payload = bot.build_turn_snapshot_payload(
            {
                "rule_variant": "cincinnati",
                "your_color": "black",
                "turn": "black",
                "move_number": 8,
                "your_fen": "fen",
                "possible_actions": ["move"],
                "allowed_moves": ["g8f6"],
                "material_summary": {
                    "white": {"pieces_remaining": 16, "pawns_captured": 0},
                    "black": {"pieces_remaining": 15, "pawns_captured": 1},
                },
                "reserve_summary": {
                    "white": {"pawns": 0, "knights": 0, "bishops": 0, "rooks": 0, "queens": 0},
                    "black": {"pawns": 0, "knights": 0, "bishops": 0, "rooks": 0, "queens": 0},
                },
                "scoresheet": {"viewer_color": "black", "turns": []},
            }
        )
        self.assertEqual(
            cincinnati_payload["material"],
            {
                "white": {"pieces_remaining": 16, "pawns_captured": 0},
                "black": {"pieces_remaining": 15, "pawns_captured": 1},
            },
        )
        self.assertNotIn("reserves", cincinnati_payload)

        crazy_payload = bot.build_turn_snapshot_payload(
            {
                "rule_variant": "crazykrieg",
                "your_color": "white",
                "turn": "white",
                "move_number": 9,
                "your_fen": "fen",
                "possible_actions": ["move", "ask_any"],
                "allowed_moves": ["g1f3"],
                "material_summary": {
                    "white": {"pieces_remaining": 17, "pawns_captured": None},
                    "black": {"pieces_remaining": 15, "pawns_captured": None},
                },
                "reserve_summary": {
                    "white": {"pawns": 0, "knights": 1, "bishops": 0, "rooks": 0, "queens": 0},
                    "black": {"pawns": 1, "knights": 0, "bishops": 0, "rooks": 0, "queens": 0},
                },
                "scoresheet": {"viewer_color": "white", "turns": []},
            }
        )
        self.assertEqual(
            crazy_payload["material"],
            {"white": {"pieces_remaining": 17}, "black": {"pieces_remaining": 15}},
        )
        self.assertEqual(crazy_payload["reserves"]["white"]["knights"], 1)
        self.assertEqual(crazy_payload["reserves"]["black"]["pawns"], 1)

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
        self.assertFalse(bot.should_create_lobby_game([]))
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_AUTO_CREATE_LOBBY_GAME": "true"}):
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
                    {"game_code": "CIN123", "created_by": "randobot", "rule_variant": "cincinnati"},
                ],
                profile_lookup=lambda username: {"role": "bot"},
            )

        self.assertEqual([game["game_code"] for game in candidates], ["BER123", "ANY123", "CIN123"])

    def test_supported_rule_variants_default_to_all_playable_rulesets(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(bot.supported_rule_variants(), list(bot.SUPPORTED_RULE_VARIANTS))

    def test_supported_rule_variants_expands_legacy_two_ruleset_default(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_SUPPORTED_RULE_VARIANTS": "berkeley,berkeley_any"}):
            self.assertEqual(bot.supported_rule_variants(), list(bot.SUPPORTED_RULE_VARIANTS))

    def test_supported_rule_variants_respects_non_legacy_custom_subset(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_SUPPORTED_RULE_VARIANTS": "wild16,crazykrieg"}):
            self.assertEqual(bot.supported_rule_variants(), ["wild16", "crazykrieg"])

    def test_sync_bot_profile_posts_supported_rule_variants(self) -> None:
        with mock.patch.object(bot, "post_json", return_value={"ok": True}) as post_json:
            self.assertTrue(bot.sync_bot_profile())

        post_json.assert_called_once_with(
            "/bots/profile",
            {"supported_rule_variants": list(bot.SUPPORTED_RULE_VARIANTS)},
        )

    def test_action_schema_does_not_request_unused_reasons(self) -> None:
        candidate_schema = bot.action_schema()["schema"]["properties"]["candidates"]["items"]
        self.assertNotIn("reason", candidate_schema["properties"])
        self.assertEqual(candidate_schema["required"], ["action", "uci"])

    def test_choose_ranked_actions_is_stateless(self) -> None:
        state = {
            "rule_variant": "berkeley_any",
            "your_color": "white",
            "turn": "white",
            "move_number": 3,
            "your_fen": "fen",
            "possible_actions": ["move"],
            "allowed_moves": ["e2e4"],
            "material_summary": {
                "white": {"pieces_remaining": 16, "pawns_captured": None},
                "black": {"pieces_remaining": 16, "pawns_captured": None},
            },
            "scoresheet": {"viewer_color": "white", "turns": []},
        }
        raw_response = {
            "content": [{"type": "text", "text": json.dumps({"candidates": [{"action": "move", "uci": "e2e4"}]})}],
            "usage": {"cache_read_input_tokens": 50, "cache_creation_input_tokens": 10},
        }
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with mock.patch.object(bot, "call_anthropic_messages", return_value=raw_response) as call_anthropic_messages:
                decisions, source, response_id = bot.choose_ranked_actions(
                    state,
                    game_id="gid1",
                    conversation={"messages": [{"role": "user", "content": "old bulky prompt"}]},
                )

        self.assertEqual(decisions, [{"action": "move", "uci": "e2e4"}])
        self.assertEqual(source, "model")
        self.assertIsNone(response_id)
        messages = call_anthropic_messages.call_args.kwargs["messages"]
        self.assertEqual(len(messages), 1)
        self.assertNotIn("old bulky prompt", json.dumps(messages))
        self.assertIn("\"private_board_fen\":\"fen\"", messages[0]["content"][0]["text"])

    def test_choose_bot_game_to_join_returns_candidate(self) -> None:
        games = [{"game_code": "BOT123", "created_by": "randobot", "rule_variant": "berkeley_any"}]

        with mock.patch.dict("os.environ", {"KRIEGSPIEL_BOT_USERNAME": "gptnano"}):
            with mock.patch.object(bot.random, "choice", side_effect=lambda items: items[0]):
                with mock.patch.object(bot, "get_public_user", return_value={"role": "bot"}):
                    self.assertEqual(bot.choose_bot_game_to_join(games, rng=bot.random)["game_code"], "BOT123")

    def test_maybe_join_bot_lobby_game_samples_once_per_cooldown_window(self) -> None:
        games = []
        open_games = [{"game_code": "BOT123", "created_by": "randobot", "rule_variant": "berkeley_any"}]

        with mock.patch.object(bot, "get_json", return_value={"games": open_games}):
            with mock.patch.object(bot, "get_public_user", return_value={"role": "bot"}):
                with mock.patch.object(bot, "load_state", return_value={"last_bot_game_join_attempt_at": 0}):
                    saved_states = []
                    with mock.patch.object(bot, "save_state", side_effect=saved_states.append):
                        with mock.patch.object(bot.random, "choice", side_effect=lambda items: items[0]):
                            with mock.patch.object(bot.random, "random", return_value=0.5):
                                with mock.patch.object(bot, "post_json") as post_json:
                                    self.assertFalse(bot.maybe_join_bot_lobby_game(games, rng=bot.random))
                                    post_json.assert_not_called()
                        self.assertEqual(len(saved_states), 1)

    def test_maybe_join_bot_lobby_game_records_sample_even_without_candidate(self) -> None:
        games = []

        with mock.patch.object(bot, "get_json", return_value={"games": []}):
            with mock.patch.object(bot, "load_state", return_value={"last_bot_game_join_attempt_at": 0}):
                saved_states = []
                with mock.patch.object(bot, "save_state", side_effect=saved_states.append):
                    with mock.patch.object(bot, "post_json") as post_json:
                        self.assertFalse(bot.maybe_join_bot_lobby_game(games, rng=bot.random))
                        post_json.assert_not_called()

        self.assertEqual(len(saved_states), 1)

    def test_maybe_join_bot_lobby_game_skips_open_sample_during_cooldown(self) -> None:
        games = []

        with mock.patch.object(bot, "load_state", return_value={"last_bot_game_join_attempt_at": 100}):
            with mock.patch.object(bot.time, "time", return_value=130):
                with mock.patch.object(bot, "get_json") as get_json:
                    self.assertFalse(bot.maybe_join_bot_lobby_game(games, rng=bot.random))
                    get_json.assert_not_called()

    def test_maybe_join_bot_lobby_game_joins_when_probability_hits(self) -> None:
        games = []
        open_games = [{"game_code": "BOT123", "created_by": "randobot", "rule_variant": "berkeley_any"}]

        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with mock.patch.object(bot, "get_json", return_value={"games": open_games}):
                with mock.patch.object(bot, "get_public_user", return_value={"role": "bot"}):
                    with mock.patch.object(bot, "load_state", return_value={"last_bot_game_join_attempt_at": 0}):
                        with mock.patch.object(bot, "save_state"):
                            with mock.patch.object(bot.random, "choice", side_effect=lambda items: items[0]):
                                with mock.patch.object(bot.random, "random", return_value=0.0005):
                                    with mock.patch.object(bot, "anthropic_preflight_status", return_value=(True, "ok")):
                                        with mock.patch.object(bot, "post_json", return_value={"game_id": "g1", "game_code": "BOT123"}) as post_json:
                                            self.assertTrue(bot.maybe_join_bot_lobby_game(games, rng=bot.random))
                                            post_json.assert_called_once_with("/game/join/BOT123")

    def test_should_not_join_bot_lobby_game_when_active_cap_reached(self) -> None:
        games = [{"state": "active"} for _ in range(5)]
        self.assertFalse(bot.should_join_bot_lobby_game(games))

    def test_has_own_waiting_game_detects_existing_lobby(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_BOT_USERNAME": "gptnano"}):
            self.assertTrue(bot.has_own_waiting_game([{"game_code": "ABC123", "created_by": "gptnano"}]))
            self.assertFalse(bot.has_own_waiting_game([{"game_code": "XYZ789", "created_by": "randobot"}]))

    def test_anthropic_preflight_status_caches_success(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with mock.patch.object(bot.requests, "post", return_value=response) as post:
                self.assertEqual(bot.anthropic_preflight_status(), (True, "ok"))
                self.assertEqual(bot.anthropic_preflight_status(), (True, "ok"))
        self.assertEqual(post.call_count, 1)

    def test_call_anthropic_messages_caches_system_prompt(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"content": []}
        messages = [{"role": "user", "content": [{"type": "text", "text": "Current turn JSON:\n{}"}]}]

        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_CACHE_TTL": "1h"}, clear=False):
            with mock.patch.object(bot.requests, "post", return_value=response) as post:
                bot.call_anthropic_messages(system_prompt="short system", messages=messages)

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(payload["system"][0]["text"], "short system")
        self.assertEqual(payload["system"][0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(payload["messages"], messages)

    def test_report_model_availability_posts_status_and_throttles_repeats(self) -> None:
        with mock.patch.object(bot, "post_json", return_value={"ok": True}) as post_json:
            self.assertTrue(bot.report_model_availability(False, "http_400: usage_limit"))
            self.assertFalse(bot.report_model_availability(False, "http_400: usage_limit"))
            self.assertTrue(bot.report_model_availability(True, "ok"))

        self.assertEqual(
            post_json.call_args_list,
            [
                mock.call("/bots/availability", {"provider": "anthropic", "ready": False, "reason": "http_400: usage_limit"}),
                mock.call("/bots/availability", {"provider": "anthropic", "ready": True, "reason": "ok"}),
            ],
        )

    def test_maybe_join_bot_lobby_game_skips_join_when_anthropic_unavailable(self) -> None:
        games = []
        open_games = [{"game_code": "BOT123", "created_by": "randobot", "rule_variant": "berkeley_any"}]

        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with mock.patch.object(bot, "get_json", return_value={"games": open_games}):
                with mock.patch.object(bot, "get_public_user", return_value={"role": "bot"}):
                    with mock.patch.object(bot, "load_state", return_value={"last_bot_game_join_attempt_at": 0}):
                        with mock.patch.object(bot, "save_state"):
                            with mock.patch.object(bot.random, "choice", side_effect=lambda items: items[0]):
                                with mock.patch.object(bot.random, "random", return_value=0.0005):
                                    with mock.patch.object(bot, "anthropic_preflight_status", return_value=(False, "http_429: credit_balance_too_low")):
                                        with mock.patch.object(bot, "post_json") as post_json:
                                            self.assertFalse(bot.maybe_join_bot_lobby_game(games, rng=bot.random))
                                            post_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
