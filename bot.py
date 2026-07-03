"""Kriegspiel bot driven by an Anthropic Haiku model.

The bot keeps the game API as the source of truth:
- the server provides the bot's private board state and legal actions
- the model recommends one action in strict JSON
- the bot validates that action locally before sending it back to the API

If the model output is malformed, stale, or unavailable, the bot falls back to a
deterministic legal action so the service remains playable.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / ".bot-state.json"
ENV_PATH = BASE_DIR / ".env"
RULESET_SUMMARY_DIR = BASE_DIR / "ruleset_summaries"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_ANTHROPIC_OUTPUT_TOKENS = 512
ACTION_SCHEMA_NAME = "kriegspiel_next_action"
DEFAULT_MAX_ACTIVE_GAMES_BEFORE_CREATE = 1
BOT_JOIN_COOLDOWN_SECONDS = 60
BOT_GAME_PICK_PROBABILITY = 0.001
DEFAULT_MODEL_BATCH_SIZE = 10
DEFAULT_MAX_MODEL_BATCHES_PER_TURN = 5
DEFAULT_ANTHROPIC_MAX_PROMPT_TURNS = 10
DEFAULT_ANTHROPIC_PREFLIGHT_SUCCESS_TTL_SECONDS = 60.0
DEFAULT_ANTHROPIC_PREFLIGHT_FAILURE_TTL_SECONDS = 15.0
DEFAULT_MODEL_AVAILABILITY_REPORT_INTERVAL_SECONDS = 30.0
USD_PER_MILLION_TOKENS = 1_000_000
ANTHROPIC_HAIKU_INPUT_USD_PER_MILLION_TOKENS = 1.00
ANTHROPIC_HAIKU_OUTPUT_USD_PER_MILLION_TOKENS = 5.00
ANTHROPIC_HAIKU_CACHE_READ_INPUT_USD_PER_MILLION_TOKENS = 0.10
ANTHROPIC_HAIKU_CACHE_WRITE_5M_USD_PER_MILLION_TOKENS = 1.25
ANTHROPIC_HAIKU_CACHE_WRITE_1H_USD_PER_MILLION_TOKENS = 2.00
SUPPORTED_RULE_VARIANTS = ("berkeley", "berkeley_any", "cincinnati", "wild16", "rand", "english", "crazykrieg")
DEFAULT_SUPPORTED_RULE_VARIANTS = ",".join(SUPPORTED_RULE_VARIANTS)
LEGACY_DEFAULT_SUPPORTED_RULE_VARIANTS = ("berkeley", "berkeley_any")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
_ANTHROPIC_PREFLIGHT_CACHE = {"ready": None, "expires_at": 0.0, "reason": "unchecked"}
_MODEL_AVAILABILITY_REPORT_CACHE = {"ready": None, "reason": "", "reported_at": 0.0}


def load_env_file(path: str | Path = ENV_PATH) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def base_url() -> str:
    return os.environ.get("KRIEGSPIEL_API_BASE", "http://localhost:8000").rstrip("/")


def anthropic_base_url() -> str:
    return os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1").rstrip("/")


def anthropic_timeout_seconds() -> float:
    raw = os.environ.get("ANTHROPIC_TIMEOUT_SECONDS", "45").strip()
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 45.0


def anthropic_preflight_success_ttl_seconds() -> float:
    raw = os.environ.get("ANTHROPIC_PREFLIGHT_SUCCESS_TTL_SECONDS", str(DEFAULT_ANTHROPIC_PREFLIGHT_SUCCESS_TTL_SECONDS)).strip()
    try:
        return max(5.0, float(raw))
    except ValueError:
        return DEFAULT_ANTHROPIC_PREFLIGHT_SUCCESS_TTL_SECONDS


def anthropic_preflight_failure_ttl_seconds() -> float:
    raw = os.environ.get("ANTHROPIC_PREFLIGHT_FAILURE_TTL_SECONDS", str(DEFAULT_ANTHROPIC_PREFLIGHT_FAILURE_TTL_SECONDS)).strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_ANTHROPIC_PREFLIGHT_FAILURE_TTL_SECONDS


def model_availability_report_interval_seconds() -> float:
    raw = os.environ.get(
        "MODEL_AVAILABILITY_REPORT_INTERVAL_SECONDS",
        str(DEFAULT_MODEL_AVAILABILITY_REPORT_INTERVAL_SECONDS),
    ).strip()
    try:
        return max(5.0, float(raw))
    except ValueError:
        return DEFAULT_MODEL_AVAILABILITY_REPORT_INTERVAL_SECONDS


def anthropic_cache_ttl() -> str:
    raw = os.environ.get("ANTHROPIC_CACHE_TTL", "1h").strip().lower()
    if raw in {"5m", "1h"}:
        return raw
    return "1h"


def anthropic_max_output_tokens() -> int:
    raw = os.environ.get("ANTHROPIC_MAX_OUTPUT_TOKENS", str(DEFAULT_ANTHROPIC_OUTPUT_TOKENS)).strip()
    try:
        return max(128, int(raw))
    except ValueError:
        return DEFAULT_ANTHROPIC_OUTPUT_TOKENS


def usage_token_count(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key)
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def anthropic_usage_cost_usd(usage: dict[str, Any], *, cache_ttl: str) -> float:
    cache_write_rate = (
        ANTHROPIC_HAIKU_CACHE_WRITE_5M_USD_PER_MILLION_TOKENS
        if cache_ttl == "5m"
        else ANTHROPIC_HAIKU_CACHE_WRITE_1H_USD_PER_MILLION_TOKENS
    )
    input_tokens = usage_token_count(usage, "input_tokens")
    output_tokens = usage_token_count(usage, "output_tokens")
    cache_read_tokens = usage_token_count(usage, "cache_read_input_tokens")
    cache_write_tokens = usage_token_count(usage, "cache_creation_input_tokens")
    return (
        input_tokens * ANTHROPIC_HAIKU_INPUT_USD_PER_MILLION_TOKENS
        + output_tokens * ANTHROPIC_HAIKU_OUTPUT_USD_PER_MILLION_TOKENS
        + cache_read_tokens * ANTHROPIC_HAIKU_CACHE_READ_INPUT_USD_PER_MILLION_TOKENS
        + cache_write_tokens * cache_write_rate
    ) / USD_PER_MILLION_TOKENS


def log_anthropic_usage(*, game_id: str, model: str, payload: dict[str, Any]) -> None:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    cache_ttl = anthropic_cache_ttl()
    response_id = str(payload.get("id") or "")
    input_tokens = usage_token_count(usage, "input_tokens")
    output_tokens = usage_token_count(usage, "output_tokens")
    cache_read_tokens = usage_token_count(usage, "cache_read_input_tokens")
    cache_write_tokens = usage_token_count(usage, "cache_creation_input_tokens")
    logger.info(
        (
            "%s: model usage provider=anthropic model=%s response_id=%s "
            "input_tokens=%s output_tokens=%s cache_read_input_tokens=%s "
            "cache_creation_input_tokens=%s cache_ttl=%s cost_usd=%.6f"
        ),
        game_id,
        model,
        response_id,
        input_tokens,
        output_tokens,
        cache_read_tokens,
        cache_write_tokens,
        cache_ttl,
        anthropic_usage_cost_usd(usage, cache_ttl=cache_ttl),
    )


def auth_headers() -> dict[str, str]:
    token = os.environ.get("KRIEGSPIEL_BOT_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def bot_username() -> str:
    return os.environ.get("KRIEGSPIEL_BOT_USERNAME", "").strip().lower()


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def save_token(token: str) -> None:
    state = load_state()
    state["token"] = token
    save_state(state)


def get_conversation_state(game_id: str) -> dict[str, Any]:
    state = load_state()
    conversations = state.get("conversations")
    if not isinstance(conversations, dict):
        return {}
    conversation = conversations.get(game_id)
    return conversation if isinstance(conversation, dict) else {}


def save_conversation_state(game_id: str, conversation: dict[str, Any]) -> None:
    state = load_state()
    conversations = state.get("conversations")
    if not isinstance(conversations, dict):
        conversations = {}
    conversations[game_id] = conversation
    state["conversations"] = conversations
    save_state(state)


def clear_conversation_state(game_id: str) -> None:
    state = load_state()
    conversations = state.get("conversations")
    if not isinstance(conversations, dict) or game_id not in conversations:
        return
    conversations.pop(game_id, None)
    state["conversations"] = conversations
    save_state(state)


def maybe_restore_token() -> None:
    if os.environ.get("KRIEGSPIEL_BOT_TOKEN"):
        return
    if STATE_PATH.exists():
        token = load_state().get("token")
        if token:
            os.environ["KRIEGSPIEL_BOT_TOKEN"] = token


def register_bot() -> None:
    response = requests.post(
        f"{base_url()}/auth/bots/register",
        json={
            "username": os.environ.get("KRIEGSPIEL_BOT_USERNAME", "haiku"),
            "display_name": os.environ.get("KRIEGSPIEL_BOT_DISPLAY_NAME", "Haiku"),
            "owner_email": os.environ.get("KRIEGSPIEL_BOT_OWNER_EMAIL", "bot-haiku@kriegspiel.org"),
            "description": os.environ.get(
                "KRIEGSPIEL_BOT_DESCRIPTION",
                "Model-driven Kriegspiel bot that chooses moves using Claude Haiku model.",
            ),
            "listed": True,
            "supported_rule_variants": supported_rule_variants(),
        },
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    save_token(payload["api_token"])
    logger.debug("%s", json.dumps(payload, indent=2))


def sync_bot_profile() -> bool:
    try:
        post_json("/bots/profile", {"supported_rule_variants": supported_rule_variants()})
    except requests.RequestException as exc:
        logger.warning("failed to sync bot profile: %s", exc)
        return False
    return True


def get_json(path: str) -> dict[str, Any]:
    response = requests.get(f"{base_url()}{path}", headers=auth_headers(), timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def get_public_user(username: str) -> dict[str, Any]:
    response = requests.get(f"{base_url()}/user/{username}", headers=auth_headers(), timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.post(
        f"{base_url()}{path}",
        headers=auth_headers(),
        json=payload or {},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def report_model_availability(ready: bool, reason: str, *, force: bool = False) -> bool:
    now = time.time()
    cached_ready = _MODEL_AVAILABILITY_REPORT_CACHE["ready"]
    cached_reason = str(_MODEL_AVAILABILITY_REPORT_CACHE["reason"])
    reported_at = float(_MODEL_AVAILABILITY_REPORT_CACHE["reported_at"])
    if (
        not force
        and cached_ready is bool(ready)
        and cached_reason == reason
        and now - reported_at < model_availability_report_interval_seconds()
    ):
        return False

    try:
        post_json("/bots/availability", {"provider": "anthropic", "ready": bool(ready), "reason": reason})
    except requests.RequestException as exc:
        logger.warning("failed to report Anthropic availability: %s", exc)
        return False

    _MODEL_AVAILABILITY_REPORT_CACHE.update({"ready": bool(ready), "reason": reason, "reported_at": now})
    return True


def report_current_model_availability() -> tuple[bool, str]:
    ready, reason = anthropic_preflight_status()
    report_model_availability(ready, reason)
    return ready, reason


def auto_create_enabled() -> bool:
    raw = os.environ.get("KRIEGSPIEL_AUTO_CREATE_LOBBY_GAME", "false").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def max_active_games_before_create() -> int:
    raw = os.environ.get("KRIEGSPIEL_MAX_ACTIVE_GAMES_BEFORE_CREATE", str(DEFAULT_MAX_ACTIVE_GAMES_BEFORE_CREATE)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MAX_ACTIVE_GAMES_BEFORE_CREATE


def create_payload() -> dict[str, str]:
    return {
        "rule_variant": os.environ.get("KRIEGSPIEL_AUTO_CREATE_RULE_VARIANT", "berkeley_any").strip() or "berkeley_any",
        "play_as": os.environ.get("KRIEGSPIEL_AUTO_CREATE_PLAY_AS", "random").strip() or "random",
        "time_control": "rapid",
        "opponent_type": "human",
    }


def supported_rule_variants() -> list[str]:
    raw = os.environ.get("KRIEGSPIEL_SUPPORTED_RULE_VARIANTS", DEFAULT_SUPPORTED_RULE_VARIANTS)
    variants: list[str] = []
    for item in raw.split(","):
        value = item.strip()
        if value in SUPPORTED_RULE_VARIANTS and value not in variants:
            variants.append(value)
    if tuple(variants) == LEGACY_DEFAULT_SUPPORTED_RULE_VARIANTS:
        return list(SUPPORTED_RULE_VARIANTS)
    return variants or list(SUPPORTED_RULE_VARIANTS)


def active_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [game for game in games if game.get("state") == "active"]


def waiting_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [game for game in games if game.get("state") == "waiting"]


def open_bot_lobby_candidates(open_games: list[dict[str, Any]], *, profile_lookup=None) -> list[dict[str, Any]]:
    profile_lookup = profile_lookup or get_public_user
    own_username = bot_username()
    candidates = []
    for game in open_games:
        creator_username = str(game.get("created_by") or "").strip()
        if not creator_username:
            continue
        if str(game.get("rule_variant") or "").strip() not in supported_rule_variants():
            continue
        creator_username_lower = creator_username.lower()
        if creator_username_lower == own_username:
            continue

        try:
            profile = profile_lookup(creator_username)
        except requests.RequestException:
            continue

        is_bot = bool(profile.get("is_bot")) or str(profile.get("role") or "").strip().lower() == "bot"
        if not is_bot:
            continue
        candidates.append(game)
    return candidates


def has_own_waiting_game(open_games: list[dict[str, Any]]) -> bool:
    own_username = bot_username()
    for game in open_games:
        created_by = str(game.get("created_by") or "").strip().lower()
        if created_by and created_by == own_username:
            return True
    return False


def can_attempt_bot_join(now: float | None = None) -> bool:
    current = time.time() if now is None else now
    last_attempt = load_state().get("last_bot_game_join_attempt_at", 0)
    try:
        last_attempt = float(last_attempt)
    except (TypeError, ValueError):
        last_attempt = 0
    return current - last_attempt >= BOT_JOIN_COOLDOWN_SECONDS


def record_bot_join_attempt(now: float | None = None) -> None:
    state = load_state()
    state["last_bot_game_join_attempt_at"] = time.time() if now is None else now
    save_state(state)


def should_join_bot_lobby_game(games: list[dict[str, Any]]) -> bool:
    return len(active_games(games)) < max_active_games_before_create()


def choose_bot_game_to_join(open_games: list[dict[str, Any]], *, rng: random.Random = random) -> dict[str, Any] | None:
    candidates = open_bot_lobby_candidates(open_games)
    if not candidates:
        return None
    return rng.choice(candidates)


def maybe_join_bot_lobby_game(games: list[dict[str, Any]], *, rng: random.Random = random) -> bool:
    if not should_join_bot_lobby_game(games):
        return False
    if not can_attempt_bot_join():
        return False

    record_bot_join_attempt()
    open_games = get_json("/game/open").get("games", [])
    candidate = choose_bot_game_to_join(open_games, rng=rng)
    if not candidate:
        return False
    if rng.random() >= BOT_GAME_PICK_PROBABILITY:
        return False

    ready, reason = anthropic_preflight_status()
    if not ready:
        logger.warning("skipping bot-game join because Anthropic is unavailable (%s)", reason)
        return False

    game_code = candidate.get("game_code")
    if not isinstance(game_code, str) or not game_code.strip():
        return False

    joined = post_json(f"/game/join/{game_code.strip()}")
    logger.debug("joined bot lobby game %s (%s)", joined["game_id"], joined["game_code"])
    return True


def should_create_lobby_game(games: list[dict[str, Any]]) -> bool:
    if not auto_create_enabled():
        return False
    if waiting_games(games):
        return False
    return len(active_games(games)) < max_active_games_before_create()


def maybe_create_lobby_game(games: list[dict[str, Any]]) -> bool:
    if not should_create_lobby_game(games):
        return False

    open_games = get_json("/game/open").get("games", [])
    if has_own_waiting_game(open_games):
        return False

    created = post_json("/game/create", create_payload())
    logger.debug("created lobby game %s (%s)", created["game_id"], created["game_code"])
    return True


def recent_scoresheet_turns(scoresheet: dict[str, Any], *, max_turns: int) -> list[dict[str, Any]]:
    turns = scoresheet.get("turns") if isinstance(scoresheet.get("turns"), list) else []
    recent_turns = turns[-max_turns:]
    payload: list[dict[str, Any]] = []
    for turn in recent_turns:
        item: dict[str, Any] = {"turn": turn.get("turn")}
        for color in ("white", "black"):
            entries = turn.get(color) if isinstance(turn.get(color), list) else []
            item[color] = [
                normalized
                for normalized in (normalize_scoresheet_entry(entry) for entry in entries)
                if normalized
            ]
        payload.append(item)
    return payload


def summarize_scoresheet_turns(scoresheet: dict[str, Any], *, max_turns: int) -> list[str]:
    lines: list[str] = []
    for turn in recent_scoresheet_turns(scoresheet, max_turns=max_turns):
        turn_number = turn.get("turn")
        for color in ("white", "black"):
            entries = turn.get(color) if isinstance(turn.get(color), list) else []
            for entry in entries:
                lines.append(f"Turn {turn_number} {color}: {entry}")
    return lines


def normalize_scoresheet_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if not isinstance(entry, dict):
        return ""

    prompt = str(entry.get("prompt") or "").strip()
    messages = entry.get("messages") if isinstance(entry.get("messages"), list) else []
    cleaned_messages = [str(item).strip() for item in messages if str(item).strip()]
    if not cleaned_messages:
        message = str(entry.get("message") or "").strip()
        if message:
            cleaned_messages = [message]

    text = " | ".join(dict.fromkeys(cleaned_messages))
    move_uci = str(entry.get("move_uci") or "").strip()

    if move_uci:
        text = f"[{move_uci}] {text}" if text else f"[{move_uci}]"
    if prompt and text:
        return f"{prompt}: {text}"
    return text or prompt


def model_batch_size() -> int:
    raw = os.environ.get("ANTHROPIC_MODEL_BATCH_SIZE", str(DEFAULT_MODEL_BATCH_SIZE)).strip()
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return DEFAULT_MODEL_BATCH_SIZE


def max_model_batches_per_turn() -> int:
    raw = os.environ.get("ANTHROPIC_MAX_BATCHES_PER_TURN", str(DEFAULT_MAX_MODEL_BATCHES_PER_TURN)).strip()
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return DEFAULT_MAX_MODEL_BATCHES_PER_TURN


def anthropic_max_prompt_turns() -> int:
    raw = os.environ.get("ANTHROPIC_MAX_PROMPT_TURNS", str(DEFAULT_ANTHROPIC_MAX_PROMPT_TURNS)).strip()
    try:
        return max(DEFAULT_ANTHROPIC_MAX_PROMPT_TURNS, int(raw))
    except ValueError:
        return DEFAULT_ANTHROPIC_MAX_PROMPT_TURNS


def extract_recent_referee_items(scoresheet: dict[str, Any], *, limit: int = 8) -> list[str]:
    turns = scoresheet.get("turns") if isinstance(scoresheet.get("turns"), list) else []
    lines: list[str] = []
    for turn in turns[-limit:]:
        turn_number = turn.get("turn")
        for color in ("white", "black"):
            entries = turn.get(color) if isinstance(turn.get(color), list) else []
            for entry in entries:
                normalized = normalize_scoresheet_entry(entry)
                if normalized:
                    lines.append(f"Turn {turn_number} {color}: {normalized}")
    return lines[-limit:]


def scoresheet_digest(scoresheet: dict[str, Any]) -> str:
    payload = json.dumps(scoresheet, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def turn_signature(state: dict[str, Any]) -> str:
    return "|".join(
        [
            str(state.get("state") or ""),
            str(state.get("move_number") or ""),
            str(state.get("turn") or ""),
            str(state.get("your_color") or ""),
        ]
    )


def new_recent_items(previous: list[str], current: list[str]) -> list[str]:
    if not previous:
        return current
    best_overlap = 0
    max_overlap = min(len(previous), len(current))
    for size in range(max_overlap, 0, -1):
        if previous[-size:] == current[:size]:
            best_overlap = size
            break
    return current[best_overlap:]


@lru_cache(maxsize=len(SUPPORTED_RULE_VARIANTS) + 1)
def load_ruleset_summary(rule_variant: str) -> str:
    variant = rule_variant if rule_variant in SUPPORTED_RULE_VARIANTS else "berkeley_any"
    return (RULESET_SUMMARY_DIR / f"{variant}.md").read_text(encoding="utf-8").strip()


def build_system_prompt(rule_variant: str) -> str:
    rules_summary = load_ruleset_summary(rule_variant)
    return (
        "You are a strong Kriegspiel player.\n"
        "Use only the provided private information and legal actions.\n"
        "Do not invent moves. Do not suggest illegal actions.\n"
        "Return only a JSON object shaped as {\"candidates\":[{\"action\":\"move\",\"uci\":\"e2e4\"}]}.\n"
        "Return minified JSON with no spaces, newlines, or prose.\n"
        "Turn JSON keys: c=your color,t=side to move,mn=move number,fen=private board FEN,"
        "mat=material,hist=recent scorecard,act=possible actions,moves=legal UCI moves,"
        "n=target candidate count,res=reserves,rej=rejected actions,fb=retry feedback.\n"
        "Colors inside mat/hist/res use w=white and b=black; hist items use n=turn,w=white entries,b=black entries.\n"
        "Return exactly n unique candidate actions ordered strictly from best to worst priority.\n"
        "Candidate 1 must be your best choice, candidate 2 your next-best choice, and so on.\n"
        "Prioritize strategically strong, tactically sound moves that are robust under uncertainty.\n"
        "If action=move, uci must exactly match one moves item.\n"
        "If action=ask_any, ask_any must appear in act and uci must be null.\n"
        "Do not explain the rules. Do not include prose outside the JSON object.\n\n"
        f"{rules_summary}\n"
    )


def _compact_color_key(color: str) -> str:
    return "w" if color == "white" else "b"


def _prompt_material_summary(state: dict[str, Any]) -> dict[str, dict[str, int]]:
    material = state.get("material_summary") if isinstance(state.get("material_summary"), dict) else {}
    payload: dict[str, dict[str, int]] = {}
    for color in ("white", "black"):
        side = material.get(color) if isinstance(material.get(color), dict) else {}
        if not side:
            continue
        item: dict[str, int] = {}
        pieces_remaining = side.get("pieces_remaining")
        if isinstance(pieces_remaining, int):
            item["pieces"] = pieces_remaining
        pawns_captured = side.get("pawns_captured")
        if isinstance(pawns_captured, int):
            item["pawns_captured"] = pawns_captured
        if item:
            payload[_compact_color_key(color)] = item
    return payload


def _prompt_reserve_summary(state: dict[str, Any], *, rule_variant: str) -> dict[str, dict[str, int]]:
    if rule_variant != "crazykrieg":
        return {}
    reserves = state.get("reserve_summary") if isinstance(state.get("reserve_summary"), dict) else {}
    payload: dict[str, dict[str, int]] = {}
    for color in ("white", "black"):
        side = reserves.get(color) if isinstance(reserves.get(color), dict) else {}
        if side:
            payload[_compact_color_key(color)] = {
                piece: int(side.get(piece, 0) or 0)
                for piece in ("pawns", "knights", "bishops", "rooks", "queens")
            }
    return payload


def compact_recent_scoresheet_turns(scoresheet: dict[str, Any], *, max_turns: int) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for turn in recent_scoresheet_turns(scoresheet, max_turns=max_turns):
        item: dict[str, Any] = {"n": turn.get("turn")}
        for color in ("white", "black"):
            key = _compact_color_key(color)
            entries = turn.get(color) if isinstance(turn.get(color), list) else []
            item[key] = entries
        compact.append(item)
    return compact


def build_turn_snapshot_payload(
    state: dict[str, Any],
    *,
    feedback: list[str] | None = None,
    exclude_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rule_variant = str(state.get("rule_variant") or "berkeley_any")
    scoresheet = state.get("scoresheet") if isinstance(state.get("scoresheet"), dict) else {}
    allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []
    possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []
    available_action_count = len(allowed_moves) + (1 if "ask_any" in possible_actions else 0)
    target_count = min(model_batch_size(), available_action_count)
    payload: dict[str, Any] = {
        "c": state.get("your_color"),
        "t": state.get("turn"),
        "mn": state.get("move_number"),
        "fen": state.get("your_fen"),
        "mat": _prompt_material_summary(state),
        "hist": compact_recent_scoresheet_turns(scoresheet, max_turns=anthropic_max_prompt_turns()),
        "act": possible_actions,
        "moves": allowed_moves,
        "n": target_count,
    }
    reserves = _prompt_reserve_summary(state, rule_variant=rule_variant)
    if reserves:
        payload["res"] = reserves
    rejected = [format_action(item) for item in (exclude_actions or [])[-20:]]
    if rejected:
        payload["rej"] = rejected
    retry_feedback = [item for item in (feedback or [])[-6:] if item]
    if retry_feedback:
        payload["fb"] = retry_feedback
    return payload


def build_turn_snapshot_user_prompt(
    state: dict[str, Any],
    *,
    feedback: list[str] | None = None,
    exclude_actions: list[dict[str, Any]] | None = None,
) -> str:
    payload = build_turn_snapshot_payload(
        state,
        feedback=feedback,
        exclude_actions=exclude_actions,
    )
    return f"Current turn JSON:\n{json.dumps(payload, separators=(',', ':'), ensure_ascii=True, sort_keys=True)}"


def build_initial_user_prompt(state: dict[str, Any]) -> str:
    return build_turn_snapshot_user_prompt(state)


def build_followup_user_prompt(
    state: dict[str, Any],
    *,
    feedback: list[str] | None = None,
    exclude_actions: list[dict[str, Any]] | None = None,
    recent_updates: list[str] | None = None,
) -> str:
    _ = recent_updates
    return build_turn_snapshot_user_prompt(
        state,
        feedback=feedback,
        exclude_actions=exclude_actions,
    )


def action_schema() -> dict[str, Any]:
    return {
        "name": ACTION_SCHEMA_NAME,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action": {"type": "string", "enum": ["move", "ask_any"]},
                            "uci": {"type": ["string", "null"]},
                        },
                        "required": ["action", "uci"],
                    },
                }
            },
            "required": ["candidates"],
        },
        "strict": True,
    }


def action_tool() -> dict[str, Any]:
    return {
        "name": ACTION_SCHEMA_NAME,
        "description": "Return ranked Kriegspiel candidate actions for the current turn.",
        "input_schema": action_schema()["schema"],
        "strict": True,
    }


def anthropic_enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def anthropic_use_tools() -> bool:
    raw = os.environ.get("ANTHROPIC_USE_TOOLS", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def cache_anthropic_preflight(ready: bool, *, reason: str, ttl_seconds: float) -> tuple[bool, str]:
    _ANTHROPIC_PREFLIGHT_CACHE["ready"] = ready
    _ANTHROPIC_PREFLIGHT_CACHE["reason"] = reason
    _ANTHROPIC_PREFLIGHT_CACHE["expires_at"] = time.time() + max(0.0, ttl_seconds)
    return ready, reason


def describe_http_error(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc) or exc.__class__.__name__

    pieces = [f"http_{response.status_code}"]
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            for key in ("type", "code", "message"):
                value = error.get(key)
                if value:
                    pieces.append(str(value))
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            pieces.append(detail)
    elif response.text:
        pieces.append(response.text.strip())

    return ": ".join(dict.fromkeys(piece for piece in pieces if piece))


def anthropic_preflight_status(force: bool = False) -> tuple[bool, str]:
    if not anthropic_enabled():
        return cache_anthropic_preflight(False, reason="missing_anthropic_api_key", ttl_seconds=anthropic_preflight_failure_ttl_seconds())

    now = time.time()
    if not force and _ANTHROPIC_PREFLIGHT_CACHE["ready"] is not None and now < float(_ANTHROPIC_PREFLIGHT_CACHE["expires_at"]):
        return bool(_ANTHROPIC_PREFLIGHT_CACHE["ready"]), str(_ANTHROPIC_PREFLIGHT_CACHE["reason"])

    try:
        response = requests.post(
            f"{anthropic_base_url()}/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"].strip(),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip(),
                "max_tokens": 1,
                "system": "Reply with OK.",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Ping"}]}],
            },
            timeout=anthropic_timeout_seconds(),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        reason = describe_http_error(exc)
        logger.warning("anthropic preflight failed: %s", reason)
        return cache_anthropic_preflight(False, reason=reason, ttl_seconds=anthropic_preflight_failure_ttl_seconds())

    return cache_anthropic_preflight(True, reason="ok", ttl_seconds=anthropic_preflight_success_ttl_seconds())


def call_anthropic_messages(
    *,
    system_prompt: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    api_key = os.environ["ANTHROPIC_API_KEY"].strip()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip()
    cache_control = {"type": "ephemeral", "ttl": anthropic_cache_ttl()}
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": anthropic_max_output_tokens(),
        "cache_control": cache_control,
        "system": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": cache_control,
            }
        ],
        "messages": messages,
    }
    if anthropic_use_tools():
        payload["tools"] = [action_tool()]
        payload["tool_choice"] = {"type": "tool", "name": ACTION_SCHEMA_NAME}
    response = requests.post(
        f"{anthropic_base_url()}/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=anthropic_timeout_seconds(),
    )
    response.raise_for_status()
    return response.json()


def extract_tool_input(payload: dict[str, Any]) -> dict[str, Any] | None:
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "tool_use" or item.get("name") != ACTION_SCHEMA_NAME:
            continue
        tool_input = item.get("input")
        if isinstance(tool_input, dict):
            return tool_input
    return None


def extract_response_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks)
    raise ValueError("No text found in Anthropic response payload")


def decode_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    first_error: json.JSONDecodeError | None = None
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError as exc:
            if first_error is None:
                first_error = exc
            continue
        if isinstance(value, dict):
            return value
    if first_error is not None:
        raise first_error
    raise ValueError("No JSON object found in model response")


def parse_model_decision(payload: dict[str, Any]) -> dict[str, Any]:
    tool_input = extract_tool_input(payload)
    if tool_input is not None:
        return tool_input

    text = extract_response_text(payload)
    try:
        decision = json.loads(text)
    except json.JSONDecodeError:
        decision = decode_first_json_object(text)

    if not isinstance(decision, dict):
        raise ValueError("Model response must decode to an object")
    return decision


def normalize_decision(decision: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []
    allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []

    action = str(decision.get("action") or "").strip().lower()
    uci = decision.get("uci")

    if action == "move":
        if "move" not in possible_actions or not isinstance(uci, str):
            return None
        normalized_uci = uci.strip()
        allowed_by_lower = {str(move).strip().lower(): str(move).strip() for move in allowed_moves}
        allowed_move = allowed_by_lower.get(normalized_uci.lower())
        if allowed_move is None:
            return None
        return {"action": "move", "uci": allowed_move}

    if action == "ask_any":
        if "ask_any" not in possible_actions:
            return None
        return {"action": "ask_any", "uci": None}

    return None


def normalize_ranked_decisions(payload: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        decision = normalize_decision(candidate, state)
        if decision is None:
            continue
        key = (decision["action"], decision["uci"])
        if key in seen:
            continue
        seen.add(key)
        normalized.append(decision)
    return normalized


def fallback_decision(state: dict[str, Any]) -> dict[str, Any] | None:
    allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []
    possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []

    if allowed_moves:
        ordered = sorted(allowed_moves)
        center_bias = ["d2d4", "e2e4", "d7d5", "e7e5"]
        for preferred in center_bias:
            if preferred in ordered:
                return {"action": "move", "uci": preferred}
        return {"action": "move", "uci": ordered[0]}
    if "ask_any" in possible_actions:
        return {"action": "ask_any", "uci": None}
    return None


def fallback_ranked_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    decision = fallback_decision(state)
    return [decision] if decision is not None else []


def choose_ranked_actions(
    state: dict[str, Any],
    *,
    game_id: str,
    conversation: dict[str, Any] | None = None,
    feedback: list[str] | None = None,
    exclude_actions: list[dict[str, Any]] | None = None,
    recent_updates: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    if not anthropic_enabled():
        return fallback_ranked_actions(state), "fallback_no_anthropic_key", None

    try:
        system_prompt = build_system_prompt(state.get("rule_variant", "berkeley_any"))
        _ = conversation, recent_updates
        user_prompt = build_turn_snapshot_user_prompt(
            state,
            feedback=feedback,
            exclude_actions=exclude_actions,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
        raw_response = call_anthropic_messages(system_prompt=system_prompt, messages=messages)
        log_anthropic_usage(
            game_id=game_id,
            model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip(),
            payload=raw_response,
        )
        decisions = normalize_ranked_decisions(parse_model_decision(raw_response), state)
        if decisions:
            return decisions, "model", None
    except requests.RequestException as exc:
        reason = describe_http_error(exc)
        logger.warning("model selection failed: %s", reason)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("model selection failed: %s", exc)

    return fallback_ranked_actions(state), "fallback", None


def format_action(decision: dict[str, Any]) -> str:
    if decision["action"] == "ask_any":
        return "ask_any"
    return str(decision["uci"])


def maybe_play_game(game_id: str) -> bool:
    metadata = get_json(f"/game/{game_id}")
    feedback: list[str] = []
    tried_actions: list[dict[str, Any]] = []
    acted = False
    conversation = get_conversation_state(game_id)

    for _batch in range(max_model_batches_per_turn()):
        state = get_json(f"/game/{game_id}/state")
        if state.get("state") != "active" or state.get("turn") != state.get("your_color"):
            return acted
        if isinstance(metadata.get("rule_variant"), str):
            state["rule_variant"] = metadata["rule_variant"]

        scoresheet = state.get("scoresheet") if isinstance(state.get("scoresheet"), dict) else {}
        recent_referee = extract_recent_referee_items(scoresheet, limit=10)
        current_turn_signature = turn_signature(state)
        previous_turn_signature = str(conversation.get("turn_signature") or "")
        if previous_turn_signature != current_turn_signature:
            tried_actions = []
            feedback = []

        previous_recent_referee = conversation.get("recent_referee_items")
        if not isinstance(previous_recent_referee, list):
            previous_recent_referee = []
        recent_updates = new_recent_items(previous_recent_referee, recent_referee)

        decisions, source, _response_id = choose_ranked_actions(
            state,
            game_id=game_id,
            conversation=conversation,
            feedback=feedback,
            exclude_actions=tried_actions,
            recent_updates=recent_updates,
        )
        conversation = {
            "turn_signature": current_turn_signature,
            "scoresheet_digest": scoresheet_digest(scoresheet),
            "recent_referee_items": recent_referee,
        }
        save_conversation_state(game_id, conversation)
        if not decisions:
            return acted

        batch_success = False
        for decision in decisions:
            tried_actions.append(decision)
            if decision["action"] == "move":
                result = post_json(f"/game/{game_id}/move", {"uci": decision["uci"]})
                acted = acted or bool(result.get("move_done"))
                logger.debug("%s: %s move %s -> %s", game_id, source, decision["uci"], result["announcement"])
                if result.get("move_done"):
                    state_after = get_json(f"/game/{game_id}/state")
                    if isinstance(metadata.get("rule_variant"), str):
                        state_after["rule_variant"] = metadata["rule_variant"]
                    scoresheet_after = state_after.get("scoresheet") if isinstance(state_after.get("scoresheet"), dict) else {}
                    recent_after = extract_recent_referee_items(scoresheet_after, limit=10)
                    recent_updates_after = new_recent_items(recent_referee, recent_after)
                    feedback = [f"Move complete: {result.get('announcement')}"]
                    if recent_updates_after:
                        feedback.append("New referee announcements: " + " || ".join(recent_updates_after))
                    conversation = {
                        "turn_signature": turn_signature(state_after),
                        "scoresheet_digest": scoresheet_digest(scoresheet_after),
                        "recent_referee_items": recent_after,
                    }
                    save_conversation_state(game_id, conversation)
                    tried_actions = []
                    batch_success = True
                    break
                feedback.append(f"Rejected move {decision['uci']}: {result.get('announcement')}")
                continue

            result = post_json(f"/game/{game_id}/ask-any")
            acted = acted or bool(result.get("move_done"))
            logger.debug("%s: %s ask-any -> %s", game_id, source, result["announcement"])
            state_after = get_json(f"/game/{game_id}/state")
            if isinstance(metadata.get("rule_variant"), str):
                state_after["rule_variant"] = metadata["rule_variant"]
            scoresheet_after = state_after.get("scoresheet") if isinstance(state_after.get("scoresheet"), dict) else {}
            recent_after = extract_recent_referee_items(scoresheet_after, limit=10)
            recent_updates_after = new_recent_items(recent_referee, recent_after)
            feedback = [f"Ask-any result: {result.get('announcement')}"]
            if recent_updates_after:
                feedback.append("New referee announcements: " + " || ".join(recent_updates_after))
            feedback.append(
                "New possible actions: "
                + json.dumps(
                    {
                        "act": state_after.get("possible_actions"),
                        "moves": state_after.get("allowed_moves"),
                    },
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            )
            conversation = {
                "turn_signature": turn_signature(state_after),
                "scoresheet_digest": scoresheet_digest(scoresheet_after),
                "recent_referee_items": recent_after,
            }
            save_conversation_state(game_id, conversation)
            tried_actions = []
            batch_success = True
            break

        if not batch_success:
            refreshed_state = get_json(f"/game/{game_id}/state")
            if isinstance(metadata.get("rule_variant"), str):
                refreshed_state["rule_variant"] = metadata["rule_variant"]
            refreshed_scoresheet = refreshed_state.get("scoresheet") if isinstance(refreshed_state.get("scoresheet"), dict) else {}
            refreshed_recent = extract_recent_referee_items(refreshed_scoresheet, limit=10)
            feedback.append("Top-ranked batch failed; provide the next best legal candidates.")
            feedback.append(
                "Current legal options now: "
                + json.dumps(
                    {
                        "act": refreshed_state.get("possible_actions"),
                        "moves": refreshed_state.get("allowed_moves"),
                    },
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            )
            conversation = {
                "turn_signature": turn_signature(refreshed_state),
                "scoresheet_digest": scoresheet_digest(refreshed_scoresheet),
                "recent_referee_items": refreshed_recent,
            }
            save_conversation_state(game_id, conversation)
            continue

    return acted


def run_loop(poll_seconds: float) -> None:
    while True:
        try:
            report_current_model_availability()
            mine = get_json("/game/mine/active")
            games = mine.get("games", [])
            active_game_ids = {game.get("game_id") for game in active_games(games)}
            state = load_state()
            conversations = state.get("conversations")
            if isinstance(conversations, dict):
                for game_id in list(conversations.keys()):
                    if game_id not in active_game_ids:
                        clear_conversation_state(str(game_id))
            maybe_create_lobby_game(games)
            maybe_join_bot_lobby_game(games)
            for game in active_games(games):
                maybe_play_game(game["game_id"])
        except requests.RequestException as exc:
            logger.warning("poll failed: %s", exc)
        time.sleep(poll_seconds)


def main() -> None:
    load_env_file()
    maybe_restore_token()

    parser = argparse.ArgumentParser(description="Run the Kriegspiel Haiku bot.")
    parser.add_argument("--register", action="store_true", help="Register the bot and persist the returned token.")
    parser.add_argument("--poll-seconds", type=float, default=3.0, help="Seconds between /game/mine/active polls.")
    args = parser.parse_args()

    if args.register:
        register_bot()
        return

    if not os.environ.get("KRIEGSPIEL_BOT_TOKEN"):
        raise SystemExit("KRIEGSPIEL_BOT_TOKEN is missing. Run with --register first.")
    if not anthropic_enabled():
        logger.warning("ANTHROPIC_API_KEY is missing; bot-vs-bot joins will be skipped and turns will use fallback mode.")

    sync_bot_profile()
    run_loop(args.poll_seconds)


if __name__ == "__main__":
    main()
