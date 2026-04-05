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
import sys
import time
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / ".bot-state.json"
ENV_PATH = BASE_DIR / ".env"
CONTENT_DIR = BASE_DIR.parent / "content" / "rules"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_ANTHROPIC_OUTPUT_TOKENS = 512
ACTION_SCHEMA_NAME = "kriegspiel_next_action"
DEFAULT_MAX_ACTIVE_GAMES_BEFORE_CREATE = 1
BOT_JOIN_COOLDOWN_SECONDS = 60
BOT_GAME_PICK_PROBABILITY = 0.01
DEFAULT_MODEL_BATCH_SIZE = 10
DEFAULT_MAX_MODEL_BATCHES_PER_TURN = 5
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
        f"{base_url()}/api/auth/bots/register",
        headers={"X-Bot-Registration-Key": os.environ["KRIEGSPIEL_BOT_REGISTRATION_KEY"]},
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


def get_json(path: str) -> dict[str, Any]:
    response = requests.get(f"{base_url()}{path}", headers=auth_headers(), timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def get_public_user(username: str) -> dict[str, Any]:
    response = requests.get(f"{base_url()}/api/user/{username}", headers=auth_headers(), timeout=DEFAULT_TIMEOUT_SECONDS)
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
    raw = os.environ.get("KRIEGSPIEL_SUPPORTED_RULE_VARIANTS", "berkeley,berkeley_any")
    variants: list[str] = []
    for item in raw.split(","):
        value = item.strip()
        if value in {"berkeley", "berkeley_any"} and value not in variants:
            variants.append(value)
    return variants or ["berkeley", "berkeley_any"]


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
    if rng.random() >= BOT_GAME_PICK_PROBABILITY:
        return None
    return rng.choice(candidates)


def maybe_join_bot_lobby_game(games: list[dict[str, Any]], *, rng: random.Random = random) -> bool:
    if not should_join_bot_lobby_game(games):
        return False
    if not can_attempt_bot_join():
        return False

    open_games = get_json("/api/game/open").get("games", [])
    candidate = choose_bot_game_to_join(open_games, rng=rng)
    if not candidate:
        return False

    game_code = candidate.get("game_code")
    if not isinstance(game_code, str) or not game_code.strip():
        return False

    record_bot_join_attempt()
    joined = post_json(f"/api/game/join/{game_code.strip()}")
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

    open_games = get_json("/api/game/open").get("games", [])
    if has_own_waiting_game(open_games):
        return False

    created = post_json("/api/game/create", create_payload())
    logger.debug("created lobby game %s (%s)", created["game_id"], created["game_code"])
    return True


def strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text.strip()
    parts = text.split("\n---\n", 1)
    return parts[1].strip() if len(parts) == 2 else text.strip()


def _variant_any_rule_excerpt() -> str:
    readme = (CONTENT_DIR / "README.md").read_text()
    lines = []
    for line in readme.splitlines():
        if line.startswith("| Berkeley + Any"):
            lines.append(line.strip())
    return "\n".join(lines)


@lru_cache(maxsize=4)
def load_rules_text(rule_variant: str) -> str:
    berkeley = strip_frontmatter((CONTENT_DIR / "berkeley.md").read_text())
    if rule_variant == "berkeley_any":
        any_excerpt = _variant_any_rule_excerpt()
        appendix = (
            "\n\nAdditional variant note for Berkeley + Any:\n"
            "The current game allows the special 'Any?' action. "
            "If the referee says there are pawn captures, the player must make one.\n"
        )
        if any_excerpt:
            appendix += f"\nRules comparison excerpt:\n{any_excerpt}\n"
        return berkeley + appendix
    return berkeley


def summarize_scoresheet_turns(scoresheet: dict[str, Any], *, max_turns: int) -> list[str]:
    viewer_color = scoresheet.get("viewer_color", "unknown")
    turns = scoresheet.get("turns") if isinstance(scoresheet.get("turns"), list) else []
    recent_turns = turns[-max_turns:]
    lines = [f"Viewer color: {viewer_color}", f"Recent turns kept: {len(recent_turns)}"]

    for turn in recent_turns:
        turn_number = turn.get("turn")
        lines.append(f"Turn {turn_number}:")
        for color in ("white", "black"):
            entries = turn.get(color) if isinstance(turn.get(color), list) else []
            if not entries:
                continue
            for entry in entries:
                normalized = normalize_scoresheet_entry(entry)
                if normalized:
                    lines.append(f"- {color}: {normalized}")
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


def build_system_prompt(rule_variant: str) -> str:
    rules_text = load_rules_text(rule_variant)
    return (
        "You are a strong Kriegspiel player.\n"
        "Use only the provided private information and legal actions.\n"
        "Do not invent moves. Do not suggest illegal actions.\n"
        "Return unique candidate actions ordered strictly from best to worse priority.\n"
        "Candidate 1 must be your best choice, candidate 2 your next-best choice, and so on.\n"
        "Prioritize strategically strong, tactically sound moves that are robust under uncertainty.\n"
        "Do not explain the rules. Do not include prose outside the JSON schema.\n\n"
        "Rules and setting:\n"
        f"Rule variant: {rule_variant}\n"
        f"{rules_text}\n"
    )


def build_initial_user_prompt(
    state: dict[str, Any],
) -> str:
    scoresheet = state.get("scoresheet") if isinstance(state.get("scoresheet"), dict) else {}
    allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []
    possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []
    max_prompt_turns = int(os.environ.get("ANTHROPIC_MAX_PROMPT_TURNS", "12"))
    scoresheet_text = summarize_scoresheet_turns(scoresheet, max_turns=max_prompt_turns)
    recent_referee = extract_recent_referee_items(scoresheet, limit=6)
    target_count = min(model_batch_size(), max(len(allowed_moves), 1 if "ask_any" in possible_actions else 0))
    payload = {
        "phase": "initial_turn_snapshot",
        "your_color": state.get("your_color"),
        "game_state": state.get("state"),
        "turn": state.get("turn"),
        "move_number": state.get("move_number"),
        "private_board_fen": state.get("your_fen"),
        "possible_actions": possible_actions,
        "allowed_moves": allowed_moves,
        "recent_scoresheet_lines": scoresheet_text[-12:],
        "recent_referee_items": recent_referee,
        "target_count": target_count,
    }
    return (
        "Current private state JSON follows.\n"
        "Return exactly target_count unique candidates when possible, ordered from best to worse priority.\n"
        "If action=move, uci must be one of allowed_moves exactly.\n"
        "If action=ask_any, uci must be null.\n\n"
        f"{json.dumps(payload, separators=(',', ':'), ensure_ascii=True, sort_keys=True)}"
    )


def build_followup_user_prompt(
    state: dict[str, Any],
    *,
    feedback: list[str] | None = None,
    exclude_actions: list[dict[str, Any]] | None = None,
    recent_updates: list[str] | None = None,
) -> str:
    allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []
    possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []
    target_count = min(model_batch_size(), max(len(allowed_moves), 1 if "ask_any" in possible_actions else 0))
    payload = {
        "phase": "turn_update",
        "turn": state.get("turn"),
        "move_number": state.get("move_number"),
        "possible_actions": possible_actions,
        "allowed_moves": allowed_moves,
        "new_scoresheet_items": (recent_updates or [])[-10:],
        "feedback_this_turn": (feedback or [])[-6:],
        "already_tried_this_turn": [format_action(item) for item in (exclude_actions or [])[-20:]],
        "target_count": target_count,
    }
    return (
        "Update since your last ranked list JSON follows.\n"
        "Use the new_scoresheet_items and feedback_this_turn to avoid stale ideas.\n"
        "Return exactly target_count unique candidates when possible, ordered from best to worse priority.\n"
        "If action=move, uci must be one of allowed_moves exactly.\n"
        "If action=ask_any, uci must be null.\n\n"
        f"{json.dumps(payload, separators=(',', ':'), ensure_ascii=True, sort_keys=True)}"
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
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action": {"type": "string", "enum": ["move", "ask_any"]},
                            "uci": {"type": ["string", "null"]},
                            "reason": {"type": "string"},
                        },
                        "required": ["action", "uci", "reason"],
                    },
                }
            },
            "required": ["candidates"],
        },
        "strict": True,
    }


def anthropic_enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def call_anthropic_messages(
    *,
    system_prompt: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    api_key = os.environ["ANTHROPIC_API_KEY"].strip()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001").strip()
    response = requests.post(
        f"{anthropic_base_url()}/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": anthropic_max_output_tokens(),
            "cache_control": {"type": "ephemeral", "ttl": anthropic_cache_ttl()},
            "system": system_prompt,
            "messages": messages,
        },
        timeout=anthropic_timeout_seconds(),
    )
    response.raise_for_status()
    return response.json()


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


def conversation_messages(conversation: dict[str, Any] | None) -> list[dict[str, Any]]:
    raw_messages = (conversation or {}).get("messages")
    if not isinstance(raw_messages, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": [{"type": "text", "text": content}]})
    return messages


def persistable_conversation_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    persisted: list[dict[str, str]] = []
    for item in messages:
        role = str(item.get("role") or "").strip()
        content_blocks = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content_blocks, list):
            continue
        texts = [str(block.get("text")).strip() for block in content_blocks if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text")).strip()]
        if texts:
            persisted.append({"role": role, "content": "\n".join(texts)})
    return persisted


def parse_model_decision(payload: dict[str, Any]) -> dict[str, Any]:
    text = extract_response_text(payload)
    try:
        decision = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        decision = json.loads(match.group(0))

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
        normalized_uci = uci.strip().lower()
        if normalized_uci not in allowed_moves:
            return None
        return {"action": "move", "uci": normalized_uci}

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
) -> tuple[list[dict[str, Any]], str, list[dict[str, str]] | None]:
    if not anthropic_enabled():
        return fallback_ranked_actions(state), "fallback_no_anthropic_key", None

    try:
        system_prompt = build_system_prompt(state.get("rule_variant", "berkeley_any"))
        prior_messages = conversation_messages(conversation)
        user_prompt = (
            build_followup_user_prompt(
                state,
                feedback=feedback,
                exclude_actions=exclude_actions,
                recent_updates=recent_updates,
            )
            if prior_messages
            else build_initial_user_prompt(state)
        )
        messages = prior_messages + [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
        raw_response = call_anthropic_messages(system_prompt=system_prompt, messages=messages)
        decisions = normalize_ranked_decisions(parse_model_decision(raw_response), state)
        assistant_text = extract_response_text(raw_response)
        usage = raw_response.get("usage") if isinstance(raw_response.get("usage"), dict) else {}
        cache_read_tokens = usage.get("cache_read_input_tokens", 0)
        cache_write_tokens = usage.get("cache_creation_input_tokens", 0)
        logger.debug("%s: model usage cache_read=%s cache_write=%s", game_id, cache_read_tokens, cache_write_tokens)
        if decisions:
            next_messages = messages + [{"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}]
            return decisions, "model", persistable_conversation_messages(next_messages)
    except (KeyError, ValueError, json.JSONDecodeError, requests.RequestException) as exc:
        logger.warning("model selection failed: %s", exc)

    return fallback_ranked_actions(state), "fallback", None


def format_action(decision: dict[str, Any]) -> str:
    if decision["action"] == "ask_any":
        return "ask_any"
    return str(decision["uci"])


def maybe_play_game(game_id: str) -> bool:
    metadata = get_json(f"/api/game/{game_id}")
    feedback: list[str] = []
    tried_actions: list[dict[str, Any]] = []
    acted = False
    conversation = get_conversation_state(game_id)

    for _batch in range(max_model_batches_per_turn()):
        state = get_json(f"/api/game/{game_id}/state")
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

        decisions, source, conversation_messages_for_save = choose_ranked_actions(
            state,
            game_id=game_id,
            conversation=conversation,
            feedback=feedback,
            exclude_actions=tried_actions,
            recent_updates=recent_updates,
        )
        conversation = {
            "messages": conversation_messages_for_save or conversation.get("messages", []),
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
                result = post_json(f"/api/game/{game_id}/move", {"uci": decision["uci"]})
                acted = acted or bool(result.get("move_done"))
                logger.debug("%s: %s move %s -> %s", game_id, source, decision["uci"], result["announcement"])
                if result.get("move_done"):
                    state_after = get_json(f"/api/game/{game_id}/state")
                    if isinstance(metadata.get("rule_variant"), str):
                        state_after["rule_variant"] = metadata["rule_variant"]
                    scoresheet_after = state_after.get("scoresheet") if isinstance(state_after.get("scoresheet"), dict) else {}
                    recent_after = extract_recent_referee_items(scoresheet_after, limit=10)
                    recent_updates_after = new_recent_items(recent_referee, recent_after)
                    feedback = [f"Move complete: {result.get('announcement')}"]
                    if recent_updates_after:
                        feedback.append("New referee announcements: " + " || ".join(recent_updates_after))
                    conversation = {
                        "messages": conversation.get("messages", []),
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

            result = post_json(f"/api/game/{game_id}/ask-any")
            acted = acted or bool(result.get("move_done"))
            logger.debug("%s: %s ask-any -> %s", game_id, source, result["announcement"])
            state_after = get_json(f"/api/game/{game_id}/state")
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
                        "possible_actions": state_after.get("possible_actions"),
                        "allowed_moves": state_after.get("allowed_moves"),
                    },
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            )
            conversation = {
                "messages": conversation.get("messages", []),
                "turn_signature": turn_signature(state_after),
                "scoresheet_digest": scoresheet_digest(scoresheet_after),
                "recent_referee_items": recent_after,
            }
            save_conversation_state(game_id, conversation)
            tried_actions = []
            batch_success = True
            break

        if not batch_success:
            refreshed_state = get_json(f"/api/game/{game_id}/state")
            if isinstance(metadata.get("rule_variant"), str):
                refreshed_state["rule_variant"] = metadata["rule_variant"]
            refreshed_scoresheet = refreshed_state.get("scoresheet") if isinstance(refreshed_state.get("scoresheet"), dict) else {}
            refreshed_recent = extract_recent_referee_items(refreshed_scoresheet, limit=10)
            feedback.append("Top-ranked batch failed; provide the next best legal candidates.")
            feedback.append(
                "Current legal options now: "
                + json.dumps(
                    {
                        "possible_actions": refreshed_state.get("possible_actions"),
                        "allowed_moves": refreshed_state.get("allowed_moves"),
                    },
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            )
            conversation = {
                "messages": conversation.get("messages", []),
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
            mine = get_json("/api/game/mine")
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
    parser.add_argument("--poll-seconds", type=float, default=3.0, help="Seconds between /api/game/mine polls.")
    args = parser.parse_args()

    if args.register:
        register_bot()
        return

    if not os.environ.get("KRIEGSPIEL_BOT_TOKEN"):
        raise SystemExit("KRIEGSPIEL_BOT_TOKEN is missing. Run with --register first.")
    if not anthropic_enabled():
        logger.warning("ANTHROPIC_API_KEY is missing; running in fallback mode.")

    run_loop(args.poll_seconds)


if __name__ == "__main__":
    main()
