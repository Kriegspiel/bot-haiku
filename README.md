# bot-haiku

Kriegspiel bot that asks an Anthropic Haiku model to choose the next action from the bot's private game state.

## What it does

- registers as a listed Kriegspiel bot
- polls assigned games from the live API
- does not create waiting lobby games by default
- can join another bot's waiting lobby game with 0.1% probability while still under its active-game cap
- builds a prompt from the current rule variant, private FEN, legal actions, and private scoresheet
- asks an Anthropic Haiku model for the top ranked next actions in strict JSON
- validates the model output against the server-provided legal actions
- falls back safely if the model response is invalid or the API call fails

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python bot.py --register
python bot.py
```

The bot reads the live rules text from the sibling `content` repository:

- `content/rules/berkeley.md`
- `content/rules/README.md`

By default the registration email is `bot-haiku@kriegspiel.org`.

By default the bot does not create open lobby games on its own. That behavior is controlled with:

- `KRIEGSPIEL_AUTO_CREATE_LOBBY_GAME=true|false`
- `KRIEGSPIEL_AUTO_CREATE_RULE_VARIANT=berkeley|berkeley_any`
- `KRIEGSPIEL_AUTO_CREATE_PLAY_AS=white|black|random`
- `KRIEGSPIEL_SUPPORTED_RULE_VARIANTS=berkeley,berkeley_any`
- `KRIEGSPIEL_MAX_ACTIVE_GAMES_BEFORE_CREATE=1`

Bot-vs-bot play is also enabled by default:

- the bot checks open waiting games
- it will only consider games created by another bot
- it samples that decision at most once per minute
- it will try to join one with 0.1% probability on that minute check
- it uses the same 1-active-game cap for intentional bot-vs-bot joins
- it keeps a local one-minute cooldown between bot-vs-bot join attempts to match backend rules

Anthropic prompting defaults:

- system prompt carries the rules and overall Kriegspiel scene
- the first user prompt carries private scoresheet history, recent referee items, and legal actions
- later user prompts carry only incremental updates: new referee items, rejected candidates, and refreshed legal options
- the bot keeps per-game conversation history and reuses it on the Anthropic Messages API
- automatic prompt caching is enabled with a 1-hour TTL to maximize cache reuse across retries and later turns
- the bot asks for the top 10 ranked candidate actions by default
- if a batch fails, it asks the model for the next batch of candidates
- defaults can be tuned with:
  - `ANTHROPIC_MODEL=claude-haiku-4-5-20251001`
  - `ANTHROPIC_MODEL_BATCH_SIZE=10`
  - `ANTHROPIC_MAX_BATCHES_PER_TURN=5`

## Test

```bash
python -m unittest discover -s tests
```

## systemd

A production host can run the bot as a service with `deploy/kriegspiel-haiku-bot.service`.
