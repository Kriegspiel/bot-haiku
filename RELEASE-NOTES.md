# Release Notes

These notes summarize the bot runtime release history reconstructed from the
current repository state. Add a new section at the top for runtime,
deployment-facing, or user-visible bot behavior changes. Test-only and
docs-only changes do not need entries unless they affect operator workflow.

## 1.0.3 — Xhigh Claude Reasoning

- Set active Sonnet, Opus, and Fable templates to xhigh adaptive effort. Keep Haiku manual thinking, token caps, timeouts, and the shared monthly budget unchanged.

## 1.0.0 — Claude Sonnet and Opus Refresh

- Upgrade the existing Sonnet and Opus instance templates to 5.5 and use stable
  display names while retaining bot identities and history.
- Send adaptive thinking, explicit low effort, automatic strict tools, and a
  structured JSON text fallback for 5.5; preserve Haiku's request behavior.
- Set bounded output budgets that include thinking and current token/cache
  prices while retaining the shared monthly Anthropic cap.
- Start explicit runtime versioning in `VERSION`; earlier releases were
  identified by git commit.

## 1.0.4 — Claude Haiku 5.5

- Upgrade the base worker to `claude-haiku-5-5` with adaptive thinking at `xhigh`, the level below `max`, automatic strict action tools and JSON fallback.
- Estimate short-prompt input/output at $0.10/$0.50 per million tokens; apply the fivefold tier above 100,000 total prompt tokens, including cached tokens. Reserve conservatively at the higher tier within the unchanged shared $18 monthly cap.
- Keep the existing 32,768-token output bound and 300-second timeout.

## Shared Anthropic Monthly Cap

- **Provider Cap**: enforce a shared `$18` cap per UTC calendar month across
  Haiku, Sonnet, Opus, and every other Anthropic bot process on the host.
- **Strict Accounting**: reserve a conservative upper-bound cost before each
  request, settle from returned token and cache usage, and consume the full
  reservation when usage cannot be trusted.
- **Fallback**: stop paid Anthropic calls, report the provider unavailable, and
  continue active games with legal local fallback moves when the cap is spent.

## Non-Billable Anthropic Availability

- **Idle Cost Fix**: replace the periodic Messages API `Ping` generation with
  authenticated `GET /v1/models/{model}` metadata checks.
- **Availability**: keep the existing readiness cache and backend reporting
  behavior without consuming input or output tokens while idle.

## Tiered Bot Join Budgets

- **Lobby Policy**: derive bot-vs-bot join probability from
  `KRIEGSPIEL_LLM_BOT_TIER` by default: T2 `0.0010`, T3 `0.0005`, T4
  `0.0002`, and T5 `0.0001`.
- **Operator Override**: allow `KRIEGSPIEL_BOT_GAME_PICK_PROBABILITY` or
  `BOT_GAME_PICK_PROBABILITY` to override the tier default per instance.

## T4 Opus Model ID

- **T4 Catalogue**: use the current Anthropic API model ID
  `claude-opus-4-8` for the Opus 4.8 instance template.

## T4 Opus Instance Template

- **T4 Catalogue**: add an Anthropic Opus 4.8 instance template for
  `llm_opus48`.
- **Bot-vs-bot Caps**: honor the backend's current `llm_bot_turn_limit` field
  before falling back to the legacy `llm_bot_ply_limit` field.

## Current Runtime Baseline

- **Bot Identity**: `llm_haiku`, the Anthropic Haiku model bot.
- **Rulesets**: supports `berkeley`, `berkeley_any`, `cincinnati`, `wild16`,
  `rand`, `english`, and `crazykrieg`, with legacy two-ruleset configs expanded
  to the full supported set.
- **Runtime Shape**: runs one process per bot identity or model instance, with
  one lightweight runner thread per active game and a configurable shared model
  call cap that defaults to 5 concurrent calls.
- **Lobby Policy**: does not create human lobby games by default, can join a
  compatible bot-created waiting game using its tier probability on a
  ten-minute scan, and checks Anthropic availability before joining new
  bot-vs-bot games.
- **Move Policy**: builds compact stateless prompts with a stable cacheable
  system-prompt strategy reference, asks Anthropic for ranked strict JSON
  candidate actions, and validates them before playing.
