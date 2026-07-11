# Release Notes

These notes summarize the bot runtime release history reconstructed from the
current repository state. Add a new section at the top for runtime,
deployment-facing, or user-visible bot behavior changes. Test-only and
docs-only changes do not need entries unless they affect operator workflow.

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
  compatible bot-created waiting game with 1% probability on a ten-minute scan,
  and checks Anthropic availability before joining new bot-vs-bot games.
- **Move Policy**: builds compact stateless prompts with a stable cacheable
  system-prompt strategy reference, asks Anthropic for ranked strict JSON
  candidate actions, and validates them before playing.
