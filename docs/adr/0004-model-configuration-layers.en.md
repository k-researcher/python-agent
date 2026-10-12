# ADR 0004: Two layers of model configuration

## Status

Accepted. Replaces the JSON string `AGENT_LLM_PROFILES` in `.env`.

## Context

The previous configuration kept all model profiles in one JSON line in `.env`. The line mixed
secrets with structure. It was difficult to edit and to review. One syntax error stopped the
startup. All models used one context window. The user also wants a UI to connect providers
and models.

## Decision

The effective configuration has two layers:

1. **Base layer:** `config/models.yaml`. It declares providers, models and routing. It does
   not contain keys. A provider refers to an environment variable through `api_key_env`.
   If the file does not exist, the runtime uses the legacy `AGENT_LLM_*` variables.
2. **UI layer:** overrides in the database, edited through `/api/settings/models`.
   - An override replaces individual fields of a YAML item. Other fields stay from YAML.
   - An override can disable a YAML item. The YAML file does not change.
   - An override can add a provider or a model that is not in YAML.
   - "Reset to file" deletes the override.
   - The UI can keep an API key. The server encrypts it with Fernet. The master key is
     `AGENT_SECRET_KEY`. In the embedded mode, if this variable is not set, the server makes
     `data/secret.key` with mode 0600.

The view for the UI shows the source of each field: file, UI, environment or legacy. The API
never returns a key. It returns only `api_key_set` and the last four characters.

The schema has these transport kinds:

- `openai`: OpenAI-compatible `/chat/completions` (gateways, GLM, LM Studio, MLX, Ollama).
- `openai_responses`: OpenAI Responses API (Codex and GPT reasoning models).
- `anthropic`: Anthropic Messages API (Claude).
- `jev`: TypeSafe decision models.

Each model has its own context window, output limit and permitted reasoning efforts. The
runtime selects the model in this sequence: the explicit model, the model of the role, the
default model.

## Consequences

- API instances and workers read the same overrides from the database. A mounted YAML file
  must be the same on all hosts.
- A provider without a key does not stop the startup. It shows as "not configured".
- An invalid override gives HTTP 422. The server does not save it.
- `agent config check --strict` validates the YAML file without the database.
- The adapters for `anthropic` and `openai_responses` are a later step. Until then, routing
  to these kinds is rejected.
