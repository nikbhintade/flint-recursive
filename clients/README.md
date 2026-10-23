# Coding-agent clients

Start the server first (`./run-flint.sh`, default 64K context on port 8080), then:

```bash
# OpenCode
OPENCODE_CONFIG="$PWD/clients/opencode.json" opencode

# pi
mkdir -p .pi-flint && cp clients/pi-models.json .pi-flint/models.json
PI_CODING_AGENT_DIR="$PWD/.pi-flint" pi --provider flint-local --model flint
```

These are the configs that ship with the model, with the context window set to 65,536 tokens and the output limit
to 16,384, to match `run-flint.sh`. If you change `CTX`, change `context` / `contextWindow` here too.
