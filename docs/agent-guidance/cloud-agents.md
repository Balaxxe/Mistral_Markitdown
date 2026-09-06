# Cloud agents

- The VM is defined by `.cursor/environment.json` (`install` sets up poppler, ghostscript, and Python deps; Cursor snapshots the VM after it succeeds).
- `MISTRAL_API_KEY` comes from the Cursor dashboard Secrets tab as an environment variable. Never expect a committed `.env` in cloud runs — and tests don't need the key.
- Project hooks run in cloud agents (command hooks only). User-level hooks do not exist there.
