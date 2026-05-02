# Switchboard

A self-hosted Telegram bot that answers natural-language questions about
your Google Calendar. You run your own instance with your own Google account
and Telegram bot — no shared servers, no third-party access to your data.

**v1 capabilities (read-only):**

- "What's on my calendar today?"
- "When is my next meeting with Alex?"
- "Do I have anything Friday afternoon?"
- Multi-turn follow-ups ("and tomorrow?")

Future versions will add event creation, updates, deletion, and proactive
reminders, each gated by an explicit Telegram approval.

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Google Cloud project with the Calendar API enabled
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- Your numeric Telegram user ID (message [@userinfobot](https://t.me/userinfobot))

## Setup

### 1. Install dependencies

```bash
git clone https://github.com/kvpratama/switchboard.git
cd switchboard
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in TELEGRAM_BOT_TOKEN, ALLOWED_TELEGRAM_USER_IDS,
# LLM_PROVIDER (openai/anthropic/etc), LLM_MODEL, LLM_API_KEY,
# GOOGLE_OAUTH_CLIENT_SECRETS_PATH, DEFAULT_TIMEZONE.
```

### 3. Create a Google OAuth client

1. Go to https://console.cloud.google.com/ → create a project.
2. APIs & Services → Library → enable **Google Calendar API**.
3. APIs & Services → **OAuth consent screen** → External → fill in app name/email → scroll down to **Test users** → **+ ADD USERS** → enter the Gmail address you'll sign in with → **SAVE**.

   > ⚠️ Wait ~30 seconds after saving, then sign in with the **exact same Gmail address** you added. Using a different account causes a 403 `access_denied` error.
4. APIs & Services → Credentials → Create Credentials → **OAuth client ID** →
   choose **Desktop app**.
5. Download the JSON file. Save it where `GOOGLE_OAUTH_CLIENT_SECRETS_PATH`
   in your `.env` points (default: `./credentials.json`).

### 4. Run the one-time auth bootstrap

```bash
uv run python -m src.auth.bootstrap
```

A browser opens. Sign in with the Gmail address you added as a test user. You'll see a **"Google hasn't verified this app"** warning — click **Advanced → Go to Switchboard (unsafe) → Continue** and grant read-only calendar access. The token is saved to `data/token.json`.

**Headless server?** Run the bootstrap on your laptop, then copy
`data/token.json` to the same path on the server. Alternatively, SSH-tunnel
the OAuth port:

```bash
ssh -L 8080:localhost:8080 user@your-server
```

Run the bootstrap on the server while the tunnel is open.

### 5. Start the bot

```bash
uv run python main.py
```

Message the bot on Telegram. Try: *"What's on my calendar today?"*

## Verification

### Smoke test
Message your bot on Telegram:
- `what's on my calendar today?` — should return today's events
- `do I have anything tomorrow?` — validates date resolution
- `find my next meeting with <name>` — tests search

### Memory test
1. Ask: `what events do I have on Friday?`
2. Follow up: `what's the location of the first one?`

The bot should remember Friday's events. If not, check that `data/checkpoints.sqlite` exists.

### Persistence test
1. Have a conversation, then stop the bot (Ctrl+C)
2. Restart: `uv run python main.py`
3. Send a follow-up — it should remember the prior session

## Optional: LangSmith tracing

Add to `.env` to enable detailed agent traces in
[LangSmith](https://smith.langchain.com/):

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=switchboard
```

## Development

```bash
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run ty check         # type check
```

Tests are co-located with their target module (`<module>_test.py`).

## Roadmap

| Version | Adds                                                                |
| ------- | ------------------------------------------------------------------- |
| **v2**  | Create events; Telegram approval prompt before any mutation         |
| **v3**  | Update / delete events                                              |
| **v4**  | Proactive reminders                                                 |

## License

See `LICENSE`.
