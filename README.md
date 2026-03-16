# TheraFlow Marketing Automation

Multi-agent marketing system for [TheraFlow](https://theraflow.in) — a SaaS platform for early intervention and autism therapy centers in India.

## Architecture

```
orchestrator.py          ← Master scheduler (runs all agents on cron)
agents/
  scraper.py             ← Google Places lead scraper
  email_gen.py           ← Claude API email sequence generator
  email_sender.py        ← Gmail SMTP sender with tracking
  content_gen.py         ← Claude API LinkedIn post generator
  engagement_monitor.py  ← Gmail reply monitor + intent classifier
  weekly_report.py       ← Sunday analytics report
  linkedin_prep.py       ← Monday LinkedIn DM prep file
data/
  leads.csv              ← Scraped leads
  pain_points.json       ← Extracted pain points
  content_log.json       ← Generated content history
```

## Weekly Schedule

| Day       | Agent                | Action                                      |
|-----------|----------------------|----------------------------------------------|
| Monday    | `scraper.py`         | Scrape therapy centers from Google Places     |
| Monday    | `linkedin_prep.py`   | Generate LinkedIn DM prep file               |
| Tue/Thu   | `email_gen.py`       | Generate personalized email sequences         |
| Tue/Thu   | `email_sender.py`    | Send emails via Gmail SMTP                   |
| Daily     | `engagement_monitor` | Monitor replies, classify intent              |
| Wednesday | `content_gen.py`     | Generate LinkedIn post content               |
| Sunday    | `weekly_report.py`   | Generate weekly analytics report             |

## Setup

### 1. Clone and install

```bash
git clone https://github.com/codehub-sm/theraops.git
cd theraops
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### 3. API keys needed

- **Anthropic API key** — [console.anthropic.com](https://console.anthropic.com/)
- **Google Places API key** — [Google Cloud Console](https://console.cloud.google.com/)
- **Gmail App Password** — [Google Account > Security > App Passwords](https://myaccount.google.com/apppasswords)
- **Gmail API credentials** (for reply monitoring) — Enable Gmail API in Google Cloud Console, download `credentials.json`

### 4. Run

```bash
# Run the orchestrator (handles scheduling)
python orchestrator.py

# Or run individual agents
python -m agents.scraper
python -m agents.email_gen
```

## Data Files

- `data/leads.csv` — Populated by scraper with center name, email, phone, city
- `data/pain_points.json` — Populated by engagement monitor from reply analysis
- `data/content_log.json` — Tracks all generated content with timestamps

## Environment Variables

See `.env.example` for the complete list of required configuration.
