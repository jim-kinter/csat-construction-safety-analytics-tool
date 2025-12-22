# CSAT – Construction Safety Analytics Tool  ([REPO HERE](https://github.com/jim-kinter/csat-construction-safety-analytics-tool))

![CSAT Analytics Dashboard](/staticfiles/Dashboard.png)

> **Every year, over 1,000 construction workers die in the United States.**  
> Most of those deaths were **preventable**.

CSAT is not just another dashboard.  
It is a **real-time safety intelligence platform** that transforms raw, public OSHA incident data into **actionable, predictive, and prescriptive insights** — giving safety leaders the power to **stop the next fatal incident before it happens**.

---

## The Problem

Construction remains one of the deadliest industries in America.  
Despite decades of regulation, training, and PPE, the fatality rate has **barely moved** in 20 years.

Why?  
Because most safety decisions are still **reactive** — made **after** someone gets hurt.

We can do better.

---

### The Solution: CSAT

CSAT is a full-stack Django web application that delivers **all four levels of analytics** in one unified tool:

| Level            | Question Answered               | CSAT Feature                                                   |
|------------------|---------------------------------|----------------------------------------------------------------|
| **Descriptive**  | What happened?                  | Interactive timeline of every incident                         |
| **Diagnostic**   | Why did it happen?              | Weather × Severity heatmaps                                    |
| **Predictive**   | What will happen next?          | 12-month risk forecast by equipment group                      |
| **Prescriptive** | What should we do to stop it?   | **AI-generated**, specific mitigation recommendations (GPT-4o) |

No generic posters.  
No “wear your PPE” slogans.  
**Real, engineering-first, OSHA-grade recommendations** — generated on demand from actual incident patterns.

---

### How It Works (Narrative Explanation)

CSAT is designed for construction safety managers, site supervisors, and risk officers who need more than static reports — they need a tool that **learns from the past** to **protect the future**.

#### Step 1: Plan Your Work

- Enter your daily activities and equipment (e.g., “Lift vessel – crane”)
- CSAT instantly analyzes against 1.6M+ OSHA incidents + 50k simulated records
- Get real-time risk scores for each step

#### Step 2: Get AI-Powered Insights

- **Predictive Model** (Random Forest): Calculates severity probability (Low/Medium/High/Fatal)
- **AI Engine** (GPT-4o): Generates **specific, actionable mitigations** based on equipment, weather, and historical patterns
  - Example: For "crane" in "windy" weather: “Install anemometer with auto-shutdown at 20 mph · Require two spotters · Use anti-two-block + load moment indicator”

#### Step 3: Forecast & Prevent

- 12-month trend forecast using Holt-Winters (smoothed lines, no negatives)
- Grouped by equipment category (e.g., Mechanical, Electrical) — see rising risks
- Dynamic "Other" legend shows top 5 uncategorized items

#### Step 4: Drill Down & Export

- View similar historical incidents (with details)
- Save/load plans
- Export PDF: Branded report with plan, risks, mitigations, and forecasts

#### Relevance

In construction, **one mistake = one life**. CSAT bridges the gap between data overload and actionable safety — reducing fatalities by making **proactive prevention** simple and intuitive. It’s not just analytics; it’s **lives saved**.

---

#### Data Sources & Limitations

| Dataset                          | Years     | Source                     | Contains Fatalities?  | Notes                                                                                      |
|----------------------------------|-----------|----------------------------|-----------------------|-------------------------------------------------------------------------------------------|
| OSHA Injury Tracking (ITA)       | 2016–2021 | osha.gov                   | No                    | Aggregate counts only                                                                                       |
| OSHA Severe Injury Reports (SIR) | 2015–2018 | osha.gov                   | Yes                   | Individual fatal events                                                                                     |
| Combined                         | 2015–2021 | Merged in this project     | Yes (2015–2018 only)  | **Known limitation**: No public individual fatalities after 2018 due to OSHA policy change                                                                                     |

**Important Reality**: You’ll notice all fatalities stop around 2018.  
This is **not a bug** — it’s a **regulatory change**. OSHA stopped releasing individual fatal incidents publicly after 2018.  
CSAT correctly reflects this — and highlights a real-world challenge in safety data transparency.

---

#### Technical Architecture

- **Backend**: Django 5.1.2 (Python 3.12)  
- **Database**: MySQL 8 (Docker)  
- **Frontend**: Plotly.js + responsive CSS  
- **ML**: Scikit-learn (Random Forest) + Statsmodels (Holt-Winters forecasting)  
- **AI**: OpenAI GPT-4o-mini (dynamic mitigation recommendations)  
- **PDF**: WeasyPrint (branded reports)  
- **Deployment**: Docker Compose (fully containerized)

---

#### Why This Matters

In construction, **one bad day** can cost a life, a family, and a company.

CSAT doesn’t just show you the past.  
It **shows you the future** — and **tells you exactly how to change it**.

This isn’t theoretical.  
This is **preventable tragedy, stopped**.

---

#### System Requirements (works on any machine)

- **Windows**: Docker Desktop (latest) → <https://www.docker.com/products/docker-desktop/>
- **macOS**: Docker Desktop (latest) → same link above
- **Linux**: Docker Engine + docker-compose plugin (already installed on most distros)

#### Try It Now (you'll need to go to <https://platform.openai.com> to get an API key and then paste that API key in the script below)

```bash
# Clone and enter
git clone https://github.com/jim-kinter/csat-construction-safety-analytics-tool.git
cd csat-construction-safety-analytics-tool

# Create .env with your OpenAI key - replace the ... with your own OpenAI Key then run it from the bash shell in the project root
echo "OPENAI_API_KEY=sk-..." > .env

# Build and run
docker compose up -d --build
# Load data (first time only)
docker compose exec web python manage.py migrate
docker compose exec web python csat_app/load_data.py
docker compose exec web python csat_app/predict_risk.py
docker compose exec web python manage.py createsuperuser
# Rebuild and run
docker compose down && docker compose build --no-cache && docker compose up -d

# Open a browser and hit http://localhost:8000 or http://localhost:8000/admin 
