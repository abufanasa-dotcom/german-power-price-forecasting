# Antigravity Project Memory
## Portfolio Engineering & Data Projects — Working Context

**Purpose:**  
This file gives an AI coding agent the working context, technical history, decisions, results, and workflow used for the user's first two portfolio projects. Read this file before starting Project 3.

The goal is not to copy either previous project. The goal is to preserve the same professional workflow:
- reproducible setup
- careful data validation
- leakage-safe modeling
- clear baselines
- honest evaluation
- useful visualizations
- practical application/demo
- clean Git/GitHub presentation
- step-by-step execution

---

# 1. General Working Style

## Core rule
Work incrementally. Do not change the entire repository at once.

For every major stage:

1. Inspect the current repository and existing files.
2. Explain what already exists.
3. Identify the next logical task.
4. Propose a short implementation plan.
5. Implement one stage.
6. Run the code/tests.
7. Inspect the output.
8. Fix errors before moving on.
9. Save useful outputs/reports.
10. Commit a stable checkpoint before a risky or large next change.

## Important behavior for the coding agent

- Never invent metrics, dataset properties, results, or completed work.
- Never claim a script ran successfully without execution evidence.
- Preserve reproducibility.
- Prefer readable Python over unnecessary abstraction.
- Keep business/scientific reasoning visible in the README.
- Avoid data leakage.
- Keep untouched test/holdout data untouched until the modeling approach is selected.
- Use validation data for model choice and threshold tuning.
- Use test data for the final unbiased evaluation only.
- When modifying code, inspect dependencies between files first.
- Do not delete working files without a clear reason.
- Show diffs or explain meaningful changes.
- Prefer small, testable steps.
- Record important numerical results in report files where appropriate.
- Keep Git history clean enough to demonstrate project development.

---

# 2. Development Environment Used

Main development environment:

- Windows
- VS Code
- PowerShell / VS Code terminal
- Python virtual environments
- Git
- GitHub
- Python data/scientific stack

Common libraries used across the portfolio work included:

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- plotly
- streamlit
- pyarrow
- pytest

The first project used a `.venv` environment.

For future projects:
- create a project-specific virtual environment
- maintain `requirements.txt` or equivalent dependency definition
- keep raw data separate from generated/processed data
- avoid committing large raw datasets when licensing/size makes this inappropriate

---

# 3. Portfolio Project 1 — Wind Turbine Performance Analysis

## Project identity

Common project name:

`wind-performance`

An earlier/local working folder name was also observed as:

`wind-preformence`

Treat `wind-performance` as the clean project/repository name.

## Project objective

Build a realistic engineering/data-science portfolio project around wind turbine SCADA data.

Main goals:

- validate operational SCADA data
- understand turbine performance
- create a baseline expected-power model
- compare machine-learning approaches
- quantify model performance
- identify sustained underperformance events
- place flagged events in operational context
- create a Streamlit demonstration
- publish a polished GitHub repository

The project was intended to demonstrate:
- renewable-energy domain understanding
- Python
- time-series/data validation
- machine learning
- anomaly/underperformance detection
- visualization
- practical engineering reasoning

---

## 3.1 Data source and scope

Dataset source used:

Kelmarsh Wind Farm SCADA data.

Dataset characteristics discussed:
- six Senvion MM92 wind turbines
- 10-minute SCADA data
- broader source data spans multiple years
- project analysis focused on 2022 data for one turbine
- rated turbine power used in the project: approximately 2,050 kW

Main working files:

- `measurements.csv`
- `setpoints.csv`
- `events.csv`

Main period checked:

`2022-01-01 00:00` through `2022-12-31 23:50`

Expected number of 10-minute records for the year:

`52,560`

Time integrity result:
- missing expected timestamps: `0`

Important measurement columns included:
- Wind speed (m/s)
- Wind direction (°)
- Power (kW)
- Turbine Power setpoint (kW)
- Data Availability (0/1)

Numeric parsing:
- key columns had `0` non-numeric values in the validation check

Selected descriptive values:
- mean wind speed: `5.97 m/s`
- mean power: `595.27 kW`
- minimum power: `-17.22 kW`
- maximum power: `2082.69 kW`

---

# 3.2 Initial repository/data structure

The working structure separated source data from generated outputs.

Conceptually:

```text
wind-performance/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
├── src/ or analysis scripts
├── app / Streamlit code
├── README.md
├── requirements.txt
└── .gitignore
```

Exact filenames evolved during development; preserve the principle rather than assuming every filename above already existed.

Raw data should remain unchanged.

Generated flags, model predictions, tables, reports, and plots belong in processed/output/report locations.

---

# 3.3 Data-quality validation

Before modeling, the project explicitly validated the data.

Checks included:

### Timestamp integrity
- start/end timestamps
- expected 10-minute frequency
- missing timestamps
- duplicate timestamps

Result:
- all `52,560` expected timestamps were present
- no missing expected timestamps

### Core measurement completeness

Availability / core-data check:

- `52,006` rows with availability/core completeness
- `554` rows unavailable/incomplete
- missing fraction was approximately `1.05%`

A consistency check was performed between:
- core measurement completeness
- `Data Availability == 1`

This agreement was used to understand whether the availability signal represented usable measurements.

### Power behavior

Negative/non-positive power values were inspected rather than silently discarded.

One analysis counted roughly:
- `6,684` negative-power readings in an early quality check

A later screening workflow also reported:
- remaining non-positive power rows: `5,385`

These numbers came from different filtering stages and should not be conflated.

---

# 3.4 Screening / eligibility logic

An initial screening stage was created to decide which SCADA records were appropriate for performance modeling.

One recorded screening output:

- Pass: `50,099`
- Do not pass: `2,461`

The screening considered signals such as:
- availability
- missing/core measurements
- operating state
- power/setpoint behavior
- other conditions affecting whether a row represented normal performance

Important principle:
**Do not train a normal-performance model on clearly invalid or operationally constrained records.**

---

# 3.5 Setpoint analysis

The turbine power setpoint was investigated to understand curtailment/operating limits.

Example rounded setpoint counts after screening included:

- 2060 kW: `5,090`
- 412 kW: `4,271`
- 1666 kW: `3,136`
- 2059 kW: `609`

Threshold checks included:

- setpoint >= 2000 kW: `8,399` rows
- setpoint >= 2050 kW: `6,776` rows

The purpose was to distinguish unrestricted/high-setpoint operation from constrained behavior.

---

# 3.6 Operational events data

`events.csv` was inspected separately.

Total event rows recorded:

`11,385`

Event-type counts:

- Informational: `10,749`
- Warning: `414`
- Stop: `206`
- Communication: `10`
- Curtailment: `6`

IEC/state labels included examples:

- Full Performance: `6,787`
- Out Env Spec: `2,294`
- Technical Standby: `1,823`
- Forced outage: `89`
- Scheduled: `34`
- Partial: `6`
- Out Electrical: `4`

### Event timestamp quality

Many informational events did not have bounded end timestamps.

Recorded checks:
- missing or unparsed event ends: `9,516`
- event ends earlier than starts: `0`

End-time availability by event type included:
- Informational: 1,233 with an end, 9,516 without
- Stop: 206 with an end
- Warning: 414 with an end
- Curtailment: 6 with an end
- Communication: 10 with an end

This mattered because simple event-window logic could not be applied identically to every event class.

### Overlap comparison

A comparison was made between different ways of flagging event overlap.

Examples:

Stop events:
- bounded events: `206`
- rows flagged using start/end logic: `1,916`
- rows where alternative flags differed: `332`

Curtailment:
- events: `6`
- rows: `289`
- differing flags: `12`

Warning:
- events: `414`
- rows: `4,498`
- differing flags: `258`

This analysis reinforced the need to document exact event-overlap logic.

---

# 3.7 Example curtailment event

One specific operational event was inspected:

`2022-09-25 23:30` → `2022-09-26 04:40`

Event code/context:
- code 108
- partial performance / curtailment-related context

This served as a concrete case for validating how event windows aligned with SCADA measurements.

---

# 3.8 Baseline expected-power model

The project deliberately established a simple baseline before testing more advanced ML.

Recorded dataset split counts:

- Train: `33,928`
- Validation: `8,005`
- Evaluation: `8,003`

Baseline coverage:

`99.98%`

Baseline validation result:
- MAE: `56.21 kW`
- relative to rated power: approximately `2.74%`
- Bias: `+30.77 kW`

Monthly validation examples:

September 2022:
- MAE: `44.18 kW`
- bias: `+20.37 kW`
- evaluated rows: `3,883`

October 2022:
- MAE: `67.55 kW`
- bias: `+40.57 kW`
- evaluated rows: `4,120`

Baseline reports were saved.

Important workflow principle:
**Always keep a transparent baseline. A more complex model should demonstrate measurable value over it.**

---

# 3.9 Machine-learning comparison

Two ML feature approaches were compared with the baseline.

Models/feature sets included:

- baseline
- `ML_speed`
- `ML_speed_direction`

September:
- baseline MAE: `44.18 kW`
- ML_speed MAE: `39.88 kW`
- ML_speed_direction MAE: `40.38 kW`

October:
- baseline MAE: `67.55 kW`
- ML_speed MAE: `61.37 kW`
- ML_speed_direction MAE: `58.97 kW`

The model selection was based on observed validation performance rather than assumed sophistication.

---

# 3.10 Final holdout evaluation

November–December were used as an evaluation/holdout period.

Recorded values:

- eligible holdout rows: `8,166`
- common evaluated rows: `8,137`
- baseline coverage: `99.64%`

Final comparison:

Baseline:
- MAE: `56.17 kW`
- bias: `17.1 kW`

Selected ML:
- MAE: `48.79 kW`
- bias: `13.4 kW`

MAE reduction:

`13.14%`

This was an important portfolio result because it showed measurable improvement while retaining a simple engineering interpretation.

---

# 3.11 Underperformance detection

After establishing expected power, the project used residual/deficit logic to identify sustained underperformance.

Recorded threshold:

`277.47 kW`

Minimum consecutive intervals:

`3`

Because data are 10-minute intervals, the rule prevented isolated single-point deviations from dominating the alert logic.

Holdout data:
- eligible rows: `8,166`
- supported prediction rows: `8,137`

Number of review events:

`5`

---

# 3.12 Top detected underperformance event

Top review event:

Event ID:
`1`

Time:
`2022-11-17 02:10` → `03:00 UTC`

Represented duration:
`60 min`

Mean deficit:
`350.88 kW`

Estimated energy gap:
`350.88 kWh`

Mean wind speed:
`9.05 m/s`

Detailed sample:

| Time | Wind speed m/s | Actual Power kW | ML Expected kW | Setpoint kW |
|---|---:|---:|---:|---:|
| 02:10 | 9.09 | 1346.76 | 1703.55 | 2060.00 |
| 02:20 | 9.46 | 1426.87 | 1793.42 | 2055.46 |
| 02:30 | 9.84 | 1493.90 | 1902.88 | 2052.01 |
| 02:40 | 9.21 | 1440.97 | 1755.94 | 2038.75 |
| 02:50 | 8.46 | 1173.35 | 1534.30 | 2053.72 |
| 03:00 | 8.00 | 1095.55 | 1428.49 | 2060.00 |

Operational context window checked:

`2022-11-17 00:10` → `05:00 UTC`

Result:
- no matching operational records found in that context window

Important interpretation rule:
A detected performance gap is a **review flag**, not automatic proof of a fault.

---

# 3.13 Streamlit app

A Streamlit app was built and successfully launched during development.

Local URL:

`http://localhost:8501`

The app was used to communicate project results visually.

A later runtime issue appeared under Python 3.14:

`ReleaseSemaphore failed`

This should be remembered for future local app development:
- prefer stable, broadly supported Python/package combinations
- avoid changing Python versions late in a working project without need
- pin dependencies when the project becomes stable

---

# 3.14 GitHub presentation work

The first project was prepared for portfolio presentation.

Work included:
- GitHub repository cleanup
- README improvement
- project screenshots/visuals
- repository topics
- pinning the project on the GitHub profile

The project was completed to a state where it was pinned on the GitHub profile.

The README should communicate:
1. problem
2. data
3. methodology
4. validation strategy
5. results
6. engineering interpretation
7. limitations
8. how to run the project

---

# 4. Portfolio Project 2 — Heart Sound Analysis

## Project identity

Local project path:

`C:\heart-sound-analysis`

Project name:

`heart-sound-analysis`

## Objective

Build a second portfolio project in biomedical/audio signal analysis using phonocardiogram recordings.

The project was intended to demonstrate:
- signal processing
- audio/time-frequency analysis
- Python
- feature engineering
- classification
- leakage-safe dataset splitting
- biomedical data reasoning
- model comparison
- reproducible evaluation

---

# 4.1 Dataset

Dataset:

**CirCor DigiScope Phonocardiogram Dataset 1.0.3**

Recorded dataset size:

- `3,163` recordings
- `942` participant records/files

Files included combinations of:
- WAV
- HEA
- TSV / metadata-related files

A manifest was built and checked against all `3,163` recordings.

---

# 4.2 Identity linkage and leakage investigation

This was one of the most important parts of Project 2.

The dataset contained participant/visit relationships that made naive random recording-level splitting unsafe.

Recorded findings:

- `70` Additional-ID pairs
- `872` resulting identity groups
- `15` linked visits where classification labels differed

Therefore the project grouped linked identities together before splitting.

Core rule:

**No recordings belonging to the same person/linked identity group should appear across train, validation, and test sets.**

This prevents identity leakage and unrealistically optimistic metrics.

---

# 4.3 Group-wise split

Final group-safe split:

### Train
- identity groups: `609`
- recordings: `2,228`

### Validation
- identity groups: `133`
- recordings: `475`

### Test
- identity groups: `130`
- recordings: `460`

Identity overlap between splits:

`0`

This is a key methodological strength and should be highlighted in the portfolio README.

---

# 4.4 Test-set discipline

The test set was deliberately kept unopened during model development.

Feature-development/model-selection data:

Train + validation:
`2,703 recordings`

Test:
`460 recordings`

Important rule for Project 3:

**Do not inspect final test performance repeatedly during model selection.**

Use:
- train for fitting
- validation for feature/model decisions
- test once for final evaluation

---

# 4.5 Signal feature extraction

A handcrafted audio-feature pipeline was developed.

Number of features per recording:

`28`

Feature extraction used:

- 2-second windows
- Welch power spectrum
- signal-level information
- frequency/power-distribution information
- temporal variation information
- robust summaries including median and IQR

The development feature table covered:

`2,703` train + validation recordings

The final test data remained protected during this phase.

General design principle:
Use understandable signal features before jumping to high-complexity deep learning, especially when building a portfolio project where explainability and methodology matter.

---

# 4.6 Labels

Classification labels:

- `Absent`
- `Present`
- `Unknown`

Predictions/evaluation were handled at participant level rather than treating every recording as an independent patient.

This was necessary because one participant could have multiple recordings.

---

# 4.7 Participant-level aggregation

Recording-level features/predictions were aggregated to participant-level evaluation.

This prevents participants with more recordings from receiving disproportionate influence and aligns model evaluation with the real classification unit.

For future biomedical/grouped-data work:
- identify the true independent entity
- split by that entity
- report results at that entity level where appropriate

---

# 4.8 Baseline model

Primary baseline model:

**Logistic Regression**

Validation result:

Balanced accuracy:

`61.1%`

Naive comparator:

Always predicting `Absent`

Balanced accuracy:

`33.3%`

The Logistic Regression model therefore showed substantial improvement over the trivial class baseline on validation data.

Detection result noted for `Present`:
- correctly detected `21 / 28` Present participants
- precision approximately `35%`

Interpretation:
recall/sensitivity for Present cases was encouraging, while false positives remained an issue.

Do not present the model as clinically validated.

This is a portfolio/data-analysis model, not a diagnostic medical system.

---

# 4.9 Nonlinear model trial

Model tested:

**SVM with RBF kernel**

Validation:
- balanced accuracy: `56.5%`
- overall accuracy: `65.5%`

The nonlinear SVM did not outperform the simpler Logistic Regression on the selected validation criterion.

Therefore it was not selected merely because it was more complex.

Important principle:

**Prefer the model supported by validation evidence, not the model with the more sophisticated name.**

---

# 4.10 Main methodological lessons from Project 2

1. Investigate identifiers before splitting.
2. Protect against subject/identity leakage.
3. Define the independent evaluation unit.
4. Preserve a final test set.
5. Start with an interpretable baseline.
6. Compare more complex models fairly.
7. Prefer balanced metrics when classes are uneven.
8. Avoid clinical claims unsupported by the experiment.
9. Document data limitations and label ambiguity.
10. Keep signal-processing steps interpretable.

---

# 4.11 GitHub/portfolio status

The second project was developed far enough to be prepared and pinned alongside the first project on the GitHub profile.

The portfolio presentation emphasized methodology rather than inflated claims.

Important README themes:
- problem definition
- CirCor data
- identity linkage
- leakage-safe grouped splitting
- feature extraction
- model comparison
- validation results
- limitations
- reproducibility

---

# 5. Lessons to Carry into Project 3

Project 3 should not be a copy of the previous projects.

It should add a new skill dimension while keeping the same disciplined workflow.

## Phase A — Define the portfolio story

Before coding, define:

- What real problem does the project solve?
- Who would care about the result?
- Which job-relevant skills will it demonstrate?
- What is the simplest defensible success metric?
- What makes the project different from Projects 1 and 2?

Projects already demonstrate:
- renewable-energy SCADA
- anomaly/performance modeling
- audio/biomedical signal analysis
- classical ML
- Streamlit
- leakage-safe evaluation
- engineering reasoning

Project 3 should complement these rather than repeat them without a reason.

---

# 6. Recommended Project 3 Workflow

## Step 1 — Inspect
Before writing code:

- inspect all files in the repository
- identify the current structure
- detect existing data
- read README/instructions
- check Python environment/dependencies
- check Git status

Do not modify files yet.

## Step 2 — Define the problem
Write down:

- objective
- input data
- output
- success metric
- baseline
- train/validation/test logic if ML is involved

## Step 3 — Validate the data
Before modeling:

- shape
- column names/types
- missing values
- duplicates
- invalid values
- time coverage if temporal
- IDs/groups
- leakage risks
- label distribution
- units/ranges

Save a short validation report.

## Step 4 — Build a baseline
Create the simplest meaningful solution first.

Examples:
- persistence/rule baseline
- simple regression
- Logistic Regression
- simple heuristic
- basic deterministic algorithm

Record the result.

## Step 5 — Improve one thing at a time
Possible improvements:
- feature engineering
- additional explanatory variables
- alternative model
- tuning
- threshold selection

Compare every improvement against the baseline using the same validation data.

## Step 6 — Final evaluation
After selecting the approach:
- freeze the methodology
- evaluate once on holdout/test
- report relevant uncertainty/limitations
- avoid retuning against the test result

## Step 7 — Build the demo
Only after the analysis pipeline is stable.

Potential outputs:
- Streamlit dashboard
- interactive visualization
- report
- CLI
- small API

The demo should expose the project's value, not hide weak methodology.

## Step 8 — Portfolio polish
Prepare:
- professional README
- key plots/screenshots
- concise project description
- architecture or workflow diagram when useful
- requirements
- run instructions
- limitations
- GitHub topics
- pinned repository

---

# 7. Git/GitHub Working Rules

Before major edits:

```powershell
git status
```

After a stable stage:
- inspect changes
- run the relevant script/test
- commit the checkpoint

Avoid committing:
- `.venv`
- temporary cache files
- secrets
- large raw datasets unless intentional and allowed
- generated junk

Typical useful files:
- `.gitignore`
- `README.md`
- `requirements.txt`
- source code
- selected small outputs/plots
- screenshots used in README

Prefer descriptive commits such as:

```text
Add data validation pipeline
Add baseline model and validation report
Add leakage-safe group split
Add feature extraction pipeline
Add final evaluation
Add Streamlit dashboard
Improve README and project visuals
```

---

# 8. Communication Protocol for Antigravity

When working with the user, use short sequential tasks.

At the start of a new session, the agent should first respond to this instruction:

> Read `ANTIGRAVITY_PROJECT_MEMORY.md` and inspect the entire current repository. Do not modify anything yet. Tell me:
> 1. what files and data are present,
> 2. what stage the project is currently at,
> 3. what risks or problems you see,
> 4. the next three logical steps.
> Then wait for the next implementation instruction.

For implementation tasks:

> Implement only Step 1. Run it, inspect the output, and explain the result. Do not start Step 2 until Step 1 works.

For debugging:

> Diagnose the error first. Identify the root cause and the smallest safe fix. Do not rewrite unrelated working code.

For model evaluation:

> Compare the new result with the existing baseline using the same split and metric. Do not use the test set for model selection.

For Git:

> Show me `git status` and summarize the changes before committing.

---

# 9. Known Portfolio Results — Quick Reference

## Project 1 — Wind Performance

- 2022 10-minute records: `52,560`
- missing expected timestamps: `0`
- available/core-complete: `52,006`
- unavailable/incomplete: `554`
- baseline validation MAE: `56.21 kW`
- baseline validation bias: `+30.77 kW`
- final holdout common rows: `8,137`
- holdout baseline MAE: `56.17 kW`
- selected ML MAE: `48.79 kW`
- MAE improvement: `13.14%`
- underperformance threshold: `277.47 kW`
- minimum consecutive intervals: `3`
- review events found: `5`
- top event estimated energy gap: `350.88 kWh`

## Project 2 — Heart Sound Analysis

- recordings: `3,163`
- participant files: `942`
- linked identity groups: `872`
- Additional-ID pairs: `70`
- differing linked-visit labels: `15`
- train: `609 groups / 2,228 recordings`
- validation: `133 groups / 475 recordings`
- test: `130 groups / 460 recordings`
- identity overlap: `0`
- handcrafted features: `28`
- Logistic Regression balanced accuracy: `61.1%`
- always-Absent baseline balanced accuracy: `33.3%`
- Present detected: `21/28`
- Present precision: approximately `35%`
- RBF-SVM balanced accuracy: `56.5%`
- RBF-SVM accuracy: `65.5%`

---

# 10. Final Instruction to the Coding Agent

The portfolio standard established by the first two projects is:

**Understand → validate → establish baseline → improve → evaluate honestly → interpret → demonstrate → document → publish.**

Do not skip directly to a complex model or polished UI.

For Project 3, preserve:
- scientific/engineering rigor
- reproducibility
- clear evidence
- realistic claims
- clean implementation
- strong GitHub presentation

Before starting Project 3, inspect the repository and create a project-specific plan consistent with these principles.
