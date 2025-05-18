# System Patterns

**System Architecture:** The project appears to be a command-line tool or script-based system for running performance tests.

**Key Technical Decisions:**
- Confirmed Python as the primary language for the performance testing logic.
- Input/output involves CSV and log files.
- Node.js related files (`package.json`, `node_modules`, `pnpm-lock.yaml`) are present in the project directory but are excluded from the Git repository as per user's decision.

**Design Patterns in Use:** (To be determined as the project is explored further)

**Component Relationships:**
- `test_performance.py`: The main script for running tests.
- LLM Frameworks/Models: External components being tested.
- Input Data: Data used for testing (e.g., sample data files, potentially generated data).
- Output Files: Results of the tests (`performance_test.log`, generated CSV files).
- Excluded Components: `node_modules` and `pnpm-lock.yaml` are present locally but not part of the version-controlled project due to exclusion.

**Critical Implementation Paths:**
- Loading and processing input data.
- Interacting with different LLM frameworks and models.
- Running performance tests and collecting metrics.
- Generating output reports in specified formats.
