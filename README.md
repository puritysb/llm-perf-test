# LLM Performance Test

This project is designed to perform performance tests on Language Models (LLMs).

## Project Setup and Execution

1.  **Install Dependencies:**
    Ensure you have Python installed. Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This project may have some Node.js related files, but Node.js specific settings are not required for the core performance testing.)*

2.  **Run Performance Test:**
    Execute the main test script:
    ```bash
    python test_performance.py
    ```

## Performance Test Log

Performance test results are logged to `performance_test.log`. This file contains details about each test run, including response times and memory usage.

## Project Structure

*   `.gitignore`: Specifies intentionally untracked files that Git should ignore (e.g., log files, generated CSVs, `node_modules`).
*   `requirements.txt`: Lists the Python dependencies required for the project.
*   `test_performance.py`: The main script for running performance tests.
*   `cline_docs/`: (If applicable) Additional documentation related to the project or Cline's interaction.
*   `logs/`: May contain performance test log files.

## Benchmark Results

Performance test results for local LLM configurations using mlx-omni-server and Ollama.

### mlx-community/Qwen3-30B-A3B-8bit (via mlx-omni-server at localhost:8082)

*   **Single-turn Test:**
    *   Total Inference Time: 63.22s
*   **Multi-turn Test:**
    *   Turn 1 Inference Time: 10.21s
    *   Turn 2 Inference Time: 9.01s
*   **Overall Averages (Successful Tests):**
    *   Average Inference Time: 27.48s
    *   Average Token Generation Speed: 43.78 tokens/s

### qwen3:30b-a3b-q8_0 (via Ollama at localhost:11434)

*   **Single-turn Test:**
    *   Total Inference Time: 120.67s
*   **Multi-turn Test:**
    *   Turn 1 Inference Time: 7.65s
    *   Turn 2 Inference Time: 8.43s
*   **Overall Averages (Successful Tests):**
    *   Average Inference Time: 45.58s
    *   Average Token Generation Speed: 37.57 tokens/s
