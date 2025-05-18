import requests
import psutil
import time
import logging
from datetime import datetime
import json
import subprocess
import tempfile
import os
import re

# Configuration
MODEL_CONFIGS = [
    {"api_url": "http://localhost:8082/v1/chat/completions", "model_name": "mlx-community/Qwen3-30B-A3B-8bit"},
    {"api_url": "http://localhost:11434/v1/chat/completions", "model_name": "qwen3:30b-a3b-q8_0"}
]

SINGLE_TURN_PROMPT = "Python의 pandas 라이브러리를 사용하여 CSV 파일을 만들고, 데이터프레임의 5행을 출력하는 코드를 작성해주세요. 코드는 반드시 ```python ``` 블록으로 감싸주세요."
MULTI_TURN_PROMPTS = [
    "이 코드에 데이터프레임의 열 이름을 'num', 'nickname'으로 설정해주세요. 수정된 전체 코드를 ```python ``` 블록으로 다시 제공해주세요.",
    "생성된 데이터프레임을 'output.csv' 파일로 저장하는 코드를 추가해주세요. 수정된 전체 코드를 ```python ``` 블록으로 다시 제공해주세요."
]

# Setup logging
logging.basicConfig(filename='performance_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_memory_usage_mb():
    """Returns the current RSS memory usage of the current process in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def get_system_memory_percent():
    """Returns the current system memory usage as a percentage."""
    return psutil.virtual_memory().percent

def get_system_memory_gb():
    """Returns the current system memory usage in GB."""
    return psutil.virtual_memory().used / (1024 ** 3)

def calculate_token_speed(response_json, response_time):
    """Calculates tokens per second from the API response and response time."""
    try:
        # Assuming the API response includes 'usage' with 'completion_tokens'
        token_count = response_json.get('usage', {}).get('completion_tokens')
        if token_count is not None and response_time > 0:
            return token_count / response_time
        return 0
    except Exception as e:
        logging.error(f"Error calculating token speed: {e}")
        return 0

def extract_and_execute_code(code_content, test_identifier):
    """
    Extracts Python code from a string, saves it to a file, and executes it.
    Args:
        code_content (str): The string containing potential code blocks.
        test_identifier (str): A string to identify the test/turn for filename.
    Returns:
        tuple: (execution_success, execution_output, execution_error)
    """
    # Use regex to find Python code blocks
    code_blocks = re.findall(r'```python\n(.*?)```', code_content, re.DOTALL)

    if not code_blocks:
        logging.info(f"No Python code block found for {test_identifier}.")
        return False, "No Python code block found.", ""

    # Execute the first code block found
    code_to_execute = code_blocks[0].strip()
    logging.info(f"Extracted code for execution ({test_identifier}):\n```python\n{code_to_execute}\n```")

    # Define the directory to save generated code
    generated_code_dir = "generated_code"
    os.makedirs(generated_code_dir, exist_ok=True)

    # Define the filename
    code_filename = os.path.join(generated_code_dir, f"{test_identifier}.py")

    # Save the extracted code to a file
    try:
        with open(code_filename, "w", encoding="utf-8") as f:
            f.write(code_to_execute)
        logging.info(f"Generated code saved to {code_filename}")
    except IOError as e:
        logging.error(f"Error saving generated code to {code_filename}: {e}")
        # Continue with execution even if saving fails

    # Write code to a temporary file for execution
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp_file:
        tmp_file_name = tmp_file.name
        tmp_file.write(code_to_execute)
        tmp_file.flush()

    execution_success = False
    execution_output = ""
    execution_error = ""

    try:
        # Execute the temporary file
        result = subprocess.run(['python', tmp_file_name], capture_output=True, text=True, timeout=10)
        execution_output = result.stdout.strip()
        execution_error = result.stderr.strip()
        if result.returncode == 0:
            execution_success = True
    except FileNotFoundError:
        execution_error = "Python executable not found."
    except subprocess.TimeoutExpired:
        execution_error = "Code execution timed out."
    except Exception as e:
        execution_error = f"An error occurred during code execution: {e}"
    finally:
        # Clean up the temporary file
        os.remove(tmp_file_name)

    return execution_success, execution_output, execution_error


def run_inference(messages, api_url, model_name, max_tokens=5000):
    """Runs inference with the given messages and returns response, time, and memory change."""
    memory_percent_before = get_system_memory_percent()
    memory_gb_before = get_system_memory_gb()
    start_time = time.time()

    request_payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens
    }
    logging.info(f"Sending request payload to {api_url} for model {model_name}: {json.dumps(request_payload, indent=2)}")

    try:
        response = requests.post(api_url, json=request_payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for {model_name} at {api_url}: {e}")
        return None, 0, 0, 0, 0, 0, 0 # Added extra 0 for the new return values

    end_time = time.time()
    memory_percent_after = get_system_memory_percent()
    memory_gb_after = get_system_memory_gb()
    response_time = end_time - start_time
    token_speed = calculate_token_speed(response_json, response_time)

    # Note: Measuring first token generation time requires streaming support from the API.
    # The current implementation uses a blocking request (requests.post), so first token time
    # cannot be accurately measured without modifying this to handle streaming.

    return response_json, response_time, memory_percent_before, memory_percent_after, memory_gb_before, memory_gb_after, token_speed

def run_tests_for_model(api_url, model_name):
    logging.info(f"\n=== Testing Model: {model_name} at {api_url} ===")
    test_results = []

    # Single-turn test
    logging.info("\n--- Single-turn Test ---")
    single_turn_messages = [{"role": "user", "content": SINGLE_TURN_PROMPT}]
    response, total_inference_time, mem_before_percent, mem_after_percent, mem_before_gb, mem_after_gb, token_speed = run_inference(single_turn_messages, api_url, model_name)

    single_turn_result = {
        "model": model_name,
        "test_type": "Single-turn",
        "prompt": SINGLE_TURN_PROMPT,
        "success": response is not None,
        "total_inference_time": total_inference_time,
        "mem_before_percent": mem_before_percent,
        "mem_after_percent": mem_after_percent,
        "mem_before_gb": mem_before_gb,
        "mem_after_gb": mem_after_gb,
        "token_speed": token_speed,
        "response_content": "",
        "code_execution_success": None,
        "code_execution_output": "",
        "code_execution_error": ""
    }

    if response:
        response_content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        single_turn_result["response_content"] = response_content
        # Log full response content for debugging
        logging.info(f"Full Response Content (Single-turn):\n{response_content}")

        # Attempt to extract and execute code, passing identifier
        execution_success, execution_output, execution_error = extract_and_execute_code(response_content, f"{model_name.replace('/', '_').replace(':', '_')}_single_turn")
        single_turn_result["code_execution_success"] = execution_success
        single_turn_result["code_execution_output"] = execution_output
        single_turn_result["code_execution_error"] = execution_error
        logging.info(f"Code Execution Success: {execution_success}")
        if execution_output:
            logging.info(f"Code Execution Output:\n{execution_output}")
        if execution_error:
            logging.error(f"Code Execution Error:\n{execution_error}")

    test_results.append(single_turn_result)


    # Multi-turn test
    logging.info("\n--- Multi-turn Test ---")
    multi_turn_messages = []
    multi_turn_results = []

    # Add the initial single-turn prompt to the multi-turn conversation history
    # Note: The multi_turn_messages list is built incrementally and passed to run_inference
    # for each turn, maintaining the conversation history.

    for i, turn_prompt in enumerate(MULTI_TURN_PROMPTS):
        logging.info(f"\n- Turn {i+1} -")
        
        # Add the user's message for the current turn to the messages list for inference
        multi_turn_messages.append({"role": "user", "content": turn_prompt})

        response, total_inference_time, mem_before_percent, mem_after_percent, mem_before_gb, mem_after_gb, token_speed = run_inference(multi_turn_messages, api_url, model_name)

        turn_result = {
            "model": model_name,
            "test_type": f"Multi-turn Turn {i+1}",
            "prompt": turn_prompt,
            "success": response is not None,
            "total_inference_time": total_inference_time,
            "mem_before_percent": mem_before_percent,
            "mem_after_percent": mem_after_percent,
            "mem_before_gb": mem_before_gb,
            "mem_after_gb": mem_after_gb,
            "token_speed": token_speed,
            "response_content": "",
            "code_execution_success": None,
            "code_execution_output": "",
            "code_execution_error": ""
        }

        if response:
            response_content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            turn_result["response_content"] = response_content
            # Log full response content for debugging
            logging.info(f"Full Response Content (Multi-turn Turn {i+1}):\n{response_content}")
            
            # Add the model's response to the messages list for the next turn
            if response_content:
                multi_turn_messages.append({"role": "assistant", "content": response_content})

            # Attempt to extract and execute code, passing identifier
            execution_success, execution_output, execution_error = extract_and_execute_code(response_content, f"{model_name.replace('/', '_').replace(':', '_')}_multi_turn_turn_{i+1}")
            turn_result["code_execution_success"] = execution_success
            turn_result["code_execution_output"] = execution_output
            turn_result["code_execution_error"] = execution_error
            logging.info(f"Code Execution Success: {execution_success}")
            if execution_output:
                logging.info(f"Code Execution Output:\n{execution_output}")
            if execution_error:
                logging.error(f"Code Execution Error:\n{execution_error}")

        else:
            logging.error(f"Multi-turn test failed at turn {i+1} for model {model_name}. Aborting multi-turn test for this model.")
            turn_result["success"] = False
            multi_turn_results.append(turn_result)
            break # Stop if a turn fails

        multi_turn_results.append(turn_result)

    test_results.extend(multi_turn_results)
    return test_results


def summarize_results(all_results):
    summary = {}
    for result in all_results:
        model_name = result['model']
        if model_name not in summary:
            summary[model_name] = {
                "total_tests": 0,
                "successful_tests": 0,
                "total_inference_time": 0,
                "total_token_speed": 0,
                "per_step_times": []
            }
        
        summary[model_name]["total_tests"] += 1
        if result['success']:
            summary[model_name]["successful_tests"] += 1
            summary[model_name]["total_inference_time"] += result['total_inference_time']
            summary[model_name]["total_token_speed"] += result['token_speed']
            summary[model_name]["per_step_times"].append({
                "test_type": result['test_type'],
                "time": result['total_inference_time']
            })
        else:
             summary[model_name]["per_step_times"].append({
                "test_type": result['test_type'],
                "time": "N/A"
            })


    logging.info(f"\n=== PERFORMANCE TEST RUN SUMMARY: {datetime.now()} ===")
    print(f"\n=== PERFORMANCE TEST RUN SUMMARY: {datetime.now()} ===")

    for model_name, data in summary.items():
        logging.info(f"Tested Model: {model_name}")
        print(f"Tested Model: {model_name}")
        logging.info(f"Total Tests Run: {data['total_tests']}")
        print(f"Total Tests Run: {data['total_tests']}")
        logging.info(f"Successful Tests: {data['successful_tests']}")
        print(f"Successful Tests: {data['successful_tests']}")

        if data['successful_tests'] > 0:
            avg_inference_time = data['total_inference_time'] / data['successful_tests']
            avg_token_speed = data['total_token_speed'] / data['successful_tests']
            logging.info(f"Average Inference Time (Successful Tests): {avg_inference_time:.2f}s")
            print(f"Average Inference Time (Successful Tests): {avg_inference_time:.2f}s")
            logging.info(f"Average Token Generation Speed (Successful Tests): {avg_token_speed:.2f} tokens/s")
            print(f"Average Token Generation Speed (Successful Tests): {avg_token_speed:.2f} tokens/s")
        else:
            logging.info("Average Inference Time (Successful Tests): N/A")
            print("Average Inference Time (Successful Tests): N/A")
            logging.info("Average Token Generation Speed (Successful Tests): N/A")
            print("Average Token Generation Speed (Successful Tests): N/A")

        logging.info("\n--- Per-Step Times ---")
        print("\n--- Per-Step Times ---")
        for step_time in data['per_step_times']:
            if step_time['time'] != "N/A":
                 logging.info(f"{step_time['test_type']}: Total Inference Time: {step_time['time']:.2f}s")
                 print(f"{step_time['test_type']}: Total Inference Time: {step_time['time']:.2f}s")
            else:
                 logging.info(f"{step_time['test_type']}: Test Failed - Time N/A")
                 print(f"{step_time['test_type']}: Test Failed - Time N/A")


    logging.info(f"=== PERFORMANCE TEST RUN SUMMARY END ===")
    print(f"=== PERFORMANCE TEST RUN SUMMARY END ===")


if __name__ == "__main__":
    all_test_results = []
    logging.info(f"=== PERFORMANCE TEST RUN START: {datetime.now()} ===")
    for config in MODEL_CONFIGS:
        results = run_tests_for_model(config["api_url"], config["model_name"])
        all_test_results.extend(results)

    summarize_results(all_test_results)
