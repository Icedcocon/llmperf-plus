import json
import os
import time
from typing import Any, Dict

import ray
import requests
import re

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote
class VLLMGenerateClient(LLMClient):
    """Client for vLLM Generate API."""

    def find_complete_json(self, text: str) -> tuple[str, int]:
        """Find the first complete JSON object in text.
        
        Returns:
            tuple[str, int]: (json_str, end_pos) if found, else ("", -1)
        """
        try:
            # First try direct JSON parsing
            json.loads(text)
            return text, len(text)
        except:
            pass
        
        # Find opening brace
        start = text.find('{')
        if start == -1:
            return "", -1
        
        stack = []
        in_string = False
        escape = False
        
        for i, c in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            
            if c == '\\':
                escape = True
                continue
            
            if c == '"' and not escape:
                in_string = not in_string
                continue
            
            if not in_string:
                if c == '{':
                    stack.append(c)
                elif c == '}':
                    if not stack:
                        return "", -1
                    stack.pop()
                    if not stack:
                        try:
                            json_str = text[start:i+1]
                            json.loads(json_str) # Validate JSON
                            return json_str, i+1
                        except:
                            continue
                        
        return "", -1

    def process_chunk(self, chunk: str, prompt: str, previous_text: str, start_time: float, most_recent_received_token_time: float) -> tuple[list[dict], str, float, list[float], int]:
        """Process a chunk of response data.
        
        Args:
            chunk: Raw response chunk
            prompt: Original prompt text
            previous_text: Previous accumulated text
            start_time: Request start time
            most_recent_received_token_time: Time of last token
            
        Returns:
            tuple containing:
            - list[dict]: List of parsed JSON objects
            - str: Updated previous_text
            - float: ttft (if first token)
            - list[float]: time_to_next_token measurements
            - int: number of new tokens
        """
        results = []
        chunk_buffer = chunk
        ttft = 0
        time_to_next_token = []
        tokens_received = 0
        
        while chunk_buffer:
            json_str, end_pos = self.find_complete_json(chunk_buffer)
            if not json_str:
                break
            
            try:
                chunk_data = json.loads(json_str)
                if 'text' in chunk_data and chunk_data['text']:
                    current_text = chunk_data['text'][0]
                    
                    # Calculate new tokens by comparing with previous text
                    if not previous_text:
                        new_text = current_text[len(prompt):]
                    else:
                        new_text = current_text[len(previous_text):]
                    
                    if new_text:
                        tokens_received += 1
                        current_time = time.monotonic()
                        
                        # Calculate TTFT for the first token
                        if tokens_received == 1:
                            ttft = current_time - start_time
                        
                        # Calculate time to next token
                        time_to_next = current_time - most_recent_received_token_time
                        time_to_next_token.append(time_to_next)
                        most_recent_received_token_time = current_time
                        
                        previous_text = current_text
                        
                results.append(chunk_data)
                chunk_buffer = chunk_buffer[end_pos:].lstrip()
                
            except Exception as e:
                print(f"[VLLMGenerateClient] JSON parse error: {str(e)}")
                print(f"[VLLMGenerateClient] Problem JSON: {json_str[:200]}")
                chunk_buffer = chunk_buffer[end_pos:].lstrip()
                
        return results, previous_text, ttft, time_to_next_token, tokens_received

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        body = {
            "prompt": prompt,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})
        
        # Truncate prompt to 100 chars in log, keep other fields unchanged
        log_body = body.copy()
        if 'prompt' in log_body:
            log_body['prompt'] = log_body['prompt'][:100] + ('...' if len(log_body['prompt']) > 100 else '')
        print(f"[VLLMGenerateClient] Request started with body: {json.dumps(log_body, ensure_ascii=False)}...")

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        previous_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}
        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()
        
        address = os.environ.get("VLLM_API_BASE")
        if not address:
            raise ValueError("the environment variable VLLM_API_BASE must be set.")
        
        if not address.endswith("/"):
            address = address + "/"
        address += "generate"

        # Add buffer for incomplete chunks
        chunk_buffer = ""
        
        try:
            print(f"[VLLMGenerateClient] Sending request to address: {address}")
            with requests.post(
                address,
                json=body,
                stream=True,
                timeout=180,
                headers={'Connection': 'keep-alive'}
            ) as response:
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    print(f"[VLLMGenerateClient] Request failed - status code: {error_response_code}, error: {error_msg}")
                    response.raise_for_status()

                for chunk in response.iter_lines(chunk_size=None):
                    if not chunk:
                        continue
                    
                    try:
                        chunk_str = chunk.decode('utf-8')
                        print(f"[VLLMGenerateClient] Processing chunk: {chunk_str[:200]}...")
                        
                        results, previous_text, new_ttft, new_times, new_tokens = self.process_chunk(
                            chunk_str, prompt, previous_text, start_time, most_recent_received_token_time
                        )
                        
                        # Update metrics
                        if new_ttft > 0:
                            ttft = new_ttft
                        time_to_next_token.extend(new_times)
                        tokens_received += new_tokens
                        if new_tokens > 0:
                            most_recent_received_token_time = time.monotonic()
                        if previous_text:
                            generated_text = previous_text[len(prompt):]
                        
                    except Exception as e:
                        print(f"[VLLMGenerateClient] Error processing chunk: {str(e)}")
                        continue

                print(f"[VLLMGenerateClient] Request completed. Total tokens received: {tokens_received}")
                print(f"[VLLMGenerateClient] Time to first token: {ttft:.3f}s")
                print(f"[VLLMGenerateClient] Total time to next token measurements: {len(time_to_next_token)}")
                if time_to_next_token:
                    print(f"[VLLMGenerateClient] Average time to next token: {sum(time_to_next_token)/len(time_to_next_token):.3f}s")
                
                # 确保至少有一个token被接收
                if tokens_received == 0:
                    print(f"[VLLMGenerateClient] Warning: No tokens received for request")
                    print(f"[VLLMGenerateClient] Final generated text: '{generated_text}'")
                    metrics[common_metrics.ERROR_MSG] = "No tokens received"
                    metrics[common_metrics.ERROR_CODE] = -2
                    metrics[common_metrics.INTER_TOKEN_LAT] = 0  # 避免除零错误
                else:
                    metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token) / tokens_received
                    print(f"[VLLMGenerateClient] Final inter-token latency: {metrics[common_metrics.INTER_TOKEN_LAT]:.3f}s")

            total_request_time = time.monotonic() - start_time
            if tokens_received > 0:
                output_throughput = tokens_received / total_request_time
            else:
                output_throughput = 0
                print(f"[VLLMGenerateClient] Warning: Cannot calculate throughput, no tokens received")

        except requests.exceptions.ChunkedEncodingError as ce:
            print(f"[VLLMGenerateClient] Connection error: {str(ce)}")
            metrics[common_metrics.ERROR_MSG] = f"Connection error: {str(ce)}"
            metrics[common_metrics.ERROR_CODE] = -3
            metrics[common_metrics.INTER_TOKEN_LAT] = 0  # 确保这个键存在
            metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
            metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len
            metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
            
        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            metrics[common_metrics.INTER_TOKEN_LAT] = 0  # 确保这个键存在
            print(f"[VLLMGenerateClient] Exception details:")
            print(f"  - Type: {type(e).__name__}")
            print(f"  - Message: {str(e)}")
            print(f"  - Error code: {error_response_code}")
            print(f"  - Error message: {error_msg}")

        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config