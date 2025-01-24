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

        try:
            print(f"[VLLMGenerateClient] Sending request to address: {address}")
            with requests.post(
                address,
                json=body,
                stream=True,
                timeout=180,
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
                        # Process each chunk
                        chunk_str = chunk.decode('utf-8')
                        print(f"[VLLMGenerateClient] Processing chunk: {chunk_str[:200]}...")
                        print(f"[VLLMGenerateClient] Full chunk length: {len(chunk_str)}")
                        
                        # Split chunk into individual JSON objects more reliably
                        json_objects = re.findall(r'{.*?}', chunk_str)
                        
                        for json_str in json_objects:
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
                                        
                                        generated_text = current_text[len(prompt):]
                                        previous_text = current_text
                            except json.JSONDecodeError as je:
                                print(f"[VLLMGenerateClient] Failed to parse JSON object: {json_str[:200]}")
                                continue

                    except Exception as e:
                        print(f"[VLLMGenerateClient] Unexpected error while processing chunk: {str(e)}")
                        print(f"[VLLMGenerateClient] Chunk content: {chunk_str[:200]}")
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

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
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