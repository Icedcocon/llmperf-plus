import asyncio
import aiohttp
import argparse
import time
import statistics
from datetime import datetime
import json
import traceback
import logging

class VLLMLoadTester:
    def __init__(self, url, concurrency, duration, max_tokens):
        self.url = url
        self.concurrency = concurrency
        self.duration = duration
        self.max_tokens = max_tokens
        self.results = []
        self.request_count = 0
        self.start_timestamp = int(time.time())
        self.model_name = "vllm-api"
        self.version = datetime.now().strftime("%Y-%m-%d")
        self.successful_requests = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    async def make_request(self, session):
        request_id = f"REQ-{int(time.time() * 1000)}"
        self.logger.info(f"[{request_id}] Starting new request")
        
        payload = {
            "prompt": f"Randomly stream lines from the following text with {self.max_tokens} output tokens. Don't generate eos tokens:",
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": 0.8
        }
        
        start_time = time.time()
        first_token_time = None
        self.request_count += 1
        
        try:
            # 设置 raise_for_status=False 让我们自己处理状态码
            async with session.post(
                self.url, 
                json=payload, 
                timeout=aiohttp.ClientTimeout(total=None, sock_read=None),
                raise_for_status=False
            ) as response:
                self.logger.info(f"[{request_id}] Response status: {response.status}")
                
                if response.status == 200:
                    try:
                        # 使用 response.content.iter_chunks() 来更好地处理分块传输
                        async for chunk, _ in response.content.iter_chunks():
                            if first_token_time is None:
                                first_token_time = time.time()
                                self.logger.info(f"[{request_id}] Received first token")
                        
                        # 如果成功完成了迭代，说明响应完整
                        if first_token_time is not None:
                            end_time = time.time()
                            total_time = end_time - start_time
                            self.logger.info(f"[{request_id}] Request completed successfully in {total_time:.2f}s")
                            
                            self.results.append({
                                'total_latency': total_time,
                                'ttft': first_token_time - start_time
                            })
                            self.successful_requests += 1
                        else:
                            self.logger.error(f"[{request_id}] Request failed: no data received")
                            
                    except aiohttp.ClientPayloadError as e:
                        if "Response payload is not completed" in str(e):
                            self.logger.warning(f"[{request_id}] Connection closed before completion, but data was received")
                            if first_token_time is not None:
                                # 如果已经收到数据，我们仍然认为这是一个成功的请求
                                end_time = time.time()
                                total_time = end_time - start_time
                                self.results.append({
                                    'total_latency': total_time,
                                    'ttft': first_token_time - start_time
                                })
                                self.successful_requests += 1
                        else:
                            self.logger.error(f"[{request_id}] Error reading response: {str(e)}")
                else:
                    self.logger.error(f"[{request_id}] Request failed with status {response.status}")
                
        except asyncio.TimeoutError:
            self.logger.error(f"[{request_id}] Request timed out")
        except Exception as e:
            self.logger.error(f"[{request_id}] Unexpected error: {str(e)}")

    async def run_load_test(self):
        try:
            connector = aiohttp.TCPConnector(limit=self.concurrency, force_close=True)
            async with aiohttp.ClientSession(connector=connector) as session:
                start_time = time.time()
                tasks = set()
                
                # 发送请求阶段
                while time.time() - start_time < self.duration:
                    while len(tasks) < self.concurrency:
                        tasks.add(asyncio.create_task(self.make_request(session)))
                    
                    done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                
                # 等待所有剩余请求完成，不设置超时
                if tasks:
                    self.logger.info(f"Test duration completed. Waiting for {len(tasks)} remaining requests to finish...")
                    await asyncio.wait(tasks)
                    self.logger.info("All remaining requests completed")
                    
        except asyncio.CancelledError:
            self.logger.info("Load test cancelled, cleaning up...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def calculate_percentile(self, data, percentile):
        sorted_data = sorted(data)
        index = (len(sorted_data) - 1) * percentile
        return sorted_data[int(index)]
        
    def print_results(self):
        if not self.results:
            print("No results to display")
            return
            
        total_latencies = [r['total_latency'] for r in self.results]
        ttfts = [r['ttft'] for r in self.results if r['ttft']]
        
        stats = {
            "version": self.version,
            "name": f"{self.model_name}_load_test",
            "model": self.model_name,
            "num_concurrent_requests": self.concurrency,
            "timestamp": self.start_timestamp,
            
            # TTFT statistics
            "results_ttft_s_mean": round(statistics.mean(ttfts), 4),
            "results_ttft_s_min": round(min(ttfts), 4),
            "results_ttft_s_max": round(max(ttfts), 4),
            "results_ttft_s_stddev": round(statistics.stdev(ttfts) if len(ttfts) > 1 else 0, 4),
            "results_ttft_s_quantiles_p50": round(self.calculate_percentile(ttfts, 0.50), 4),
            "results_ttft_s_quantiles_p90": round(self.calculate_percentile(ttfts, 0.90), 4),
            "results_ttft_s_quantiles_p95": round(self.calculate_percentile(ttfts, 0.95), 4),
            "results_ttft_s_quantiles_p99": round(self.calculate_percentile(ttfts, 0.99), 4),
            
            # End-to-end latency statistics
            "results_end_to_end_latency_s_mean": round(statistics.mean(total_latencies), 4),
            "results_end_to_end_latency_s_min": round(min(total_latencies), 4),
            "results_end_to_end_latency_s_max": round(max(total_latencies), 4),
            "results_end_to_end_latency_s_stddev": round(statistics.stdev(total_latencies) if len(total_latencies) > 1 else 0, 4),
            "results_end_to_end_latency_s_quantiles_p50": round(self.calculate_percentile(total_latencies, 0.50), 4),
            "results_end_to_end_latency_s_quantiles_p90": round(self.calculate_percentile(total_latencies, 0.90), 4),
            "results_end_to_end_latency_s_quantiles_p95": round(self.calculate_percentile(total_latencies, 0.95), 4),
            "results_end_to_end_latency_s_quantiles_p99": round(self.calculate_percentile(total_latencies, 0.99), 4),
            
            # Request statistics
            "results_num_requests_started": self.request_count,
            "results_num_completed_requests": self.successful_requests,
            "results_num_completed_requests_per_min": round((self.successful_requests * 60) / self.duration, 4)
        }
        
        print("\n=== Load Test Results ===")
        print("\nTest Configuration:")
        print(f"Model: {stats['model']}")
        print(f"Concurrent Requests: {stats['num_concurrent_requests']}")
        print(f"Test Duration: {self.duration} seconds")
        
        print("\nRequest Statistics:")
        print(f"Total Requests Started: {stats['results_num_requests_started']}")
        print(f"Successful Requests: {stats['results_num_completed_requests']}")
        print(f"Requests/min: {stats['results_num_completed_requests_per_min']}")
        
        print("\nLatency Statistics (seconds):")
        print("Time To First Token (TTFT):")
        print(f"  Mean: {stats['results_ttft_s_mean']}")
        print(f"  Min/Max: {stats['results_ttft_s_min']} / {stats['results_ttft_s_max']}")
        print(f"  P50/P90/P99: {stats['results_ttft_s_quantiles_p50']} / {stats['results_ttft_s_quantiles_p90']} / {stats['results_ttft_s_quantiles_p99']}")
        
        print("\nEnd-to-End Latency:")
        print(f"  Mean: {stats['results_end_to_end_latency_s_mean']}")
        print(f"  Min/Max: {stats['results_end_to_end_latency_s_min']} / {stats['results_end_to_end_latency_s_max']}")
        print(f"  P50/P90/P99: {stats['results_end_to_end_latency_s_quantiles_p50']} / {stats['results_end_to_end_latency_s_quantiles_p90']} / {stats['results_end_to_end_latency_s_quantiles_p99']}")
        
        print("\nDetailed JSON Results:")
        print(json.dumps(stats, indent=2))

async def main():
    parser = argparse.ArgumentParser(description='VLLM API Load Tester')
    parser.add_argument('--url', default='http://localhost:8000/generate',
                      help='VLLM API endpoint URL')
    parser.add_argument('--concurrency', type=int, default=10,
                      help='Number of concurrent requests')
    parser.add_argument('--duration', type=int, default=60,
                      help='Test duration in seconds')
    parser.add_argument('--max-tokens', type=int, default=32,
                      help='Maximum number of tokens to generate')
    
    args = parser.parse_args()
    
    try:
        tester = VLLMLoadTester(args.url, args.concurrency, args.duration, args.max_tokens)
        print(f"Starting load test with {args.concurrency} concurrent requests for {args.duration} seconds...")
        await tester.run_load_test()
        tester.print_results()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down gracefully...")
    finally:
        # 确保所有pending的任务都被清理
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user") 