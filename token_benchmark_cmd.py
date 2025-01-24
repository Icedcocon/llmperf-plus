import os
import sys
from token_benchmark_ray import run_token_benchmark

def main():
    # 设置环境变量
    os.environ["OPENAI_API_KEY"] = "xxxx"
    os.environ["OPENAI_API_BASE"] = "http://192.168.5.3:9997"
    os.environ["VLLM_API_BASE"] = "http://192.168.5.3:9997"

    # 设置参数
    params = {
        "llm_api": "vllmgenerate",
        "model": "qwen-chat",
        "test_timeout_s": 600,
        "max_num_completed_requests": 10,  # 使用默认值
        "mean_input_tokens": 1024,
        "stddev_input_tokens": 128,
        "mean_output_tokens": 128,
        "stddev_output_tokens": 128,
        "num_concurrent_requests": 10,  # 使用默认值
        "additional_sampling_params": "{}",
        "results_dir": "",
        "user_metadata": {}
    }

    # 初始化 Ray
    import ray
    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})

    # 运行基准测试
    run_token_benchmark(**params)

if __name__ == "__main__":
    main() 