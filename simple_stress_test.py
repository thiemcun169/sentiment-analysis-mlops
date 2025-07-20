#!/usr/bin/env python3
"""
Simple stress test for sentiment analysis Triton server
Tests the working TensorRT model directly
"""

import asyncio
import aiohttp
import time
import json
import statistics
import numpy as np
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List
import argparse

@dataclass
class TestResult:
    success: bool
    response_time: float
    status_code: int = None
    error: str = None

class TensorRTStressTest:
    def __init__(self):
        self.triton_url = "http://localhost:8000"
        self.model_name = "sentiment_analysis_tensorrt"
        self.tokenizer = None
        
        # Test texts in different languages
        self.test_texts = [
            "I love this product! It's amazing and works perfectly.",
            "This is terrible quality, waste of money.",
            "Average product, nothing special but okay.",
            "¡Excelente servicio! Muy recomendado para todos.",
            "Terrible experience, would not recommend at all.",
            "Great value for money, exceeded my expectations completely.",
            "Outstanding customer service and fast delivery.",
            "Product broke after one day of use, very poor quality.",
            "Absolutely fantastic! Best purchase I've made this year!",
            "Worst customer service experience ever, completely unhelpful staff.",
        ]

    def load_tokenizer(self):
        """Load the tokenizer"""
        if self.tokenizer is None:
            print("🔄 Loading tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "tabularisai/multilingual-sentiment-analysis",
                    local_files_only=True
                )
                print("✅ Tokenizer loaded from cache")
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "tabularisai/multilingual-sentiment-analysis"
                )
                print("✅ Tokenizer loaded from download")

    def prepare_triton_payload(self, text: str):
        """Prepare payload for Triton inference"""
        # Tokenize the text
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np"
        )
        
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)
        
        # Create Triton inference payload
        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": input_ids.shape,
                    "datatype": "INT64",
                    "data": input_ids.flatten().tolist()
                },
                {
                    "name": "attention_mask",
                    "shape": attention_mask.shape,
                    "datatype": "INT64",
                    "data": attention_mask.flatten().tolist()
                }
            ],
            "outputs": [
                {
                    "name": "output"
                }
            ]
        }
        return payload

    async def test_triton_inference(self, session: aiohttp.ClientSession, text: str) -> TestResult:
        """Test Triton inference endpoint"""
        start_time = time.time()
        try:
            payload = self.prepare_triton_payload(text)
            
            async with session.post(
                f"{self.triton_url}/v2/models/{self.model_name}/infer",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_text = await response.text()
                response_time = time.time() - start_time
                
                if response.status == 200:
                    # Parse response to verify it's working
                    result = json.loads(response_text)
                    if "outputs" in result and len(result["outputs"]) > 0:
                        return TestResult(
                            success=True,
                            response_time=response_time,
                            status_code=response.status
                        )
                
                return TestResult(
                    success=False,
                    response_time=response_time,
                    status_code=response.status,
                    error=response_text[:200]  # First 200 chars of error
                )
                
        except Exception as e:
            return TestResult(
                success=False,
                response_time=time.time() - start_time,
                error=str(e)
            )

    async def run_concurrent_test(self, num_requests: int, concurrency: int, test_name: str):
        """Run concurrent inference requests"""
        print(f"\n🚀 Starting {test_name}")
        print(f"📊 Requests: {num_requests}, Concurrency: {concurrency}")
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_test(session, text):
            async with semaphore:
                return await self.test_triton_inference(session, text)
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            # Create tasks
            tasks = []
            for i in range(num_requests):
                text = self.test_texts[i % len(self.test_texts)]
                task = limited_test(session, text)
                tasks.append(task)
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            # Process results
            successful_results = [r for r in results if isinstance(r, TestResult) and r.success]
            failed_results = [r for r in results if isinstance(r, TestResult) and not r.success]
            exceptions = [r for r in results if not isinstance(r, TestResult)]
            
            # Calculate metrics
            if successful_results:
                response_times = [r.response_time for r in successful_results]
                avg_response_time = statistics.mean(response_times)
                p50_response_time = statistics.median(response_times)
                p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 5 else max(response_times)
                min_response_time = min(response_times)
                max_response_time = max(response_times)
            else:
                avg_response_time = p50_response_time = p95_response_time = min_response_time = max_response_time = 0
            
            requests_per_second = num_requests / total_time
            success_rate = len(successful_results) / num_requests * 100
            
            # Print results
            print(f"\n📈 {test_name} Results:")
            print(f"  ✅ Success Rate: {success_rate:.1f}% ({len(successful_results)}/{num_requests})")
            print(f"  ⚡ Throughput: {requests_per_second:.2f} req/s")
            print(f"  ⏱️  Total Time: {total_time:.2f}s")
            print(f"  📊 Response Times:")
            print(f"     Average: {avg_response_time*1000:.2f}ms")
            print(f"     Median (P50): {p50_response_time*1000:.2f}ms")
            print(f"     P95: {p95_response_time*1000:.2f}ms")
            print(f"     Min: {min_response_time*1000:.2f}ms")
            print(f"     Max: {max_response_time*1000:.2f}ms")
            
            if failed_results:
                print(f"  ❌ Failed: {len(failed_results)}")
                error_counts = {}
                for result in failed_results:
                    error = result.error or f"HTTP {result.status_code}"
                    error_counts[error] = error_counts.get(error, 0) + 1
                for error, count in list(error_counts.items())[:3]:  # Show top 3 errors
                    print(f"     {error[:50]}...: {count}")
            
            if exceptions:
                print(f"  💥 Exceptions: {len(exceptions)}")
            
            return {
                'success_rate': success_rate,
                'requests_per_second': requests_per_second,
                'avg_response_time_ms': avg_response_time * 1000,
                'p95_response_time_ms': p95_response_time * 1000,
                'throughput': requests_per_second
            }

    async def health_check(self):
        """Check if Triton server is healthy"""
        print("🔍 Checking Triton server health...")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Check server health
                async with session.get(f"{self.triton_url}/v2/health/ready") as response:
                    server_ready = response.status == 200
                
                # Check model availability
                async with session.get(f"{self.triton_url}/v2/models/{self.model_name}") as response:
                    model_available = response.status == 200
                    
            except Exception as e:
                print(f"❌ Health check failed: {e}")
                return False
        
        print(f"  Triton Server: {'✅' if server_ready else '❌'}")
        print(f"  Model Available: {'✅' if model_available else '❌'}")
        
        return server_ready and model_available

    async def run_stress_tests(self, args):
        """Run the stress test suite"""
        print("🔥 TENSORRT SENTIMENT ANALYSIS STRESS TEST")
        print("=" * 50)
        
        # Load tokenizer
        self.load_tokenizer()
        
        # Health check
        if not await self.health_check():
            print("❌ Services not healthy! Please check your setup.")
            return
        
        print("\n🎯 Starting TensorRT stress tests...\n")
        
        test_scenarios = [
            (args.light_requests, args.light_concurrency, "Light Load"),
            (args.medium_requests, args.medium_concurrency, "Medium Load"),
            (args.heavy_requests, args.heavy_concurrency, "Heavy Load"),
        ]
        
        results = {}
        
        for requests, concurrency, name in test_scenarios:
            if requests > 0:
                results[name] = await self.run_concurrent_test(requests, concurrency, name)
                await asyncio.sleep(1)  # Brief pause between tests
        
        # Summary
        print("\n" + "=" * 50)
        print("🏆 PERFORMANCE SUMMARY")
        print("=" * 50)
        
        for test_name, result in results.items():
            print(f"\n{test_name}:")
            print(f"  Success: {result['success_rate']:.1f}%")
            print(f"  Throughput: {result['requests_per_second']:.2f} req/s")
            print(f"  Avg Latency: {result['avg_response_time_ms']:.2f}ms")
            print(f"  P95 Latency: {result['p95_response_time_ms']:.2f}ms")
        
        # Find best performance
        if results:
            best_throughput = max(results.values(), key=lambda x: x['throughput'])
            best_test = [k for k, v in results.items() if v['throughput'] == best_throughput['throughput']][0]
            print(f"\n🏅 Best Performance: {best_test} ({best_throughput['throughput']:.2f} req/s)")

def main():
    parser = argparse.ArgumentParser(description="Stress test TensorRT sentiment analysis")
    parser.add_argument("--light-requests", type=int, default=30, help="Light load requests")
    parser.add_argument("--light-concurrency", type=int, default=3, help="Light load concurrency")
    parser.add_argument("--medium-requests", type=int, default=100, help="Medium load requests")
    parser.add_argument("--medium-concurrency", type=int, default=10, help="Medium load concurrency")
    parser.add_argument("--heavy-requests", type=int, default=300, help="Heavy load requests")
    parser.add_argument("--heavy-concurrency", type=int, default=25, help="Heavy load concurrency")
    parser.add_argument("--quick", action="store_true", help="Quick test (reduced load)")
    
    args = parser.parse_args()
    
    if args.quick:
        args.light_requests = 10
        args.light_concurrency = 2
        args.medium_requests = 30
        args.medium_concurrency = 5
        args.heavy_requests = 60
        args.heavy_concurrency = 10
    
    stress_tester = TensorRTStressTest()
    asyncio.run(stress_tester.run_stress_tests(args))

if __name__ == "__main__":
    main() 