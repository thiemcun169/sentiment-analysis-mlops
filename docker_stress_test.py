#!/usr/bin/env python3
"""
Docker Compose Stress Test for Sentiment Analysis Service
Tests both FastAPI and direct Triton endpoints in containerized environment
"""

import asyncio
import aiohttp
import time
import json
import statistics
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import argparse

@dataclass
class TestResult:
    success: bool
    response_time: float
    status_code: int = None
    error: str = None

class DockerStressTest:
    def __init__(self):
        self.fastapi_url = "http://localhost:8080"
        self.triton_url = "http://localhost:8000"
        
        # Test texts for sentiment analysis
        self.test_texts = [
            "I absolutely love this product! Outstanding quality and amazing features.",
            "This is the worst purchase I've ever made. Complete waste of money.",
            "The product is okay, nothing special but it works as expected.",
            "¡Excelente servicio al cliente! Muy satisfecho con la compra.",
            "Service client terrible, je ne recommande pas du tout.",
            "Outstanding value for money! Exceeded all my expectations completely.",
            "Fantastic customer support team, they solved my issue immediately.",
            "Poor quality materials, broke within the first week of use.",
            "Best product I've bought this year! Highly recommend to everyone.",
            "Completely disappointed with this purchase, requesting full refund.",
            "Great design and functionality, very pleased with the results.",
            "Average quality product, could be better for the price paid.",
        ]

    async def test_fastapi_health(self, session: aiohttp.ClientSession) -> bool:
        """Test FastAPI service health"""
        try:
            async with session.get(f"{self.fastapi_url}/health") as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("status") == "healthy"
        except:
            pass
        return False

    async def test_triton_health(self, session: aiohttp.ClientSession) -> bool:
        """Test Triton server health"""
        try:
            async with session.get(f"{self.triton_url}/v2/health/ready") as response:
                return response.status == 200
        except:
            pass
        return False

    async def test_fastapi_predict(self, session: aiohttp.ClientSession, texts: List[str]) -> TestResult:
        """Test FastAPI predict endpoint"""
        start_time = time.time()
        try:
            payload = {
                "texts": texts,
                "return_scores": True
            }
            
            async with session.post(
                f"{self.fastapi_url}/predict",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_text = await response.text()
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = json.loads(response_text)
                    if "results" in result and len(result["results"]) == len(texts):
                        return TestResult(
                            success=True,
                            response_time=response_time,
                            status_code=response.status
                        )
                
                return TestResult(
                    success=False,
                    response_time=response_time,
                    status_code=response.status,
                    error=response_text[:200]
                )
                
        except Exception as e:
            return TestResult(
                success=False,
                response_time=time.time() - start_time,
                error=str(e)
            )

    async def test_fastapi_single(self, session: aiohttp.ClientSession, text: str) -> TestResult:
        """Test FastAPI single prediction endpoint"""
        start_time = time.time()
        try:
            import urllib.parse
            # Properly encode the text parameter
            encoded_text = urllib.parse.quote(text)
            url = f"{self.fastapi_url}/predict/single?text={encoded_text}&return_scores=true"
            
            async with session.post(
                url,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_text = await response.text()
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = json.loads(response_text)
                    if "label" in result and "score" in result:
                        return TestResult(
                            success=True,
                            response_time=response_time,
                            status_code=response.status
                        )
                
                return TestResult(
                    success=False,
                    response_time=response_time,
                    status_code=response.status,
                    error=response_text[:200]
                )
                
        except Exception as e:
            return TestResult(
                success=False,
                response_time=time.time() - start_time,
                error=str(e)
            )

    async def run_concurrent_test(self, test_func, test_data, num_requests: int, 
                                 concurrency: int, test_name: str):
        """Run concurrent tests"""
        print(f"\n🚀 Starting {test_name}")
        print(f"📊 Requests: {num_requests}, Concurrency: {concurrency}")
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_test(session, data):
            async with semaphore:
                return await test_func(session, data)
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            # Create tasks
            tasks = []
            for i in range(num_requests):
                data = test_data[i % len(test_data)]
                task = limited_test(session, data)
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
                for error, count in list(error_counts.items())[:3]:
                    print(f"     {error[:50]}...: {count}")
            
            if exceptions:
                print(f"  💥 Exceptions: {len(exceptions)}")
            
            return {
                'test_name': test_name,
                'success_rate': success_rate,
                'requests_per_second': requests_per_second,
                'avg_response_time_ms': avg_response_time * 1000,
                'p95_response_time_ms': p95_response_time * 1000,
                'total_requests': num_requests,
                'successful_requests': len(successful_results)
            }

    async def health_check(self):
        """Check if all services are healthy"""
        print("🔍 Checking service health...")
        
        async with aiohttp.ClientSession() as session:
            fastapi_healthy = await self.test_fastapi_health(session)
            triton_healthy = await self.test_triton_health(session)
        
        print(f"  FastAPI Service: {'✅' if fastapi_healthy else '❌'}")
        print(f"  Triton Server: {'✅' if triton_healthy else '❌'}")
        
        return fastapi_healthy and triton_healthy

    async def run_stress_tests(self, args):
        """Run the complete stress test suite"""
        print("🔥 DOCKER COMPOSE SENTIMENT ANALYSIS STRESS TEST")
        print("=" * 60)
        
        # Health check
        if not await self.health_check():
            print("❌ Services not healthy! Please check your Docker Compose setup.")
            print("\n🚀 Quick fix commands:")
            print("   docker-compose up -d")
            print("   docker-compose logs -f")
            return
        
        print("\n🎯 Starting FastAPI stress tests...\n")
        
        # Test scenarios
        test_scenarios = [
            (args.light_requests, args.light_concurrency, "FastAPI Light Load - Batch"),
            (args.medium_requests, args.medium_concurrency, "FastAPI Medium Load - Batch"),
            (args.heavy_requests, args.heavy_concurrency, "FastAPI Heavy Load - Batch"),
        ]
        
        # Single text scenarios
        single_scenarios = [
            (args.light_requests, args.light_concurrency, "FastAPI Light Load - Single"),
            (args.medium_requests, args.medium_concurrency, "FastAPI Medium Load - Single"),
        ]
        
        results = []
        
        # Run batch prediction tests
        for requests, concurrency, name in test_scenarios:
            if requests > 0:
                # Use batches of 2-4 texts
                batch_data = []
                for i in range(requests):
                    batch_size = 2 + (i % 3)  # Batch sizes 2, 3, 4
                    batch = [self.test_texts[j % len(self.test_texts)] for j in range(i, i + batch_size)]
                    batch_data.append(batch)
                
                result = await self.run_concurrent_test(
                    self.test_fastapi_predict, 
                    batch_data, 
                    requests, 
                    concurrency, 
                    name
                )
                results.append(result)
                await asyncio.sleep(2)  # Pause between tests
        
        # Run single prediction tests
        for requests, concurrency, name in single_scenarios:
            if requests > 0:
                result = await self.run_concurrent_test(
                    self.test_fastapi_single,
                    self.test_texts,
                    requests,
                    concurrency,
                    name
                )
                results.append(result)
                await asyncio.sleep(2)
        
        # Summary
        print("\n" + "=" * 60)
        print("🏆 PERFORMANCE SUMMARY")
        print("=" * 60)
        
        for result in results:
            print(f"\n{result['test_name']}:")
            print(f"  Success: {result['success_rate']:.1f}% ({result['successful_requests']}/{result['total_requests']})")
            print(f"  Throughput: {result['requests_per_second']:.2f} req/s")
            print(f"  Avg Latency: {result['avg_response_time_ms']:.2f}ms")
            print(f"  P95 Latency: {result['p95_response_time_ms']:.2f}ms")
        
        # Find best performance
        if results:
            best_throughput = max(results, key=lambda x: x['requests_per_second'])
            print(f"\n🏅 Best Performance: {best_throughput['test_name']}")
            print(f"   Throughput: {best_throughput['requests_per_second']:.2f} req/s")
            print(f"   Success Rate: {best_throughput['success_rate']:.1f}%")
        
        # Service info
        print(f"\n📋 Service Endpoints:")
        print(f"   FastAPI: {self.fastapi_url}")
        print(f"   Triton: {self.triton_url}")
        print(f"   Docs: {self.fastapi_url}/docs")

def main():
    parser = argparse.ArgumentParser(description="Docker Compose stress test for sentiment analysis")
    parser.add_argument("--light-requests", type=int, default=20, help="Light load requests")
    parser.add_argument("--light-concurrency", type=int, default=3, help="Light load concurrency")
    parser.add_argument("--medium-requests", type=int, default=50, help="Medium load requests")
    parser.add_argument("--medium-concurrency", type=int, default=8, help="Medium load concurrency")
    parser.add_argument("--heavy-requests", type=int, default=100, help="Heavy load requests")
    parser.add_argument("--heavy-concurrency", type=int, default=15, help="Heavy load concurrency")
    parser.add_argument("--quick", action="store_true", help="Quick test (reduced load)")
    
    args = parser.parse_args()
    
    if args.quick:
        args.light_requests = 10
        args.light_concurrency = 2
        args.medium_requests = 25
        args.medium_concurrency = 5
        args.heavy_requests = 50
        args.heavy_concurrency = 10
    
    stress_tester = DockerStressTest()
    asyncio.run(stress_tester.run_stress_tests(args))

if __name__ == "__main__":
    main() 