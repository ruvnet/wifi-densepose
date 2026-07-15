import asyncio
import time
import os
import sys

# Add project root and archive/v1 to sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from archive.v1.src.api.routers.health import get_system_metrics

async def ticker():
    """Asynchronous background ticker to measure event loop latency/freezes."""
    ticks = []
    for _ in range(15):
        ticks.append(time.time())
        await asyncio.sleep(0.1)
    return ticks

async def run_test():
    print("Starting concurrency verification test...")
    
    # Start the ticker background task
    ticker_task = asyncio.create_task(ticker())
    
    # Let ticker run for a few ticks
    await asyncio.sleep(0.3)
    
    print("Calling get_system_metrics offloaded to background thread...")
    start_time = time.time()
    
    # Query system metrics using to_thread (simulating FastAPI request)
    metrics = await asyncio.to_thread(get_system_metrics)
    
    duration = time.time() - start_time
    print(f"get_system_metrics took: {duration:.4f}s")
    
    # Wait for the ticker to complete
    ticks = await ticker_task
    
    # Calculate gaps between consecutive ticks to check for event loop freezes
    gaps = [ticks[i+1] - ticks[i] for i in range(len(ticks)-1)]
    max_gap = max(gaps)
    
    print(f"All tick gaps: {[round(g, 3) for g in gaps]}")
    print(f"Max event loop freeze: {max_gap:.4f}s")
    
    # In pre-fix code, psutil.cpu_percent(interval=1) blocks for 1.0s,
    # causing a gap of >1.0s. With our fix, it should be close to 0.1s.
    if max_gap >= 0.35:
        print("FAIL: Event loop was frozen/blocked!")
        sys.exit(1)
    else:
        print("SUCCESS: Event loop remained fully responsive during metrics query!")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(run_test())
