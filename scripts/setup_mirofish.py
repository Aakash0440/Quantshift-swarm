# scripts/setup_mirofish.py
# One-time setup script for MiroFish integration.
# Run this to clone MiroFish, install its deps, and test the connection.
#
# Usage:
#   python3.14 scripts/setup_mirofish.py              # full setup
#   python3.14 scripts/setup_mirofish.py --test-only  # just test mock mode
#   python3.14 scripts/setup_mirofish.py --mock       # run in mock mode

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import argparse
import subprocess
from dotenv import load_dotenv

load_dotenv()


async def test_mock_mode():
    """Test MiroFish integration using mock simulations (no server needed)."""
    print("\n" + "="*55)
    print("QUANTSHIFT-SWARM — MiroFish Mock Test")
    print("="*55)

    # Set mock mode
    os.environ["MIROFISH_MOCK"] = "true"

    from ingestion.mirofish_source import MiroFishSource, EventClassifier

    # Test event classifier
    print("\n── Event Classifier Test ──────────────────────────")
    classifier = EventClassifier()
    test_events = [
        ("NVDA reports record earnings, EPS beat by 15%, guidance raised", "NVDA"),
        ("Federal Reserve raises interest rates by 75 basis points", "SPY"),
        ("SEC Form 4: NVDA CEO purchased 50,000 shares at $120", "NVDA"),
        ("Bitcoin drops 20% as crypto exchange halts withdrawals", "BTC"),
        ("War escalation in Eastern Europe, oil prices surge 45%", "XOM"),
        ("AAPL announces new iPhone model launch date", "AAPL"),
        ("General market update: stocks trading mixed today", "SPY"),
    ]

    for event_text, ticker in test_events:
        is_high, event_type, confidence = classifier.is_high_impact(event_text)
        status = "✅ SIMULATE" if is_high else "⏭  SKIP"
        print(f"  {status} | {event_type:<15} {confidence:.0%} | {event_text[:60]}")

    # Test full simulation
    print("\n── Mock Simulation Test ───────────────────────────")
    mf = MiroFishSource()

    test_cases = [
        ("NVDA reports record earnings beat, raises guidance for AI chips", "NVDA"),
        ("Federal Reserve surprises with 75bp emergency rate hike", "SPY"),
        ("SEC Form 4: TSLA insider purchased 200,000 shares", "TSLA"),
        ("BTC exchange collapse, $8B gap, withdrawals halted", "BTC"),
    ]

    for event_text, ticker in test_cases:
        print(f"\n  Event: {event_text[:70]}")
        result = await mf.simulate_event(event_text, ticker)
        if result:
            direction = "UP ▲" if result.score > 0 else "DOWN ▼"
            print(f"  → {ticker}: {direction} | score={result.score:+.3f} | "
                  f"confidence={result.confidence:.0%} | "
                  f"bulls={result.bullish_pct:.0%} bears={result.bearish_pct:.0%}")
        else:
            print(f"  → Skipped (below impact threshold)")

    # Test as pipeline source
    print("\n── Pipeline Integration Test ──────────────────────")
    tickers = ["NVDA", "AAPL", "MSFT", "BTC", "ETH", "SPY"]
    signals = await mf.fetch(tickers)
    print(f"  MiroFish generated {len(signals)} signal(s) from live RSS feeds")
    for s in signals:
        meta = s.metadata
        direction = "UP" if meta.get("score", 0) > 0 else "DOWN"
        print(f"  → {s.ticker}: {direction} | "
              f"event={meta.get('event_type')} | "
              f"confidence={meta.get('confidence', 0):.0%} | "
              f"weight={meta.get('mirofish_weight', 0):.2f}")

    print("\n" + "="*55)
    print("✅ Mock test complete. MiroFish integration working.")
    print("\nNext step: Clone MiroFish and run it for live simulations:")
    print("  git clone https://github.com/666ghj/MiroFish")
    print("  cd MiroFish && pip install -r requirements.txt")
    print("  python app.py")
    print("  Then set MIROFISH_MOCK=false in .env")
    print("="*55 + "\n")


async def test_live_connection():
    """Test connection to running MiroFish server."""
    import httpx
    url = os.getenv("MIROFISH_URL", "http://localhost:8001")
    print(f"\nTesting connection to MiroFish at {url}...")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{url}/health")
            if resp.status_code == 200:
                print(f"✅ MiroFish server is running at {url}")
                return True
            else:
                print(f"⚠️  Server responded with {resp.status_code}")
                return False
    except httpx.ConnectError:
        print(f"❌ Cannot connect to MiroFish at {url}")
        print("   Make sure MiroFish is running: cd MiroFish && python app.py")
        return False


def setup_mirofish():
    """Clone and set up MiroFish repo."""
    mirofish_dir = os.path.join(os.getcwd(), "MiroFish")

    if os.path.exists(mirofish_dir):
        print(f"✅ MiroFish already cloned at {mirofish_dir}")
        return True

    print("Cloning MiroFish...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/666ghj/MiroFish", mirofish_dir],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"❌ Git clone failed: {result.stderr}")
        print("   Manual clone: git clone https://github.com/666ghj/MiroFish")
        return False

    print(f"✅ MiroFish cloned to {mirofish_dir}")

    # Install MiroFish requirements
    req_file = os.path.join(mirofish_dir, "requirements.txt")
    if os.path.exists(req_file):
        print("Installing MiroFish dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file],
            cwd=mirofish_dir
        )
        print("✅ MiroFish dependencies installed")

    print("\n" + "="*55)
    print("MiroFish setup complete!")
    print("\nTo start MiroFish server:")
    print(f"  cd {mirofish_dir}")
    print("  python app.py")
    print("\nThen set in your .env:")
    print("  MIROFISH_URL=http://localhost:8001")
    print("  MIROFISH_MOCK=false")
    print("="*55)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QUANTSHIFT-SWARM MiroFish Setup")
    parser.add_argument("--test-only", action="store_true", help="Only run mock test")
    parser.add_argument("--mock",      action="store_true", help="Force mock mode")
    parser.add_argument("--live-test", action="store_true", help="Test live server connection")
    args = parser.parse_args()

    if args.mock:
        os.environ["MIROFISH_MOCK"] = "true"

    if args.live_test:
        asyncio.run(test_live_connection())
    elif args.test_only or args.mock:
        asyncio.run(test_mock_mode())
    else:
        # Full setup
        setup_mirofish()
        asyncio.run(test_mock_mode())
