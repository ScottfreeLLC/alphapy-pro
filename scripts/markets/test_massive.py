#!/usr/bin/env python
"""
Quick Massive API Test Script (formerly Polygon.io)
Tests basic Massive API connectivity and functionality
"""

import os
import time
from datetime import datetime, timedelta
from massive import RESTClient, WebSocketClient
from massive.websocket.models import WebSocketMessage, EquityTrade

def test_api_key():
    """Check if API key is set"""
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        print("❌ MASSIVE_API_KEY environment variable not set")
        return False
    print(f"✓ API Key found: {api_key[:8]}...")
    return True

def test_rest_client():
    """Test REST client initialization"""
    try:
        client = RESTClient()
        print("✓ REST client initialized")
        return client
    except Exception as e:
        print(f"❌ Failed to initialize REST client: {e}")
        return None

def test_stock_quote(client, symbol="AAPL"):
    """Test getting a stock quote"""
    try:
        # Get last trade
        trades = client.list_trades(symbol, limit=1)
        for trade in trades:
            print(f"✓ Latest trade for {symbol}:")
            print(f"  Price: ${trade.price}")
            print(f"  Size: {trade.size}")
            print(f"  Time: {datetime.fromtimestamp(trade.participant_timestamp / 1000000000)}")
            return True
    except Exception as e:
        print(f"❌ Failed to get quote for {symbol}: {e}")
        return False

def test_historical_data(client, symbol="AAPL", days=5):
    """Test fetching historical data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get aggregates (OHLCV bars)
        aggs = client.get_aggs(
            symbol,
            1,
            "day",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            limit=50000
        )

        bars = list(aggs)
        if bars:
            print(f"✓ Historical data for {symbol} (last {days} days):")
            print(f"  Bars received: {len(bars)}")
            latest = bars[-1]
            print(f"  Latest close: ${latest.close}")
            print(f"  Volume: {latest.volume:,}")
            return True
        else:
            print(f"❌ No historical data returned for {symbol}")
            return False
    except Exception as e:
        print(f"❌ Failed to get historical data: {e}")
        return False

def test_ticker_details(client, symbol="AAPL"):
    """Test getting ticker details"""
    try:
        details = client.get_ticker_details(symbol)
        print(f"✓ Ticker details for {symbol}:")
        print(f"  Name: {details.name}")
        print(f"  Market: {details.market}")
        print(f"  Type: {details.type}")
        return True
    except Exception as e:
        print(f"❌ Failed to get ticker details: {e}")
        return False

def test_websocket():
    """Test WebSocket connection and trade stream"""
    print("Testing WebSocket connection (will run for 10 seconds)...")

    trade_count = 0
    symbols_seen = set()

    def handle_msg(msgs):
        nonlocal trade_count, symbols_seen
        for msg in msgs:
            if isinstance(msg, EquityTrade):
                trade_count += 1
                symbols_seen.add(msg.symbol)

    try:
        client = WebSocketClient()
        client.subscribe("T.AAPL", "T.TSLA", "T.NVDA")  # Subscribe to specific tickers

        # Run for 10 seconds in background
        import threading
        ws_thread = threading.Thread(target=lambda: client.run(handle_msg))
        ws_thread.daemon = True
        ws_thread.start()

        # Wait 10 seconds
        time.sleep(10)

        if trade_count > 0:
            print(f"✓ WebSocket working!")
            print(f"  Trades received: {trade_count}")
            print(f"  Symbols seen: {', '.join(sorted(symbols_seen))}")
            return True
        else:
            print(f"⚠️  WebSocket connected but no trades received (market may be closed)")
            return True

    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Massive API Test Suite")
    print("=" * 60)
    print()

    # Test 1: API Key
    if not test_api_key():
        print("\nTests aborted - no API key found")
        return

    print()

    # Test 2: REST Client
    client = test_rest_client()
    if not client:
        print("\nTests aborted - client initialization failed")
        return

    print()

    # Test 3: Stock Quote
    test_stock_quote(client)
    print()

    # Test 4: Historical Data
    test_historical_data(client)
    print()

    # Test 5: Ticker Details
    test_ticker_details(client)
    print()

    # Test 6: WebSocket
    test_websocket()
    print()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
