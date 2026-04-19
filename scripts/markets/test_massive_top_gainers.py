#!/usr/bin/env python
"""
Massive Top Gainers Test Script v5 (formerly Polygon.io)
Uses the Top Market Movers endpoint: /v2/snapshot/locale/us/markets/stocks/{direction}
"""

import os
from massive import RESTClient

def test_api_key():
    """Check if API key is set"""
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        print("❌ MASSIVE_API_KEY environment variable not set")
        return False
    print(f"✓ API Key found: {api_key[:8]}...")
    return True

def get_top_gainers(client, limit=20):
    """Get top gainers using top market movers endpoint"""
    try:
        print(f"\nFetching top {limit} gainers...")

        # Try the direct method call
        try:
            resp = client.get_top_market_movers(direction="gainers")
        except AttributeError:
            # If that doesn't work, try the raw request
            import requests
            api_key = os.getenv("MASSIVE_API_KEY")
            url = f"https://api.massive.com/v2/snapshot/locale/us/markets/stocks/gainers?apiKey={api_key}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'OK':
                print(f"❌ API returned status: {data.get('status')}")
                return False

            tickers = data.get('tickers', [])

            gainers = []
            for ticker_data in tickers:
                symbol = ticker_data.get('ticker', '')
                today_change_pct = ticker_data.get('todaysChangePerc', 0)
                price = 0
                volume = 0

                day = ticker_data.get('day', {})
                if day:
                    price = day.get('c', 0)
                    volume = day.get('v', 0)

                gainers.append({
                    'symbol': symbol,
                    'price': price,
                    'change_pct': today_change_pct,
                    'volume': volume
                })

            # Sort by percent change (descending)
            gainers.sort(key=lambda x: x['change_pct'], reverse=True)

            # Get top N
            top_gainers = gainers[:limit]

            print(f"Total gainers received: {len(gainers)}")
            print()

            if top_gainers:
                print(f"{'Symbol':<10} {'Price':<12} {'Change %':<12} {'Volume':<15}")
                print("=" * 49)

                for gainer in top_gainers:
                    print(
                        f"{gainer['symbol']:<10} "
                        f"${gainer['price']:<11.2f} "
                        f"{gainer['change_pct']:>10.2f}% "
                        f"{int(gainer['volume']):>14,}"
                    )

                return True
            else:
                print("❌ No gainers data available")
                return False

    except Exception as e:
        print(f"❌ Failed to get gainers: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_top_losers(client, limit=20):
    """Get top losers using top market movers endpoint"""
    try:
        print(f"\nFetching top {limit} losers...")

        # Use raw request approach
        import requests
        api_key = os.getenv("MASSIVE_API_KEY")
        url = f"https://api.massive.com/v2/snapshot/locale/us/markets/stocks/losers?apiKey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get('status') != 'OK':
            print(f"❌ API returned status: {data.get('status')}")
            return False

        tickers = data.get('tickers', [])

        losers = []
        for ticker_data in tickers:
            symbol = ticker_data.get('ticker', '')
            today_change_pct = ticker_data.get('todaysChangePerc', 0)
            price = 0
            volume = 0

            day = ticker_data.get('day', {})
            if day:
                price = day.get('c', 0)
                volume = day.get('v', 0)

            losers.append({
                'symbol': symbol,
                'price': price,
                'change_pct': today_change_pct,
                'volume': volume
            })

        # Sort by percent change (ascending for losers)
        losers.sort(key=lambda x: x['change_pct'])

        # Get top N
        top_losers = losers[:limit]

        print(f"Total losers received: {len(losers)}")
        print()

        if top_losers:
            print(f"{'Symbol':<10} {'Price':<12} {'Change %':<12} {'Volume':<15}")
            print("=" * 49)

            for loser in top_losers:
                print(
                    f"{loser['symbol']:<10} "
                    f"${loser['price']:<11.2f} "
                    f"{loser['change_pct']:>10.2f}% "
                    f"{int(loser['volume']):>14,}"
                )

            return True
        else:
            print("❌ No losers data available")
            return False

    except Exception as e:
        print(f"❌ Failed to get losers: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run gainers/losers test"""
    print("=" * 80)
    print("Massive Top Gainers & Losers Test (v5 - Using Top Market Movers Endpoint)")
    print("=" * 80)
    print()

    # Test API Key
    if not test_api_key():
        print("\nTest aborted - no API key found")
        return

    print()

    # Initialize REST Client
    try:
        client = RESTClient()
        print("✓ REST client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize REST client: {e}")
        return

    # Get top gainers
    get_top_gainers(client, limit=20)

    print()
    print("-" * 80)

    # Get top losers
    get_top_losers(client, limit=20)

    print()
    print("=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
