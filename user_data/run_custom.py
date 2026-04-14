import ccxt
import ccxt.async_support
import ccxt.pro
from freqtrade.exchange.common import MAP_EXCHANGE_CHILDCLASS, SUPPORTED_EXCHANGES

CUSTOM_EXCHANGES = ['portfoliobench', 'polymarket']

for name in CUSTOM_EXCHANGES:
    if name not in ccxt.exchanges:
        ccxt.exchanges.append(name)
    if hasattr(ccxt.async_support, 'exchanges') and name not in ccxt.async_support.exchanges:
        ccxt.async_support.exchanges.append(name)
    if hasattr(ccxt.pro, 'exchanges') and name not in ccxt.pro.exchanges:
        ccxt.pro.exchanges.append(name)

    setattr(ccxt, name, ccxt.binance)
    setattr(ccxt.async_support, name, ccxt.async_support.binance)
    setattr(ccxt.pro, name, ccxt.pro.binance)

    if name not in SUPPORTED_EXCHANGES:
        SUPPORTED_EXCHANGES.append(name)
    MAP_EXCHANGE_CHILDCLASS[name] = name

from freqtrade.main import main

if __name__ == '__main__':
    main()