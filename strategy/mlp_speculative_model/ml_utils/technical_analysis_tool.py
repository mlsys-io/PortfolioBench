import numpy as np
import pandas as pd
import talib as talib

BUY = -1
HOLD = 0
SELL = 1


class TecnicalAnalysis:

    @staticmethod
    def compute_oscillators(data):
        log_return = np.log(data['close']) - np.log(data['close'].shift(1))
        data['z_score'] = ((log_return - log_return.rolling(20).mean()) / log_return.rolling(20).std())
        data['rsi'] = ((talib.RSI(data['close'])) / 100)
        upper_band, _, lower_band = talib.BBANDS(data['close'], nbdevup=2, nbdevdn=2, matype=0)
        data['boll'] = ((data['close'] - lower_band) / (upper_band - lower_band))
        data['ULTOSC'] = ((talib.ULTOSC(data['high'], data['low'], data['close'])) / 100)
        data['pct_change'] = (data['close'].pct_change())
        data['zsVol'] = (data['volume'] - data['volume'].mean()) / data['volume'].std()
        data['PR_MA_Ratio_short'] = \
            ((data['close'] - talib.SMA(data['close'], 21)) / talib.SMA(data['close'], 21))
        data['MA_Ratio_short'] = \
            ((talib.SMA(data['close'], 21) - talib.SMA(data['close'], 50)) / talib.SMA(data['close'], 50))
        data['MA_Ratio'] = (
                    (talib.SMA(data['close'], 50) - talib.SMA(data['close'], 100)) / talib.SMA(data['close'], 100))
        data['PR_MA_Ratio'] = ((data['close'] - talib.SMA(data['close'], 50)) / talib.SMA(data['close'], 50))

        return data


    @staticmethod
    def add_timely_data(data):
        data['DayOfWeek'] = pd.to_datetime(data['date']).dt.dayofweek
        data['Month'] = pd.to_datetime(data['date']).dt.month
        data['Hourly'] = pd.to_datetime(data['date']).dt.hour / 4
        return data
    

    @staticmethod
    def assign_labels(data, b_window, f_window, alpha, beta):
        x = data.copy()
        x['close_MA'] = x['close'].ewm(span=b_window).mean()
        x['s-1'] = x['close'].shift(-1 * f_window)
        x['alpha'] = alpha
        x['beta'] = beta * (1 + (f_window * 0.1))
        x['label'] = x.apply(TecnicalAnalysis.check_label, axis=1)
        return x['label']


    @staticmethod
    def check_label(z):
        if (abs((z['s-1'] - z['close_MA']) / z['close_MA']) > z['alpha']) and \
                (abs((z['s-1'] - z['close_MA']) / z['close_MA']) < (z['beta'])):
            if z['s-1'] > z['close_MA']:
                return -1
            elif z['s-1'] < z['close_MA']:
                return 1
            else:
                return 0
        else:
            return 0


    @staticmethod
    def find_patterns(x):
        x['CDL2CROWS'] = talib.CDL2CROWS(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDL3INSIDE'] = talib.CDL3INSIDE(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLBELTHOLD'] = talib.CDLBELTHOLD(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLDOJISTAR'] = talib.CDLDOJISTAR(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLENGULFING'] = talib.CDLENGULFING(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLhighWAVE'] = talib.CDLhighWAVE(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLHIKKAKE'] = talib.CDLHIKKAKE(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLKICKING'] = talib.CDLKICKING(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLLONGLINE'] = talib.CDLLONGLINE(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLMARUBOZU'] = talib.CDLMARUBOZU(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLMATCHINGlow'] = talib.CDLMATCHINGlow(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLMATHOLD'] = talib.CDLMATHOLD(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLONNECK'] = talib.CDLONNECK(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLPIERCING'] = talib.CDLPIERCING(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLSHORTLINE'] = talib.CDLSHORTLINE(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLTHRUSTING'] = talib.CDLTHRUSTING(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLTRISTAR'] = talib.CDLTRISTAR(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(x['open'], x['high'], x['low'], x['close']) / 100
        x['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(x['open'], x['high'], x['low'], x['close']) / 100
        # x['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(x['open'], x['high'], x['low'], x['close']) / 100
        return x
