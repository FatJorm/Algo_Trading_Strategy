import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV as rcv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer


class Strategies:
    def __init__(self, omx30):
        self.omx30 = omx30#OMX30_ST(update=update)
        self.companies = self.omx30.companies
        self.stock = self.omx30.stock
        self.strategies = [self.trend, self.volume, self.bolling_band_method, self.stochastic_method, self.add_signal]
        self.add_strategies_to_stock()

    def add_strategies_to_stock(self):
        for company in self.companies:
            for strategy in self.strategies:
                strategy(self.stock[company])

    @staticmethod
    def trend(df):
        short_window = 20
        long_window = 55
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0.0
        signals['Adj Close'] = df['Adj Close'].interpolate()
        signals['short_mavg'] = df['Adj Close'].interpolate().rolling(window=short_window, min_periods=1,
                                                                      center=False).mean()
        signals['long_mavg'] = df['Adj Close'].interpolate().rolling(window=long_window, min_periods=1,
                                                                     center=False).mean()
        signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]
                                                    > signals['long_mavg'][short_window:], 1.0, 0.0)
        df['trend'] = signals['signal']

    @staticmethod
    def volume(df):
        for i in range(1, len(df['Volume'])):
            if df['Volume'][i] > df['Volume'][i-1]:
                df.ix[i, 'volyme_direction'] = 1.0
            else:
                df.ix[i, 'volyme_direction'] = 0.0

    @staticmethod
    def stochastic_method(df):
        window = 14
        signals = pd.DataFrame(index=df.index)
        signals['high_low'] = 0.0
        signals['D_K'] = 0.0
        signals['signal'] = 0.0
        signals['Adj Close'] = df['Adj Close'].interpolate()
        signals['positions'] = signals['signal'].diff()
        signals['L14'] = df['Low'].interpolate().rolling(window=window).min()
        signals['H14'] = df['High'].interpolate().rolling(window=window).max()
        signals['%K'] = 100*((df['Close'] - signals['L14']) / (signals['H14'] - signals['L14']) )
        signals['%D'] = signals['%K'].rolling(window=3).mean()
        signals['Sell Entry'] = ((signals['%K'] < signals['%D']) & (signals['%K'].shift(1) > signals['%D'].shift(1))) & (signals['%D'] > 80)
        signals['Sell Exit'] = ((signals['%K'] > signals['%D']) & (signals['%K'].shift(1) < signals['%D'].shift(1)))
        signals['Short'] = np.nan
        signals.loc[signals['Sell Entry'],'Short'] = 0
        signals.loc[signals['Sell Exit'],'Short'] = 0
        signals.loc[0, 'Short'] = 0
        signals['Short'] = signals['Short'].fillna(method='pad')
        signals['Buy Entry'] = ((signals['%K'] < signals['%D']) & (signals['%K'].shift(1) < signals['%D'].shift(1))) & (signals['%D'] < 20)
        signals['Buy Exit'] = ((signals['%K'] < signals['%D']) & (signals['%K'].shift(1) > signals['%D'].shift(1)))
        signals['Long'] = np.nan
        signals.loc[signals['Buy Entry'],'Long'] = 1
        signals.loc[signals['Buy Exit'],'Long'] = 0
        signals.loc[0,'Long'] = 0
        signals['Long'] = signals['Long'].fillna(method='pad')
        signals['signal'] = signals['Long'] + signals['Short']
        #for i in range(len(signals['signal'])):
        #    if signals.ix[i, 'signal'] == 0.0:
        #       signals.ix[i, 'signal'] = -1.0
        df['stochastic_signal'] = signals['signal']

    @staticmethod
    def bolling_band_method(df):
        window = 20
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0.0
        signals['Adj Close'] = df['Adj Close'].interpolate()
        signals['20 ma'] = df['Adj Close'].interpolate().rolling(window=window, min_periods=1, center=False).mean()
        signals['20 sd'] = df['Adj Close'].interpolate().rolling(window=window, min_periods=1, center=False).std()
        signals['Upper Band'] = signals['20 ma'] + (signals['20 sd']*2)
        signals['Lower Band'] = signals['20 ma'] - (signals['20 sd']*2)
        signals['signal'][window:] = np.where(signals['Adj Close'][window:]
                                              > signals['Upper Band'][window:], 0.0, np.where(signals['Adj Close'][window:]
                                              < signals['Lower Band'][window:], 1.0, -1.0))
        for i in range(window, len(signals['signal'])):
            if signals['signal'][i] == -1.0:
                signals['signal'][i] = signals['signal'][i-1]
        df['bolling_signal'] = signals['signal']

    @staticmethod
    def machine_learn(df):
        df.dropna(inplace=True)
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        steps = [('imputation', imp),
                 ('scaler', StandardScaler()),
                 ('lasso', Lasso())]
        pipeline = Pipeline(steps)
        parameters = {'lasso__alpha': np.arange(0.0001, 10, .0001),
                      'lasso__max_iter': np.random.uniform(100, 100000, 4)}
        reg = rcv(pipeline, parameters, cv=5)
        X = df[['trend', 'bolling_signal', 'stochastic_signal', 'volyme_direction']]
        y = df['Close']
        avg_err = {}
        for t in np.arange(50, 97, 3):
            split = int(t*len(X)/100)
            reg.fit(X[:split], y[:split])
            best_alpha = reg.best_params_['lasso__alpha']
            best_iter = reg.best_params_['lasso__max_iter']
            reg1 = Lasso(alpha=best_alpha, max_iter=best_iter)
            reg1.fit(X[:split], y[:split])
            df['P_C_%i' % t] = 0
            df.iloc[split:, df.columns.get_loc('P_C_%i' % t)] = reg1.predict(X[split:])
            df['Error_%i' % t] = np.abs(df['P_C_%i' % t] - df['Close'])
            e = np.mean(df['Error_%i' % t][split:])
            avg_err[t] = e

            Range = df['High'].shift(1)[split:]-df['Low'].shift(1)[split:]
            plt.scatter(avg_err.keys(), avg_err.values())
            print('\nAverage Range of the Day:', np.average(Range))

    @staticmethod
    def add_signal(df):
        df['signal'] = 0
        for i in range(len(df['signal'])):

            #All
            #if df.ix[i, 'bolling_signal'] == 1.0 and df.ix[i, 'stochastic_signal'] == 1.0 and df.ix[i, 'trend'] == 1.0:
            #    df.ix[i, 'signal'] = 1.0

            #Bolling
            if df.ix[i, 'bolling_signal'] == 1.0:
                df.ix[i, 'signal'] = 1.0

            #Bolling and Trend
            #if df.ix[i, 'bolling_signal'] == 1.0 and df.ix[i, 'trend'] == 1.0:
            # df.ix[i, 'signal'] = 1.0

            #Stochastic
            #if df.ix[i, 'stochastic_signal'] == 1.0 :
            #    df.ix[i, 'signal'] = 1.0

            #Trend
            #if  df.ix[i, 'trend'] == 1.0:
            #    df.ix[i, 'signal'] = 1.0
            else:
                df.ix[i, 'signal'] = 0.0

            if df.ix[i, 'signal'] == 1.0 and df.ix[i-1, 'signal'] == 0.0:
                df.ix[i, 'signal'] = df.ix[i, 'volyme_direction']


#df.ix[i, 'volyme_direction'] == 1.0

if __name__ == '__main__':
    None





