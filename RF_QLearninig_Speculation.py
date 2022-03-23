import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from collections import deque
import random
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

name = 'Q-learning agent_v2'
class Agent:
    def __init__(self, state_size, window_size, trend, skip, batch_size):
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.skip = skip
        self.action_size = 3 # LIczba możliwych akcji do wykonania
        self.batch_size = batch_size
        
        self.portfolio_volume = 0 # wolumen otwartej pozycji na giełdzie
        self.portfolio_value = 0 # wartość wolumenu otwartej pozycji

        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.X = tf.placeholder(tf.float32, [None, self.state_size])
        self.Y = tf.placeholder(tf.float32, [None, self.action_size])
        feed = tf.layers.dense(self.X, 256, activation = tf.nn.relu)
        self.logits = tf.layers.dense(feed, self.action_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.GradientDescentOptimizer(1e-5).minimize(
            self.cost
        )
        self.sess.run(tf.global_variables_initializer())

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(
            self.sess.run(self.logits, feed_dict = {self.X: state})[0]
        )

    def get_state(self, t):
        '''
        Get information about Market at state t
        '''
        window_size = self.window_size + 1
        d = t - window_size + 1
        if (d >= 0):
            block = self.trend[d : t + 1]
        else:
            block =  -d * [self.trend[0]] + self.trend[0 : t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res])

    def replay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size, l):
            mini_batch.append(self.memory[i])
        replay_size = len(mini_batch)

        X = np.empty((replay_size, self.state_size))
        Y = np.empty((replay_size, self.action_size))

        states = np.array([a[0][0] for a in mini_batch])
        new_states = np.array([a[3][0] for a in mini_batch])
        Q = self.sess.run(self.logits, feed_dict = {self.X: states})
        Q_new = self.sess.run(self.logits, feed_dict = {self.X: new_states})
        for i in range(len(mini_batch)):
            state, action, reward, next_state, done = mini_batch[i]
            target = Q[i]
            target[action] = reward
            if not done:
                target[action] += self.gamma * np.amax(Q_new[i])
            X[i] = state
            Y[i] = target
        cost, _ = self.sess.run(
            [self.cost, self.optimizer], feed_dict = {self.X: X, self.Y: Y}
        )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return cost

    def test(self, initial_money):
        starting_money = initial_money
        states_sell = []
        states_buy = []
        current_money = starting_money
        portfolio_volume = 0
        portfolio_value = 0
        state = self.get_state(0)

            
        for t in range(0, len(self.trend) - 1, self.skip):
            MAX_OPEN_POSITION = 10
            left_transaction_days = len(self.trend) - t
            if left_transaction_days < self.window_size:
                limit_open_position = MAX_OPEN_POSITION // (self.window_size - left_transaction_days) 
            else:
                limit_open_position = MAX_OPEN_POSITION
            
            action = self.act(state)
            next_state = self.get_state(t + 1)

            # Action Buy
            if (action == 1) and (portfolio_volume < limit_open_position):
                buy_volume = 1
                states_buy.append(t)
                current_money -= self.trend[t] * buy_volume
                # change price of portfolio
                if portfolio_volume >= 0:
                    portfolio_volume += buy_volume
                    portfolio_value += self.trend[t] * buy_volume
                    portfolio_price = portfolio_value / portfolio_volume
                elif (portfolio_volume + buy_volume) < 0:
                    portfolio_value = portfolio_value * (1 + buy_volume / portfolio_volume)
                    portfolio_volume += buy_volume
                elif (portfolio_volume + buy_volume) > 0:
                    portfolio_value = self.trend[t] * (portfolio_volume + buy_volume)
                    portfolio_volume += buy_volume
                    portfolio_price = portfolio_value / portfolio_volume
                else:
                    portfolio_value = 0 
                    portfolio_volume = 0
                    portfolio_price = None
                
            
            # Action Sell
            elif action == 2 and ((-1 * portfolio_volume) < limit_open_position):
                sell_volume = -1
                states_sell.append(t)
                current_money -= self.trend[t] * sell_volume
                # change price of portfolio
                if portfolio_volume <= 0:
                    portfolio_volume += sell_volume
                    portfolio_value += self.trend[t] * sell_volume
                    portfolio_price = portfolio_value / portfolio_volume
                elif (portfolio_volume + sell_volume) < 0:
                    portfolio_value = portfolio_value * (1 + sell_volume / portfolio_volume)
                    portfolio_volume += sell_volume
                elif (portfolio_volume + sell_volume) > 0:
                    portfolio_value = self.trend[t] * (portfolio_volume + sell_volume)
                    portfolio_volume += sell_volume
                    portfolio_price = portfolio_value / portfolio_volume
                else:
                    portfolio_value = 0 
                    portfolio_volume = 0
                    portfolio_price = None

        
            if portfolio_volume < 0:
                portoflio_value_market = portfolio_volume * self.trend[t] * 1.01
            else: 
                portoflio_value_market = portfolio_volume * self.trend[t] * 0.99
                    
            total_profit = current_money - initial_money + portoflio_value_market
            invest = (total_profit / initial_money) * 100
            print(f'Action: {action} ## Bilans: {current_money},{total_profit}, {portoflio_value_market} Current Portfel: {portfolio_volume}, with value {portfolio_value}; Market Price {self.trend[t]}')
            state = next_state
        
        return states_buy, states_sell, total_profit, invest

    def train(self, iterations, checkpoint, initial_money):
        for i in range(iterations):
            total_profit = 0
            state = self.get_state(0)
            current_money = initial_money
            
            portfolio_volume = 0
            portfolio_value = 0
            
            for t in range(0, len(self.trend) - 1, self.skip):
                MAX_OPEN_POSITION = 10
                left_transaction_days = len(self.trend) - t
                if left_transaction_days < self.half_window:
                    limit_open_position = MAX_OPEN_POSITION // (self.half_window - left_transaction_days) 
                else:
                    limit_open_position = MAX_OPEN_POSITION
                
                action = self.act(state)
                next_state = self.get_state(t + 1)

                # Action Buy
                if (action == 1) and (portfolio_volume < limit_open_position):
                    buy_volume = 1
                    current_money -= self.trend[t] * buy_volume
                    # change price of portfolio
                    if portfolio_volume >= 0:
                        portfolio_volume += buy_volume
                        portfolio_value += self.trend[t] * buy_volume
                        portfolio_price = portfolio_value / portfolio_volume
                    elif (portfolio_volume + buy_volume) < 0:
                        portfolio_value = portfolio_value * (1 + buy_volume / portfolio_volume)
                        portfolio_volume += buy_volume
                    elif (portfolio_volume + buy_volume) > 0:
                        portfolio_value = self.trend[t] * (portfolio_volume + buy_volume)
                        portfolio_volume += buy_volume
                        portfolio_price = portfolio_value / portfolio_volume
                    else:
                        portfolio_value = 0 
                        portfolio_volume = 0
                        portfolio_price = None
                    
                
                # Action Sell
                elif action == 2 and ((-1 * portfolio_volume) < limit_open_position):
                    sell_volume = -1
                    current_money -= self.trend[t] * sell_volume
                    # change price of portfolio
                    if portfolio_volume <= 0:
                        portfolio_volume += sell_volume
                        portfolio_value += self.trend[t] * sell_volume
                        portfolio_price = portfolio_value / portfolio_volume
                    elif (portfolio_volume + sell_volume) < 0:
                        portfolio_value = portfolio_value * (1 + sell_volume / portfolio_volume)
                        portfolio_volume += sell_volume
                    elif (portfolio_volume + sell_volume) > 0:
                        portfolio_value = self.trend[t] * (portfolio_volume + sell_volume)
                        portfolio_volume += sell_volume
                        portfolio_price = portfolio_value / portfolio_volume
                    else:
                        portfolio_value = 0 
                        portfolio_volume = 0
                        portfolio_price = None
                                
                if portfolio_volume < 0:
                    portoflio_value_market = portfolio_volume * self.trend[t] * 1.01
                else: 
                    portoflio_value_market = portfolio_volume * self.trend[t] * 0.99

                total_profit = current_money - initial_money + portoflio_value_market
                local_profit = portoflio_value_market - portfolio_value

                invest = (total_profit / initial_money)
                
                self.memory.append((state, action, local_profit, 
                                    next_state, local_profit < 0))

                state = next_state
                batch_size = min(self.batch_size, len(self.memory))
                cost = self.replay(batch_size)

            if (i+1) % checkpoint == 0:
                print('epoch: %d, total rewards: %f.3, cost: %f.3, total money: %f.3'%(i + 1, total_profit, cost,
                                                                                  current_money))
                print(f'{portfolio_volume}, {portfolio_value}')


from google.cloud import bigquery

project_id = 'pl-ist-global-trading-dev'
RTT_query = """
    SELECT 
        ProductName,
        Price_PLNperMWh,
        Volume_MWh,
        TradedDatePL
    FROM `pl-ist-global-trading-dev.TGE.vPowerMarket_RTT`
    WHERE ProductName like "BASE_Y%"
    """

client = bigquery.Client(project=project_id)
df = client.query(RTT_query).to_dataframe()
year_21_df = df[ (df['ProductName'] == "BASE_Y-21") & (df['Volume_MWh'] > 0) ].set_index('TradedDatePL').sort_index()
close = year_21_df['Price_PLNperMWh'].values.tolist()

INITIAL_MONEY = 10e6
initial_money = INITIAL_MONEY / 8760
window_size = 10
skip = 1
batch_size = 32
agent = Agent(state_size = window_size, 
              window_size = window_size, 
              trend = close, 
              skip = skip, 
              batch_size = batch_size)

agent.train(iterations = 100, checkpoint = 10, initial_money = initial_money)

states_buy, states_sell, total_gains, invest = agent.test(initial_money = initial_money)

fig = plt.figure(figsize = (15,5))
plt.plot(close, color='r', lw=2.)
plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
plt.legend()
plt.savefig(name+'.png')
plt.show()