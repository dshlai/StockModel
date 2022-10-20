import numpy as np
import pandas as pd


def mode(a, axis=0):
    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts


class BaselineTrader(object):
    def __init__(self, traing_file="", stats_mode="mean") -> None:
        
        print("\nCurrent stats mode is: {}".format(stats_mode))

        self.train_data = self.get_train_data(traing_file)

        self.opening_avg = np.average(self.train_data[:, 0])
        self.max_avg = np.average(self.train_data[:, 1])
        self.min_avg = np.average(self.train_data[:, 2])
        self.closing_avg = np.average(self.train_data[:, 3])

        self.opening_mode = mode(self.train_data[:, 0])
        self.max_mode = mode(self.train_data[:, 1])
        self.min_mode = mode(self.train_data[:, 2])
        self.closing_mode = mode(self.train_data[:, 3])

        self.opening_med = np.median(self.train_data[:, 0])
        self.max_med = np.median(self.train_data[:, 1])
        self.min_med = np.median(self.train_data[:, 2])
        self.closing_med = np.median(self.train_data[:, 3])

        if stats_mode == "mean":
            self.model_val = np.average([self.opening_avg, self.max_avg, self.min_avg, self.closing_avg])
        
        elif stats_mode == "closing":
            self.model_val = self.closing_avg
        
        elif stats_mode == "opening":
            self.model_val = self.opening_avg

        elif stats_mode == "minmax":
            self.model_val = np.median([np.max(self.train_data[:, 1]), np.min(self.train_data[:, 2])])

        elif stats_mode == "median":
            self.model_val = np.median([self.opening_med, self.max_med, self.min_med, self.closing_med])
        
        else:
            self.model_val = np.average([self.opening_mode, self.max_mode, self.min_mode, self.closing_mode])
      
        self.call_buffer = []
        self.last_actions = [0]

    def show_stat(self):
        o, m, n, c = self.opening_avg, self.min_avg, self.max_avg, self.closing_avg
        print("Average for open, min, max, close are: {}, {}, {}, {}".format(o, m, n, c))

    @property
    def is_holding_stock(self):
        return self.holding_price is not None

    @property
    def is_shorting_stock(self):
        return self.sell_short_price is not None

    def eval_sell(self, eval_price):
        return eval_price > self.model_val
    
    def eval_buy(self, eval_price):
        return eval_price < self.model_val

    @staticmethod
    def extract_day_price_feature(day_data: pd.DataFrame, stats_mode="mean"):
        
        arr = day_data.to_numpy()
        
        if stats_mode == "mean":
            result = np.average(arr)
        else:
            result = np.median(arr)
        
        return result

    def single_call(self, day_data):
        
        # extrace price feature for the day
        day_price_feature = self.extract_day_price_feature(day_data)
        action = 0

        # eval the price feature and use last action buffer to make sure to not make the wrong sell
        if self.eval_sell(day_price_feature) and not self.last_actions[-1] == -1:
            action = -1

        # eval the price feature and use last action buffer to make sure to not make the wrong buy
        elif self.eval_buy(day_price_feature) < self.model_val and not self.last_actions[-1] == 1:
            action = 1                

        # if not holding, make sure the last call is remembered
        if not action == 0:
            self.last_actions.append(action)

        self.call_buffer.append(action)
    
    def initial_call(self, day_feature_price):

        action = 0

        if self.eval_sell(day_feature_price):
            action = -1

        elif self.eval_buy(day_feature_price):
            action = 1

        if not action == 0:
            self.last_actions.append(action)
        
        self.call_buffer.append(action)

    def run(self, test_file: str, output_file: str):
        
        # read the testing data
        df = pd.read_csv(test_file, names=['open', 'max', 'min', 'close'])
                
        # because the distribution of testing price is different from training prices
        # need to move the model value to fix the distribution shift
        self.model_val = self.model_val * (df.iloc[0].mean() / self.model_val)
        
        # initialize the call and last action buffer
        first_day = df.iloc[0]
        self.initial_call(first_day['close'])
        df.drop(0, axis=0, inplace=True)
        
        # after the action buffer is initialized, we can continue the rest of calls
        for _, row in df.iterrows():
            self.single_call(row)
        
        actions = pd.DataFrame(self.call_buffer[:-2])
        actions.to_csv(output_file, index=False)

    @staticmethod
    def get_train_data(training_file):

        train_data = pd.read_csv(training_file)
        return train_data.to_numpy()


if __name__ == '__main__':
 # You should not modify this part.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default='training_data.csv', help='input training data file name')
    parser.add_argument('--testing', default='testing_data.csv', help='input testing data file name')
    parser.add_argument('--output', default='output.csv', help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    
    trader = BaselineTrader(traing_file=args.training, stats_mode="mean")
    trader.run(test_file=args.testing, output_file=args.output)