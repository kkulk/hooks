import numpy as np

class TradeResult:
    def __init__(self, x_out, y_out):
        self.x_out = x_out
        self.y_out = y_out

def trade(self, x_in, y_in):
        # always convert for convenience
        x_in = np.array(x_in)
        y_in = np.array(y_in)

        # This actually all works just fine if both x_in and y_in are nonzero,
        # but for the purposes of this simulation we're only
        # simulating one-way trades, and using this function for convenience
        # so this is to make sure we're not simulating the wrong thing anywhere
        assert np.sum(x_in) == 0 or np.sum(y_in) == 0

        total_x = self.x_reserves + x_in.sum()
        total_y = self.y_reserves + y_in.sum()

        # Trading all of x for all of y and divvying up shares accordingly
        self_x_share = self.x_reserves / total_x
        self_y_share = self.y_reserves / total_y

        x_in_shares = x_in / total_x
        y_in_shares = y_in / total_y

        # Update
        self.x_reserves = self_y_share * total_x
        self.y_reserves = self_x_share * total_y

        x_out = y_in_shares * total_x
        y_out = x_in_shares * total_y

        return TradeResult(x_out, y_out)

def instantaneous_y_price(self):
        return float(self.x_reserves / self.y_reserves)

if __name__ == "__main__":
    pass
