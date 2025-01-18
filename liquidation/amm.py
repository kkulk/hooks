import numpy as np

class TradeResult:
    def __init__(self, x_out, y_out):
        self.x_out = x_out
        self.y_out = y_out

class AMM:
    def __init__(self, x, y, gamma):
        self.x_reserves = x
        self.y_reserves = y
        self.gamma = gamma

    def trade(self, x_in, y_in):
        # Trades that are tendering to the AMM (at least one of these is zero)
        x_in = np.array(x_in)
        y_in = np.array(y_in)

        # This actually all works just fine if both x_in and y_in are nonzero,
        # but for the purposes of this simulation we're only
        # simulating one-way trades, and using this function for convenience
        # so this is to make sure we're not simulating the wrong thing anywhere
        assert np.sum(x_in) == 0 or np.sum(y_in) == 0

        K = self.x_reserves * self.y_reserves

        if np.sum(x_in) == 0:
            new_y_reserves = self.y_reserves + y_in.sum()
            new_x_reserves = K / new_y_reserves 
            output_x = (new_x_reserves - self.x_reserves)*(1-self.gamma)
            self.x_reserves = (new_x_reserves - self.x_reserves)*(self.gamma) + new_x_reserves
            self.y_reserves = new_y_reserves
            output_y = 0
        if np.sum(y_in) == 0:
            new_x_reserves = self.x_reserves + x_in.sum()
            new_y_reserves = K / new_x_reserves
            output_y = (new_y_reserves - self.y_reserves)*(1-self.gamma)
            self.x_reserves = new_x_reserves
            self.y_reserves = (new_y_reserves - self.y_reserves)*(self.gamma) + new_y_reserves
            output_x = 0

        return TradeResult(output_x, output_y)

    def infinitesimal_trade(self, x_in, y_in):
        # always convert for convenience
        x_in = np.array(x_in)
        y_in = np.array(y_in)

        # Need to check for one-sided orders to avoid math error
        if np.sum(x_in) == 0 or np.sum(y_in) == 0:
            return self.trade(x_in, y_in)

        k = self.x_reserves * self.y_reserves
        c = (
            np.sqrt(self.x_reserves * np.sum(y_in))
            - np.sqrt(self.y_reserves * np.sum(x_in))
        ) / (
            np.sqrt(self.x_reserves * np.sum(y_in))
            + np.sqrt(self.y_reserves * np.sum(x_in))
        )
        exponent = 2 * np.sqrt(np.sum(x_in) * np.sum(y_in) / k)
        end_x_sum = (
            np.sqrt(k * np.sum(x_in) / np.sum(y_in))
            * (np.exp(exponent) + c)
            / (np.exp(exponent) - c)
        )
        end_y_sum = k / end_x_sum

        x_out_sum = self.x_reserves - end_x_sum + np.sum(x_in)
        y_out_sum = self.y_reserves - end_y_sum + np.sum(y_in)

        # Update
        self.x_reserves = end_x_sum
        self.y_reserves = end_y_sum

        x_out = x_out_sum * y_in / np.sum(y_in)
        y_out = y_out_sum * x_in / np.sum(x_in)

        return TradeResult(x_out, y_out)

    def instantaneous_y_price(self):
        return float(self.x_reserves / self.y_reserves)


if __name__ == "__main__":
    pass
