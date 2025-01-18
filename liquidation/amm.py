import numpy as np

class TradeResult:
    def __init__(self, x_out, y_out):
        self.x_out = x_out
        self.y_out = y_out

class AMM:
    def __init__(self, x, y, gamma):
        self.x_reserves = x
        self.y_reserves = y
        self.gamma = gamma/10000

    def trade(self, x_in, y_in):
        # Trades that are tendering to the AMM (at least one of these is zero)
        initial_price = self.instantaneous_x_price()
        x_in = np.array(x_in)
        y_in = np.array(y_in)

        # This actually all works just fine if both x_in and y_in are nonzero,
        # but for the purposes of this simulation we're only
        # simulating one-way trades, and using this function for convenience
        # so this is to make sure we're not simulating the wrong thing anywhere
        assert np.sum(x_in) == 0 or np.sum(y_in) == 0

        K = self.x_reserves * self.y_reserves
        # print(f"\nInitial state:")
        # print(f"K = {K}")
        # print(f"x_reserves = {self.x_reserves}")
        # print(f"y_reserves = {self.y_reserves}")

        if np.sum(x_in) == 0:
            # print(f"\nTrading y_in = {y_in.sum()}")
            new_y_reserves = self.y_reserves + y_in.sum() *(1-self.gamma)
            new_x_reserves = K / new_y_reserves 
            # print(f"new_x_reserves = {new_x_reserves}")
            # print(f"new_y_reserves = {new_y_reserves}")
            
            delta_x = new_x_reserves - self.x_reserves
            # print(f"delta_x = {delta_x}")
            
            output_x = delta_x 
            # print(f"output_x (after fee) = {output_x}")
            
            # Pool keeps the fee portion of delta_x
            self.x_reserves = new_x_reserves
            self.y_reserves = new_y_reserves
            output_y = 0
            
            # print(f"\nFinal state:")
            # print(f"final x_reserves = {self.x_reserves}")
            # print(f"final y_reserves = {self.y_reserves}")
            # print(f"final K = {self.x_reserves * self.y_reserves}")
            # print(f"K diff = {abs(self.x_reserves * self.y_reserves - K)}")

        if np.sum(y_in) == 0:
            # print(f"\nTrading x_in = {x_in.sum()}")
            new_x_reserves = self.x_reserves + x_in.sum() *(1-self.gamma)
            new_y_reserves = K / new_x_reserves
            # print(f"new_x_reserves = {new_x_reserves}")
            # print(f"new_y_reserves = {new_y_reserves}")
            
            delta_y =self.y_reserves - new_y_reserves 
            # print(f"delta_y = {delta_y}")
            
            output_y = delta_y 
            # print(f"output_y (after fee) = {output_y}")
            
            # Pool keeps the fee portion of delta_y
            self.x_reserves = new_x_reserves
            self.y_reserves = new_y_reserves
            output_x = 0
            
            # print(f"\nFinal state:")
            # print(f"final x_reserves = {self.x_reserves}")
            # print(f"final y_reserves = {self.y_reserves}")
            # print(f"final K = {self.x_reserves * self.y_reserves}")
            # print(f"K diff = {abs(self.x_reserves * self.y_reserves - K)}")

        assert abs(self.x_reserves*self.y_reserves - K) < 1e-6

        average_price = (self.instantaneous_x_price() + initial_price)/2

        return (TradeResult(output_x, output_y), average_price) 
    
    def arb(self, z):
        # z is the mispricing level
        # return the arbitrage opportunity
        K = self.x_reserves * self.y_reserves
        starting_x_price = self.instantaneous_x_price()
        exchange_price = starting_x_price * np.exp(z)

        # arbitraging the pool based on the constant exchange price
        # if z > gamma, then we can arbitrage by buying x and selling y
        if z > self.gamma:
            final_pool_price = exchange_price * np.exp(-self.gamma)
            x_final = np.sqrt(K/final_pool_price)
            y_final = K/x_final
            self.x_reserves = x_final
            self.y_reserves = y_final
        # if z < -gamma, then we can arbitrage by selling x and buying y
        elif z < -self.gamma:
            final_pool_price = exchange_price * np.exp(self.gamma)
            x_final = np.sqrt(K/final_pool_price)
            y_final = K/x_final
            self.x_reserves = x_final
            self.y_reserves = y_final


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

    def instantaneous_x_price(self):
        return float(self.y_reserves / self.x_reserves)

def test_amm():
    # Test initialization
    def test_init():
        amm = AMM(100, 100, 100)  # 100 bps = 1% fee
        assert amm.x_reserves == 100
        assert amm.y_reserves == 100
        assert amm.gamma == 0.01
        print("✓ Init test passed")

    # Test basic price calculations
    def test_prices():
        amm = AMM(100, 200, 100)
        assert amm.instantaneous_x_price() == 2.0  # 200/100
        assert amm.instantaneous_y_price() == 0.5  # 100/200
        print("✓ Price calculation test passed")

    # Test simple trade with no fees
    def test_trade_no_fees():
        amm = AMM(1000, 1000, 0)
        result, avg_price = amm.trade(100, 0)  # Add 100 x tokens
        assert abs(amm.x_reserves - 1100) < 1e-10
        assert abs(amm.y_reserves - 909.0909091) < 1e-6  # 1000000/1100
        print("✓ No-fee trade test passed")

    # Test trade with fees
    def test_trade_with_fees():
        amm = AMM(1000, 1000, 100)  # 1% fee
        result, avg_price = amm.trade(100, 0)
        # Check that 1% of output is kept as fee
        assert abs(result.y_out * 0.99) == abs(result.y_out * (1 - 0.01))
        print("✓ Fee calculation test passed")

    # Test constant product invariant
    def test_constant_product():
        amm = AMM(1000, 1000, 100)
        initial_k = amm.x_reserves * amm.y_reserves
        amm.trade(100, 0)
        final_k = amm.x_reserves * amm.y_reserves
        assert abs(initial_k - final_k) < 1e-6
        print("✓ Constant product invariant test passed")

    # Test arbitrage function
    def test_arbitrage():
        # Test basic arbitrage with positive mispricing
        def test_positive_arb():
            amm = AMM(1000, 1000, 100)  # 1% fee
            initial_price = amm.instantaneous_x_price()
            initial_k = amm.x_reserves * amm.y_reserves
            
            z = 0.02  # 200 bps, greater than 100 bps fee
            amm.arb(z)
            new_price = amm.instantaneous_x_price()
            
            # Check price adjustment
            assert abs(np.log(new_price/initial_price) - (z - amm.gamma)) < 1e-6
            # Check constant product maintained
            assert abs(amm.x_reserves * amm.y_reserves - initial_k) < 1e-6
            print("✓ Positive arbitrage test passed")

        # Test basic arbitrage with negative mispricing
        def test_negative_arb():
            amm = AMM(1000, 1000, 100)
            initial_price = amm.instantaneous_x_price()
            initial_k = amm.x_reserves * amm.y_reserves
            
            z = -0.02  # -200 bps
            amm.arb(z)
            new_price = amm.instantaneous_x_price()
            
            # Check price adjustment
            assert abs(np.log(new_price/initial_price) - (z + amm.gamma)) < 1e-6
            # Check constant product maintained
            assert abs(amm.x_reserves * amm.y_reserves - initial_k) < 1e-6
            print("✓ Negative arbitrage test passed")

        # Test no arbitrage when within fee bounds
        def test_no_arb_within_bounds():
            amm = AMM(1000, 1000, 100)
            initial_x = amm.x_reserves
            initial_y = amm.y_reserves
            
            # Test mispricing less than fee
            z = 0.005  # 50 bps, less than 100 bps fee
            amm.arb(z)
            
            # Should not change reserves
            assert abs(amm.x_reserves - initial_x) < 1e-6
            assert abs(amm.y_reserves - initial_y) < 1e-6
            print("✓ No arbitrage within bounds test passed")

        # Test arbitrage with asymmetric reserves
        def test_asymmetric_arb():
            amm = AMM(2000, 1000, 100)  # 2:1 ratio
            initial_price = amm.instantaneous_x_price()
            initial_k = amm.x_reserves * amm.y_reserves
            
            z = 0.02
            amm.arb(z)
            new_price = amm.instantaneous_x_price()
            
            # Check price adjustment
            assert abs(np.log(new_price/initial_price) - (z - amm.gamma)) < 1e-6
            # Check constant product maintained
            assert abs(amm.x_reserves * amm.y_reserves - initial_k) < 1e-6
            print("✓ Asymmetric reserves arbitrage test passed")

        # Test sequence of arbitrages
        def test_sequential_arb():
            amm = AMM(1000, 1000, 100)
            initial_k = amm.x_reserves * amm.y_reserves
            
            # Perform sequence of arbitrages
            zs = [0.02, -0.02, 0.015, -0.015]
            for z in zs:
                amm.arb(z)
                # Check constant product maintained after each arb
                assert abs(amm.x_reserves * amm.y_reserves - initial_k) < 1e-6
            print("✓ Sequential arbitrage test passed")

        # Run all arbitrage tests
        print("\nRunning arbitrage tests...")
        test_positive_arb()
        test_negative_arb()
        test_no_arb_within_bounds()
        test_asymmetric_arb()
        test_sequential_arb()
        print("All arbitrage tests passed!")

    # Test sequence of trades
    def test_trade_sequence():
        amm = AMM(1000, 1000, 100)
        initial_k = amm.x_reserves * amm.y_reserves
        
        # Perform a sequence of trades
        trades = [(100, 0), (0, 50), (75, 0)]
        for x_in, y_in in trades:
            amm.trade(x_in, y_in)
            current_k = amm.x_reserves * amm.y_reserves
            assert current_k >= initial_k  # K should never decrease
        print("✓ Trade sequence test passed")

    # Run all tests
    print("Running AMM tests...")
    test_init()
    test_prices()
    test_trade_no_fees()
    test_trade_with_fees()
    test_constant_product()
    test_arbitrage()  # This will now run all the arbitrage tests
    test_trade_sequence()
    print("All tests passed!")

if __name__ == "__main__":
    test_amm()
