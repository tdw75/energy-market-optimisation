import pandas as pd


def evaluate_pnl(schedule: pd.DataFrame, actual: pd.Series, forecasts: pd.Series) -> pd.DataFrame:
    """
    Evaluates the profit and loss (PnL) of a trading strategy based on the true and predicted values.

    Parameters:
    - schedule: The trading schedule (e.g., buy and sell decisions).
    - prices: The true values (e.g., actual prices).
    - forecasts: The predicted values (e.g., forecasted prices).

    Returns:
    - pnl: The profit and loss from the trading strategy.
    """

    pnl = pd.DataFrame(
        index=schedule.index,
        columns=["pnl_forecast", "pnl_actual"],
    )
    pnl["pnl_actual"] = schedule["sell"] * actual["price"] - schedule["buy"] * actual["price"]
    pnl["pnl_forecast"] = schedule["sell"] * forecasts - schedule["buy"] * forecasts
    pnl["pnl_gap"] = pnl["pnl_forecast"] - pnl["pnl_actual"]

    return pnl