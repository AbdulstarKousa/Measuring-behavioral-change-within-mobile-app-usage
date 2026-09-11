import pandas as pd


def create_balance_column(
    account_transactions: pd.DataFrame,
    anchor_date,
    anchor_balance: float
) -> pd.DataFrame:
    """
    Reconstruct account balances backwards from a known balance.

    Parameters
    ----------
    account_transactions : pd.DataFrame
        Transactions for ONE account.
        Required columns:
            - DATO
            - HAVET/INDSAT

    anchor_date :
        Date on which the known anchor_balance applies.

    anchor_balance : float
        Known balance at the anchor_date.

    Returns
    -------
    pd.DataFrame
        Original transactions with an additional BALANCE column.

    Logic
    -----
    Balance at day D =
        anchor_balance
        - sum(all transactions after day D)

    Artificial year 9999 dates are ignored.
    """

    df = account_transactions.copy()

    # ---------------------------------------------------------
    # 1. Prepare dates and amounts
    # ---------------------------------------------------------

    df["DATO"] = pd.to_datetime(
        df["DATO"],
        errors="coerce"
    )

    df["HAVET/INDSAT"] = pd.to_numeric(
        df["HAVET/INDSAT"],
        errors="coerce"
    )

    anchor_date = pd.to_datetime(
        anchor_date,
        errors="coerce"
    )

    # ---------------------------------------------------------
    # 2. Remove invalid data
    # ---------------------------------------------------------

    df = df[
        df["DATO"].notna()
        & df["HAVET/INDSAT"].notna()
    ].copy()

    # Ignore artificial 9999 dates
    df = df[
        df["DATO"].dt.year != 9999
    ].copy()

    # Only transactions up to the anchor date
    df = df[
        df["DATO"] <= anchor_date
    ].copy()

    if df.empty:
        df["BALANCE"] = pd.Series(dtype=float)
        return df

    # ---------------------------------------------------------
    # 3. Calculate net transaction amount per day
    # ---------------------------------------------------------

    daily = (
        df.groupby(
            "DATO",
            sort=True
        )["HAVET/INDSAT"]
        .sum()
        .reset_index(name="DAY_SUM")
    )

    # ---------------------------------------------------------
    # 4. Calculate transactions AFTER each day
    # ---------------------------------------------------------

    future_sum = (
        daily["DAY_SUM"]
        .iloc[::-1]
        .cumsum()
        .iloc[::-1]
        .shift(-1)
        .fillna(0.0)
    )

    # ---------------------------------------------------------
    # 5. Reconstruct balance backwards
    # ---------------------------------------------------------

    daily["BALANCE"] = (
        anchor_balance - future_sum
    )

    # ---------------------------------------------------------
    # 6. Add balance back to transactions
    # ---------------------------------------------------------

    df = df.merge(
        daily[
            [
                "DATO",
                "BALANCE"
            ]
        ],
        on="DATO",
        how="left"
    )

    # Keep original transaction order
    df = df.sort_index()

    return df
