import pandas as pd


def create_balance_column(
    account_transactions: pd.DataFrame,
    anchor_date,
    anchor_balance: float,
    date_col: str = "DATO",
    amount_col: str = "HAVET/INDSAT",
) -> pd.DataFrame:
    """
    Fill a BALANCE column using one known balance as an anchor.

    The known anchor_balance is assumed to be the balance at the
    END of anchor_date.

    Balances are then reconstructed:

        Before anchor:
            balance(day) =
                anchor_balance
                - sum(transactions between day and anchor)

        After anchor:
            balance(day) =
                anchor_balance
                + sum(transactions between anchor and day)

    The original DataFrame is preserved:
        - no rows are dropped
        - no rows are removed because of invalid dates
        - no rows are removed because of 9999 dates
        - original row order is preserved
        - original columns are preserved

    Only valid dated/numeric transactions participate in the
    balance calculation.

    Parameters
    ----------
    account_transactions : pd.DataFrame
        Transactions for one account.

    anchor_date :
        Date where the known balance applies.

    anchor_balance : float
        Known balance at the END of anchor_date.

    date_col : str
        Transaction date column.

    amount_col : str
        Transaction amount column.

    Returns
    -------
    pd.DataFrame
        Original transactions with BALANCE added.
    """

    # ---------------------------------------------------------
    # 1. NEVER modify the original dataframe
    # ---------------------------------------------------------

    df = account_transactions.copy()

    # Always create the output column
    df["BALANCE"] = pd.NA

    # ---------------------------------------------------------
    # 2. Prepare a calculation-only copy
    #
    # IMPORTANT:
    # We do NOT drop anything from df.
    # This temporary dataframe is only for calculation.
    # ---------------------------------------------------------

    calc = df[[date_col, amount_col]].copy()

    calc["_original_index"] = calc.index

    calc["_date"] = pd.to_datetime(
        calc[date_col],
        errors="coerce"
    )

    calc["_amount"] = pd.to_numeric(
        calc[amount_col],
        errors="coerce"
    )

    anchor_date = pd.to_datetime(
        anchor_date,
        errors="coerce"
    )

    if pd.isna(anchor_date):
        return df

    # ---------------------------------------------------------
    # 3. Only valid rows participate in calculation
    #
    # The original rows remain untouched in df.
    # ---------------------------------------------------------

    valid = calc[
        calc["_date"].notna()
        & calc["_amount"].notna()
    ].copy()

    if valid.empty:
        return df

    # ---------------------------------------------------------
    # 4. Aggregate transactions per day
    #
    # This avoids problems when several transactions happen
    # on the same date.
    # ---------------------------------------------------------

    daily = (
        valid.groupby(
            "_date",
            sort=True
        )["_amount"]
        .sum()
        .rename("DAY_SUM")
        .to_frame()
    )

    # ---------------------------------------------------------
    # 5. Make sure anchor date exists in the daily timeline
    #
    # If there are no transactions exactly on anchor_date,
    # we can still use the anchor as a balance point.
    # ---------------------------------------------------------

    all_dates = daily.index.union(
        pd.DatetimeIndex([anchor_date])
    ).sort_values()

    daily = daily.reindex(
        all_dates,
        fill_value=0.0
    )

    # ---------------------------------------------------------
    # 6. Set the known anchor balance
    # ---------------------------------------------------------

    daily["BALANCE"] = pd.NA

    daily.loc[
        anchor_date,
        "BALANCE"
    ] = float(anchor_balance)

    # ---------------------------------------------------------
    # 7. FORWARD calculation
    #
    # Balance next day =
    #     current balance + next day's transactions
    # ---------------------------------------------------------

    dates = daily.index

    anchor_position = dates.get_loc(anchor_date)

    balance = float(anchor_balance)

    for i in range(
        anchor_position + 1,
        len(dates)
    ):

        balance += float(
            daily.iloc[i]["DAY_SUM"]
        )

        daily.iloc[i, daily.columns.get_loc("BALANCE")] = balance

    # ---------------------------------------------------------
    # 8. BACKWARD calculation
    #
    # Balance previous day =
    #     current balance - current day's transactions
    #
    # Example:
    #
    # previous balance + today's transactions
    #     = today's balance
    #
    # therefore:
    #
    # previous balance
    #     = today's balance - today's transactions
    # ---------------------------------------------------------

    balance = float(anchor_balance)

    for i in range(
        anchor_position - 1,
        -1,
        -1
    ):

        # Transactions on the NEXT day move the balance
        # from previous day to next day.
        next_day_sum = float(
            daily.iloc[i + 1]["DAY_SUM"]
        )

        balance -= next_day_sum

        daily.iloc[
            i,
            daily.columns.get_loc("BALANCE")
        ] = balance

    # ---------------------------------------------------------
    # 9. Map daily balance back to EVERY original transaction
    #
    # No rows are removed.
    # Multiple transactions on the same date receive the
    # balance calculated for that date.
    # ---------------------------------------------------------

    balance_map = daily["BALANCE"]

    for idx in valid.index:

        transaction_date = valid.loc[
            idx,
            "_date"
        ]

        df.loc[
            idx,
            "BALANCE"
        ] = balance_map.loc[
            transaction_date
        ]

    # ---------------------------------------------------------
    # 10. Return original structure/order
    # ---------------------------------------------------------

    return df
