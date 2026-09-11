def classify_from_balance(
    balances,
    dm_date,
):
    """
    Classify based on the reconstructed balance immediately
    before the debt movement.
    """

    dm_date = pd.to_datetime(dm_date)

    if balances.empty:
        return "None"

    # Important:
    # Same-day transactions are treated as part of the DM day.
    before_dm = balances[
        balances["DATO"] < dm_date
    ]

    if before_dm.empty:
        return "None"

    latest_date = before_dm["DATO"].max()

    balance = before_dm.loc[
        before_dm["DATO"] == latest_date,
        "BALANCE"
    ].iloc[0]

    if pd.isna(balance):
        return "None"

    if balance >= 0:
        return "True"



def classify_payment_pre_2016(
    acc_nr,
    dm_date,
    connection,
    get_debt_spans_function,
):

    transactions = get_account_transactions(
        acc_nr,
        connection
    )

    if transactions.empty:
        return classify_old_payment_logic(
            transactions,
            dm_date,
            get_debt_spans_function,
            acc_nr,
        )

    # ---------------------------------------------------------
    # 1. Closed account
    # ---------------------------------------------------------

    if is_account_closed(transactions):

        anchor_date = pd.to_datetime(
            transactions["DATO"],
            errors="coerce"
        ).max()

        balances = create_balance_column(
            transactions,
            anchor_date,
            0.0
        )

        return classify_from_balance(
            balances,
            dm_date
        )

    # ---------------------------------------------------------
    # 2. Historical balance
    # ---------------------------------------------------------

    historical = get_historical_balance_point(
        acc_nr,
        dm_date,
        connection
    )

    if historical is not None:

        balances = create_balance_column(
            transactions,
            historical["date"],
            historical["balance"]
        )

        result = classify_from_balance(
            balances,
            dm_date
        )

        if result != "None":
            return result

    # ---------------------------------------------------------
    # 3. Product-change statistics
    # ---------------------------------------------------------

    stats = get_product_change_stats(
        acc_nr,
        dm_date,
        connection
    )

    if stats is not None:

        result = classify_using_product_change_stats(
            transactions,
            dm_date,
            stats
        )

        if result != "None":
            return result

    # ---------------------------------------------------------
    # 4. UB sum == 0
    # ---------------------------------------------------------

    if account_sum_is_zero(transactions):

        anchor_date = pd.to_datetime(
            transactions["DATO"],
            errors="coerce"
        ).max()

        balances = create_balance_column(
            transactions,
            anchor_date,
            0.0
        )

        result = classify_from_balance(
            balances,
            dm_date
        )

        if result != "None":
            return result

    # ---------------------------------------------------------
    # 5. Old logic
    # ---------------------------------------------------------

    return classify_old_payment_logic(
        transactions,
        dm_date,
        get_debt_spans_function,
        acc_nr,
    )
    
    return "False"
