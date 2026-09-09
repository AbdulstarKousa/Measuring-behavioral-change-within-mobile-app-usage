import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DM_CUTOFF_DATE = pd.Timestamp("2016-01-01")

# Used only by the old fallback logic
OLD_PAYMENT_WINDOW_DAYS = 5

# Financial tolerance for "sum == 0"
ZERO_TOLERANCE = 0.01


# ============================================================
# 1. GENERAL HELPERS
# ============================================================

def parse_dates(df, columns):
    """
    Convert selected columns to datetime.
    Invalid dates become NaT.
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

    return df


def is_zero(value, tolerance=ZERO_TOLERANCE):
    """
    Check whether a numeric value is effectively zero.
    """
    if pd.isna(value):
        return False

    return bool(
        np.isclose(
            float(value),
            0.0,
            atol=tolerance
        )
    )


# ============================================================
# 2. ACCOUNT TRANSACTIONS
# ============================================================

def get_account_transactions(
    acc_nr,
    connection,
):
    """
    Get transaction history for one account.

    IMPORTANT:
    Replace the table and column names below with the actual
    transaction source used in your project.

    Required output columns:

        ktonr
        DATO
        HAVET/INDSAT
    """

    query = f"""
        SELECT
            ktonr,
            DATO,
            "HAVET/INDSAT"
        FROM datalabs.aragorn_core.YOUR_TRANSACTION_TABLE
        WHERE ktonr = '{acc_nr}'
          AND DATO IS NOT NULL
        ORDER BY DATO
    """

    df = pd.read_sql(query, connection)

    return df


# ============================================================
# 3. CLEAN TRANSACTION HISTORY
# ============================================================

def prepare_account_transactions(
    account_transactions,
):
    """
    Prepare transaction history for balance reconstruction.
    """

    df = account_transactions.copy()

    if df.empty:
        return df

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    df["DATO"] = pd.to_datetime(
        df["DATO"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    df["HAVET/INDSAT"] = pd.to_numeric(
        df["HAVET/INDSAT"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid records
    # --------------------------------------------------------

    df = df[
        df["DATO"].notna()
        & df["HAVET/INDSAT"].notna()
    ].copy()

    # --------------------------------------------------------
    # Ignore artificial 9999 date
    # --------------------------------------------------------

    df = df[
        df["DATO"].dt.year != 9999
    ].copy()

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        "DATO"
    ).reset_index(drop=True)

    return df


# ============================================================
# 4. CHECK ZERO RECONCILIATION
# ============================================================

def account_sum_is_zero(
    account_transactions,
    tolerance=ZERO_TOLERANCE,
):
    """
    Check whether all valid transaction amounts reconcile to zero.

    This is the final fallback condition before using the old
    payment logic.
    """

    df = prepare_account_transactions(
        account_transactions
    )

    if df.empty:
        return False

    total = df["HAVET/INDSAT"].sum()

    return is_zero(
        total,
        tolerance=tolerance
    )


# ============================================================
# 5. ACCOUNT CLOSED
# ============================================================

def is_account_closed(
    account_transactions,
):
    """
    Determine whether the account can be treated as closed
    for balance reconstruction.

    IMPORTANT:
    This is deliberately separate from account_sum_is_zero().

    A sample account may not formally be closed while still
    reconciling to zero.
    """

    df = prepare_account_transactions(
        account_transactions
    )

    if df.empty:
        return False

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Replace this with the actual account-closure indicator
    # if one exists in your transaction/account source.
    #
    # For now we use zero reconciliation as a conservative
    # proxy.
    # --------------------------------------------------------

    return account_sum_is_zero(df)


# ============================================================
# 6. RECONSTRUCT BALANCE BACKWARD
# ============================================================

def create_balance_column(
    account_transactions,
    anchor_date,
    anchor_balance,
):
    """
    Reconstruct historical daily balances backwards from
    a known anchor balance.

    Example:

        anchor_date    = 2020-01-01
        anchor_balance = 0

    The function calculates the balance at the end of each
    transaction date.

    Formula:

        balance(day)
            =
        anchor_balance
        -
        sum(transactions AFTER day)
    """

    df = prepare_account_transactions(
        account_transactions
    )

    if df.empty:
        return df

    anchor_date = pd.to_datetime(
        anchor_date,
        errors="coerce"
    )

    if pd.isna(anchor_date):
        return pd.DataFrame()

    # --------------------------------------------------------
    # Only transactions up to anchor date
    # --------------------------------------------------------

    df = df[
        df["DATO"] <= anchor_date
    ].copy()

    if df.empty:
        return df

    # --------------------------------------------------------
    # One net transaction amount per day
    # --------------------------------------------------------

    daily = (
        df.groupby(
            "DATO",
            sort=True
        )["HAVET/INDSAT"]
        .sum()
        .reset_index(name="DAY_SUM")
    )

    # --------------------------------------------------------
    # Sum of transactions AFTER each day
    # --------------------------------------------------------

    future_sum = (
        daily["DAY_SUM"]
        .iloc[::-1]
        .cumsum()
        .iloc[::-1]
        .shift(-1)
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Reconstruct balance
    # --------------------------------------------------------

    daily["BALANCE"] = (
        float(anchor_balance)
        - future_sum.values
    )

    # --------------------------------------------------------
    # Merge balance back to transactions
    # --------------------------------------------------------

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

    return df


# ============================================================
# 7. RECONSTRUCT FROM CLOSED ACCOUNT
# ============================================================

def calculate_backward_from_closed_account(
    account_transactions,
):
    """
    Closed account:

        terminal balance = 0

    Reconstruct balances backwards.
    """

    df = prepare_account_transactions(
        account_transactions
    )

    if df.empty:
        return df

    anchor_date = df["DATO"].max()

    return create_balance_column(
        account_transactions=df,
        anchor_date=anchor_date,
        anchor_balance=0.0,
    )


# ============================================================
# 8. RECONSTRUCT FROM ZERO-SUM UB ACCOUNT
# ============================================================

def calculate_backward_from_zero_sum(
    account_transactions,
):
    """
    UB sum == zero:

        terminal balance = 0

    Reconstruct balances backwards.
    """

    df = prepare_account_transactions(
        account_transactions
    )

    if df.empty:
        return df

    if not account_sum_is_zero(df):
        return pd.DataFrame()

    anchor_date = df["DATO"].max()

    return create_balance_column(
        account_transactions=df,
        anchor_date=anchor_date,
        anchor_balance=0.0,
    )


# ============================================================
# 9. GET BALANCE BEFORE DM
# ============================================================

def get_balance_before_dm(
    balance_transactions,
    dm_date,
):
    """
    Find the reconstructed balance immediately before
    the debt movement.

    Transactions on the same date as the DM are NOT included.

    This follows the agreed interpretation:

        DATO < DM date
    """

    dm_date = pd.to_datetime(
        dm_date,
        errors="coerce"
    )

    if pd.isna(dm_date):
        return None

    if balance_transactions.empty:
        return None

    before_dm = balance_transactions[
        balance_transactions["DATO"] < dm_date
    ].copy()

    if before_dm.empty:
        return None

    latest_date = before_dm["DATO"].max()

    balance = before_dm.loc[
        before_dm["DATO"] == latest_date,
        "BALANCE"
    ].iloc[0]

    return {
        "balance_before_dm": float(balance),
        "balance_date": latest_date,
        "days_before_dm": (
            dm_date - latest_date
        ).days,
    }


# ============================================================
# 10. CLASSIFY USING RECONSTRUCTED BALANCE
# ============================================================

def classify_using_backward_balance(
    account_transactions,
    dm_date,
):
    """
    Classify using reconstructed balance.

    Current candidate rule:

        balance before DM >= 0
            -> Payment

        balance before DM < 0
            -> Genuine DM

    IMPORTANT:
    This rule should be validated against SME examples before
    being considered final.
    """

    reconstructed = (
        calculate_backward_from_zero_sum(
            account_transactions
        )
    )

    if reconstructed.empty:
        return "None"

    balance_info = get_balance_before_dm(
        reconstructed,
        dm_date
    )

    if balance_info is None:
        return "None"

    balance = balance_info[
        "balance_before_dm"
    ]

    if balance >= 0:
        return "True"

    return "False"


# ============================================================
# 11. HISTORICAL BALANCE POINT
# ============================================================

def get_historical_balance_point(
    acc_nr,
    dm_date,
    connection,
):
    """
    Retrieve a historical balance point for the account.

    Replace the query below with the exact balance source
    already used by your existing post-2016 logic.

    Return:

        {
            "date": ...,
            "balance": ...
        }

    or None.
    """

    dm_date = pd.to_datetime(
        dm_date,
        errors="coerce"
    )

    if pd.isna(dm_date):
        return None

    query = f"""
        SELECT
            ktonr,
            business_dt,
            balance
        FROM datalabs.aragorn_core.balance_credmax_interest_all_accounts
        WHERE ktonr = '{acc_nr}'
          AND business_dt < '{dm_date.strftime("%Y-%m-%d")}'
        ORDER BY business_dt DESC
        LIMIT 1
    """

    df = pd.read_sql(
        query,
        connection
    )

    if df.empty:
        return None

    return {
        "date": pd.to_datetime(
            df["business_dt"].iloc[0]
        ),
        "balance": float(
            df["balance"].iloc[0]
        ),
    }


# ============================================================
# 12. CLASSIFY USING HISTORICAL BALANCE
# ============================================================

def classify_using_historical_balance(
    account_transactions,
    dm_date,
    historical_balance,
):
    """
    Reconstruct backwards from an actual historical balance
    point.
    """

    if historical_balance is None:
        return "None"

    anchor_date = historical_balance["date"]
    anchor_balance = historical_balance["balance"]

    reconstructed = create_balance_column(
        account_transactions=account_transactions,
        anchor_date=anchor_date,
        anchor_balance=anchor_balance,
    )

    if reconstructed.empty:
        return "None"

    balance_info = get_balance_before_dm(
        reconstructed,
        dm_date
    )

    if balance_info is None:
        return "None"

    if balance_info["balance_before_dm"] >= 0:
        return "True"

    return "False"


# ============================================================
# 13. PRODUCT CHANGE / STATS
# ============================================================

def get_product_change_stats(
    acc_nr,
    dm_date,
    connection,
):
    """
    Find product-change/statistics information.

    THIS IS A PLACEHOLDER FOR THE SME-DEFINED LOGIC.

    Once the exact source and meaning of the product-change
    statistic are confirmed, implement it here.

    Return:

        {
            "date": ...,
            "balance": ...
        }

    or None.
    """

    # --------------------------------------------------------
    # TODO:
    #
    # Implement the actual SME-approved product-change
    # statistics query here.
    #
    # For now, return None so the flow safely continues
    # to the UB zero-sum branch.
    # --------------------------------------------------------

    return None


# ============================================================
# 14. BACKWARD + FORWARD FROM PRODUCT CHANGE
# ============================================================

def classify_using_product_change_stats(
    account_transactions,
    dm_date,
    stats,
):
    """
    Product-change branch.

    This is intentionally isolated because the exact
    backward/forward calculation still needs to be defined
    with the SME.
    """

    if stats is None:
        return "None"

    # --------------------------------------------------------
    # TODO:
    #
    # Implement:
    #
    #   backward calculation
    #   +
    #   forward calculation
    #
    # based on the agreed product-change statistics.
    # --------------------------------------------------------

    return "None"


# ============================================================
# 15. OLD PRE-2016 PAYMENT LOGIC
# ============================================================

def classify_old_payment_logic(
    account_transactions,
    dm_date,
    get_debt_spans_function,
    acc_nr,
):
    """
    Existing pre-2016 fallback logic.

    This preserves the old behaviour whenever the new
    reconstruction approach cannot safely be applied.

    Existing rule:

        max_date - DM date <= 5 days

    and:

        if DM is first transaction
            -> Write-off

        otherwise
            -> Payment
    """

    life_span = get_debt_spans_function(
        acc_nr
    )

    if life_span is None or life_span.empty:
        return "None"

    life_span = life_span[
        life_span["ktonr"] == acc_nr
    ].reset_index(drop=True)

    if life_span.empty:
        return "None"

    min_date = pd.to_datetime(
        life_span["min_date"].iloc[0],
        errors="coerce"
    )

    max_date = pd.to_datetime(
        life_span["max_date"].iloc[0],
        errors="coerce"
    )

    dm_date = pd.to_datetime(
        dm_date,
        errors="coerce"
    )

    if pd.isna(min_date) or pd.isna(max_date):
        return "None"

    diff = max_date - dm_date

    # --------------------------------------------------------
    # Old 5-day rule
    # --------------------------------------------------------

    if diff <= pd.Timedelta(
        days=OLD_PAYMENT_WINDOW_DAYS
    ):

        # Old write-off condition:
        # DM is first transaction
        if min_date == dm_date:
            return "Write-off"

        return "True"

    return "False"


# ============================================================
# 16. POST-2016 PAYMENT LOGIC
# ============================================================

def classify_payment_post_2016(
    acc_nr,
    dm_date,
    connection,
):
    """
    Existing post-2016 balance-based payment logic.

    Logic:

        latest available balance before DM >= 0

    Then check loss/provision information for write-off.
    """

    query = f"""
        SELECT
            ktonr,
            business_dt,
            balance
        FROM datalabs.aragorn_core.balance_credmax_interest_all_accounts
        WHERE ktonr = '{acc_nr}'
          AND business_dt <=
              '{(dm_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")}'
        ORDER BY business_dt
    """

    df_bal = pd.read_sql(
        query,
        connection
    )

    if df_bal.empty:
        return "None"

    latest_balance = float(
        df_bal["balance"].iloc[-1]
    )

    # --------------------------------------------------------
    # Balance < 0
    # --------------------------------------------------------

    if latest_balance < 0:
        return "False"

    # --------------------------------------------------------
    # Check write-off / loss provision
    # --------------------------------------------------------

    query = f"""
        SELECT
            ford_nr,
            ford_bogf_dato
        FROM datalabs.aragorn_core.loss_provisions
        WHERE ford_nr = '{acc_nr}'
          AND to_date(ford_bogf_dato)
              >= '{dm_date.strftime("%Y-%m-%d")}'
        LIMIT 1
    """

    df_fordring = pd.read_sql(
        query,
        connection
    )

    # No loss/provision
    if df_fordring.empty:
        return "True"

    # Loss/provision exists
    if latest_balance > 0:
        return "True"

    return "Write-off"


# ============================================================
# 17. NEW PRE-2016 CLASSIFICATION
# ============================================================

def classify_payment_pre_2016(
    acc_nr,
    dm_date,
    connection,
    get_debt_spans_function,
):
    """
    SME-agreed pre-2016 decision tree.

    Priority:

        1. to_account closed
        2. historical balance point
        3. product-change stats
        4. UB sum == zero
        5. old logic fallback
    """

    # --------------------------------------------------------
    # Get account transactions
    # --------------------------------------------------------

    account_transactions = get_account_transactions(
        acc_nr,
        connection
    )

    if account_transactions.empty:
        return "None"

    account_transactions = (
        prepare_account_transactions(
            account_transactions
        )
    )

    # --------------------------------------------------------
    # 1. TO ACCOUNT CLOSED
    # --------------------------------------------------------

    if is_account_closed(
        account_transactions
    ):

        reconstructed = (
            calculate_backward_from_closed_account(
                account_transactions
            )
        )

        balance_info = get_balance_before_dm(
            reconstructed,
            dm_date
        )

        if balance_info is not None:

            if balance_info[
                "balance_before_dm"
            ] >= 0:
                return "True"

            return "False"

    # --------------------------------------------------------
    # 2. HISTORICAL BALANCE DATA POINT
    # --------------------------------------------------------

    historical_balance = (
        get_historical_balance_point(
            acc_nr=acc_nr,
            dm_date=dm_date,
            connection=connection,
        )
    )

    if historical_balance is not None:

        result = classify_using_historical_balance(
            account_transactions=account_transactions,
            dm_date=dm_date,
            historical_balance=historical_balance,
        )

        if result != "None":
            return result

    # --------------------------------------------------------
    # 3. PRODUCT CHANGE / STATS
    # --------------------------------------------------------

    stats = get_product_change_stats(
        acc_nr=acc_nr,
        dm_date=dm_date,
        connection=connection,
    )

    if stats is not None:

        result = classify_using_product_change_stats(
            account_transactions=account_transactions,
            dm_date=dm_date,
            stats=stats,
        )

        if result != "None":
            return result

    # --------------------------------------------------------
    # 4. UB SUM == ZERO
    # --------------------------------------------------------

    if account_sum_is_zero(
        account_transactions
    ):

        reconstructed = (
            calculate_backward_from_zero_sum(
                account_transactions
            )
        )

        balance_info = get_balance_before_dm(
            reconstructed,
            dm_date
        )

        if balance_info is not None:

            if balance_info[
                "balance_before_dm"
            ] >= 0:
                return "True"

            return "False"

    # --------------------------------------------------------
    # 5. FALLBACK TO OLD LOGIC
    # --------------------------------------------------------

    return classify_old_payment_logic(
        account_transactions=account_transactions,
        dm_date=dm_date,
        get_debt_spans_function=get_debt_spans_function,
        acc_nr=acc_nr,
    )


# ============================================================
# 18. MAIN FUNCTION
# ============================================================

def clean_debt_movement_payment(
    dms,
    get_debt_spans_function,
):
    """
    Main payment-cleaning function.

    Output:

        payment = "True"
            -> payment

        payment = "False"
            -> genuine debt movement

        payment = "Write-off"
            -> write-off

        payment = "None"
            -> insufficient information

    SME AGREED FLOW:

        DM >= 2016
            -> historical balance logic

        DM < 2016
            -> closed account
            -> historical balance point
            -> product-change stats
            -> UB sum == zero
            -> old logic fallback
    """

    connection = get_snowflake_connection()

    dms = dms.copy()

    # --------------------------------------------------------
    # Prepare DM dates
    # --------------------------------------------------------

    dms["postvandato_to"] = pd.to_datetime(
        dms["postvandato_to"],
        errors="coerce"
    )

    payment_results = []

    # ========================================================
    # PROCESS EACH DEBT MOVEMENT
    # ========================================================

    for i in range(len(dms)):

        acc_nr = dms.loc[
            i,
            "ktonr_to"
        ]

        dm_date = dms.loc[
            i,
            "postvandato_to"
        ]

        # ----------------------------------------------------
        # Missing information
        # ----------------------------------------------------

        if pd.isna(acc_nr) or pd.isna(dm_date):

            payment_results.append(
                "None"
            )

            continue

        # ====================================================
        # DM >= 2016
        # ====================================================

        if dm_date >= DM_CUTOFF_DATE:

            result = classify_payment_post_2016(
                acc_nr=acc_nr,
                dm_date=dm_date,
                connection=connection,
            )

            payment_results.append(
                result
            )

            continue

        # ====================================================
        # DM < 2016
        # ====================================================

        result = classify_payment_pre_2016(
            acc_nr=acc_nr,
            dm_date=dm_date,
            connection=connection,
            get_debt_spans_function=get_debt_spans_function,
        )

        payment_results.append(
            result
        )

    # --------------------------------------------------------
    # Add result
    # --------------------------------------------------------

    dms["payment"] = payment_results

    return dms



# dms = clean_debt_movement_payment(
#     dms,
#     get_debt_spans_function=get_debt_spans,
# )
