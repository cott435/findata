"""
demo_aapl.py
============

End-to-end demo of SECPipeline on Apple (AAPL).

    1. init_ticker  -> download & persist every 10-K / 10-Q / 8-K back to min_date
                       and build the quarterly numeric DB
    2. get_latest_10k
    3. get_latest_10q
    4. get_10k_for_year(2018)
    5. get_financials(years=5)  -> quarterly BS / IS / CF for the last 5 years

Requires network access to SEC EDGAR and a valid identity string. SEC mandates a
descriptive User-Agent of the form "Name email@domain".

Run:
    EDGAR_IDENTITY="Jane Doe jane@example.com" python demo_aapl.py
    # or pass identity= in the constructor below
"""

import asyncio
from findata.configs import EDGAR_IDENTITY, DATA_DIR

TICKER = "MSFT"


def preview_sections(filing, n_chars=300):
    print(f"\n{filing!r}")
    print(f"  period_of_report={filing.period_of_report}  fiscal_year={filing.fiscal_year}")
    print(f"  saved sections ({len(filing.sections)}): {', '.join(filing.sections) or 'none'}")
    print(f"  statement csvs: {', '.join(filing.statements) or 'none'}")
    mda = filing.sections.get("item_07_mda") or filing.sections.get("part_i_item_2_mda")
    if mda:
        snippet = " ".join(mda.split())[:n_chars]
        print(f"  MD&A preview: {snippet}...")


def show_quarterly(financials):
    for name, df in financials.items():
        print(f"\n===== {name}  ({len(df)} rows, "
            f"{df['concept'].nunique() if not df.empty else 0} concepts) =====")
        if df.empty:
            print("  (empty)")
            continue
        # show the trajectory of one representative line item across quarters
        key = "Revenues" if name == "income_statement" else (
            "Assets" if name == "balance_sheet" else
            "NetCashProvidedByUsedInOperatingActivities")
        sub = df[df["concept"].str.contains(key, case=False, na=False)]
        if sub.empty:
            sub = df  # fall back to whatever is there
        cols = ["fiscal_year", "fiscal_quarter", "period_start", "period_end",
                "filing_date", "form_type", "label", "value", "derived"]
        print(sub[cols].tail(12).to_string(index=False))


async def main():
    pipe = SECPipeline(
        DATA_DIR,
        identity=EDGAR_IDENTITY,
        parse_extra_data=False,   # essential sections only; set True for the rest
        requests_per_second=6.0,  # stay well under SEC's 10 req/s ceiling
        max_concurrency=5,
    )

    # 1) Full download (this is the heavy step; it walks every filing since min_date).
    #    Drop "8-K" from forms if you only care about the periodic reports.
    print(f">>> init_ticker({TICKER}) ...")
    """counts = await pipe.init_ticker(
        TICKER,
        min_date="2010-01-01",
        force=False,
        forms=("10-K", "10-Q"),
    )
    print("saved per form:", counts)"""

    # 2-4) Retrieval (served from disk now that init has run).
    latest_10k = await pipe.aget_latest_10k(TICKER)
    preview_sections(latest_10k)

    latest_10q = await pipe.aget_latest_10q(TICKER)
    preview_sections(latest_10q)

    tenk_2018 = await pipe.aget_10k_for_year(TICKER, 2018)
    preview_sections(tenk_2018)

    # 5) Quarterly numeric records for the last 5 years.
    print("\n>>> get_financials(years=5)")
    financials = await pipe.aget_financials(TICKER, years=5)
    show_quarterly(financials)


if __name__ == "__main__":
    asyncio.run(main())
