from pathlib import Path

import pandas as pd


def audit_coverage(dataset: pd.DataFrame, news: pd.DataFrame, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    data = dataset.assign(year=pd.to_datetime(dataset["Date"]).dt.year)
    records = news.assign(year=pd.to_datetime(news["trading_date"]).dt.year)
    coverage = data.groupby(["year", "ticker"]).agg(
        sessions=("target", "size"), recorded_news_days=("has_news", "sum"),
        recorded_news=("news_count", "sum"))
    coverage["recorded_day_ratio"] = coverage["recorded_news_days"] / coverage["sessions"]
    coverage["coverage_verified"] = False
    sources = records.groupby(["year", "ticker", "source"], dropna=False).size().rename("news").reset_index()
    relevance = records.groupby(["year", "ticker"])["relevance_score"].agg(["count", "mean", "min", "max"])
    duplicates = int(records.duplicated(["ticker", "url", "title", "published_at"]).sum())
    repeated_titles = int(records.duplicated(["ticker", "trading_date", "title"]).sum())
    coverage.to_csv(output / "coverage_by_year_ticker.csv")
    sources.to_csv(output / "sources_by_year_ticker.csv", index=False)
    relevance.to_csv(output / "relevance_by_year_ticker.csv")
    return {"news_rows": len(news), "exact_duplicates": duplicates,
            "repeated_titles_same_session": repeated_titles,
            "assigned_later_date": int(news["assigned_later_date"].sum()),
            "coverage_verified": False,
            "absence_interpretation": "No recorded news; provider completeness is unknown."}
