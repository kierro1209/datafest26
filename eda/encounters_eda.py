import pandas as pd
from pathlib import Path

def main():
    df = pd.read_csv("event_enriched.csv")
    print(df.columns)

    cols = ["VisitType", "VisitTypeDescription", "ProviderDurableKey"]
    df = df[cols]

    print("\n=== BASIC OVERVIEW ===")
    print(df.head(10))
    print("\nShape:", df.shape)

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== VISIT TYPE VALUE COUNTS ===")
    print(df["VisitType"].value_counts(dropna=False).head(20))

    print("\n=== VISIT TYPE DESCRIPTION VALUE COUNTS ===")
    print(df["VisitTypeDescription"].value_counts(dropna=False).head(20))

    print("\n=== PROVIDER DIVERSITY ===")
    print("Unique providers:", df["ProviderDurableKey"].nunique())
    print("Top 10 providers:")
    print(df["ProviderDurableKey"].value_counts().head(10))

    print("\n=== CROSS TAB: VisitType vs Description ===")
    print(pd.crosstab(df["VisitType"], df["VisitTypeDescription"]).head(20))


if __name__ == "__main__":
    main()