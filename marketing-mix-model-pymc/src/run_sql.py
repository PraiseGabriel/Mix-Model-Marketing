
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "mmm.sqlite"


def main():
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    pd.read_csv(ROOT / "data" / "raw_ad_spend.csv").to_sql("raw_ad_spend", con, index=False)
    pd.read_csv(ROOT / "data" / "raw_sales.csv").to_sql("raw_sales", con, index=False)

    for path in sorted((ROOT / "sql").glob("*.sql")):
        print(f"\n== {path.name} ==")
        statements = [s.strip() for s in path.read_text().split(";") if s.strip()]
        for stmt in statements:
            if not any(line.strip() and not line.strip().startswith("--") for line in stmt.splitlines()):
                continue
            cur = con.execute(stmt)
            if cur.description:  # a SELECT: show the result
                cols = [d[0] for d in cur.description]
                print(pd.DataFrame(cur.fetchall(), columns=cols).to_string(index=False))
    out = pd.read_sql("SELECT * FROM model_input", con)
    out.to_csv(ROOT / "data" / "model_input.csv", index=False)
    print(f"\nexported data/model_input.csv ({len(out)} weeks)")
    con.close()


if __name__ == "__main__":
    main()
