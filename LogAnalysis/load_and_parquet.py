import sys
from pathlib import Path

import duckdb

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "logs.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "logs_expanded_parquet"
DEFAULT_SAMPLE_SIZE = 200000
DEFAULT_TEMP_DIR = "/tmp/duckdb"
ROW_LIMIT = None
PREVIEW_ROWS = 0

ATTRIBUTE_COLUMNS = [
    ("ScenarioStepIndex", "INTEGER", False),
    ("ScenarioLocalStep", "INTEGER", False),
    ("SpikeWindow", "BOOLEAN", False),
    ("FailureWindow", "BOOLEAN", False),
    ("TrendWindow", "BOOLEAN", False),
    ("CpuUsage", "DOUBLE", False),
    ("MemoryUsageMb", "INTEGER", False),
    ("RequestSizeBytes", "INTEGER", False),
    ("ResponseSizeBytes", "INTEGER", False),
    ("Retries", "INTEGER", False),
    ("SystemRole", "VARCHAR", True),
    ("ClusterNode", "VARCHAR", True),
    ("Region", "VARCHAR", True),
    ("BusinessKey", "VARCHAR", True),
    ("DiskQueueLength", "INTEGER", False),
    ("NetworkErrors", "INTEGER", False),
    ("LocalQps", "DOUBLE", False),
    ("IsSpikeByLatency", "BOOLEAN", False),
    ("IsSpikeByCpu", "BOOLEAN", False),
    ("IsSpikeByQps", "BOOLEAN", False),
    ("IsBackgroundMetric", "BOOLEAN", False),
]


def escape_path(path: Path) -> str:
    return str(path).replace("'", "''")


def detect_resources() -> dict:
    try:
        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root / "LogAnalysis"))
        from loganalysis.config import detect_resources as _detect
    except Exception:
        return {"threads": 4, "memory_limit_gb": 8, "codec": "ZSTD"}
    return _detect()


def default_row_group_mb(memory_limit_gb: int) -> int:
    if memory_limit_gb >= 32:
        return 512
    if memory_limit_gb >= 16:
        return 256
    return 128


def main() -> None:
    input_path = DEFAULT_INPUT

    if not input_path.exists():
        raise SystemExit(f"Nie znaleziono pliku: {input_path}")

    cfg = detect_resources()
    threads = cfg["threads"]
    memory_gb = cfg["memory_limit_gb"]
    codec = cfg["codec"].upper()
    row_group_mb = max(256, default_row_group_mb(memory_gb))

    con = duckdb.connect()
    con.execute("PRAGMA threads=?", [threads])
    con.execute("PRAGMA memory_limit=?", [f"{memory_gb}GB"])
    con.execute("PRAGMA temp_directory=?", [DEFAULT_TEMP_DIR])
    con.execute("LOAD json;")

    csv_sql = (
        "read_csv_auto("
        f"'{escape_path(input_path)}', "
        "delim=';', "
        "header=true, "
        "parallel=true, "
        f"sample_size={DEFAULT_SAMPLE_SIZE}"
        ")"
    )
    if ROW_LIMIT:
        con.execute(f"CREATE VIEW logs AS SELECT * FROM {csv_sql} LIMIT {ROW_LIMIT}")
    else:
        con.execute(f"CREATE VIEW logs AS SELECT * FROM {csv_sql}")

    columns = [row[1] for row in con.execute("PRAGMA table_info('logs')").fetchall()]
    if "AttributesJson" not in columns:
        raise SystemExit("Brak kolumny AttributesJson w CSV")

    def quote_ident(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    base_columns = [col for col in columns if col != "AttributesJson"]
    base_select = ", ".join(quote_ident(col) for col in base_columns)
    attr_select = ", ".join(
        (
            f"json_extract_string(AttributesJson, '$.{name}') AS {name}"
            if is_string
            else f"try_cast(json_extract_string(AttributesJson, '$.{name}') AS {dtype}) AS {name}"
        )
        for name, dtype, is_string in ATTRIBUTE_COLUMNS
    )
    expanded_select = (
        "SELECT "
        f"{base_select}, {attr_select}, "
        "substr(CAST(Timestamp AS VARCHAR), 1, 10) AS LogDate "
        "FROM logs"
    )
    con.execute(f"CREATE OR REPLACE VIEW expanded AS {expanded_select}")

    if PREVIEW_ROWS > 0:
        print("Podglad po parsowaniu:")
        preview_df = con.execute(f"SELECT * FROM expanded LIMIT {PREVIEW_ROWS}").df()
        print(preview_df.to_string(index=False))
        print()

    print("Rozdzielono AttributesJson na osobne kolumny.")
    print(f"Nowa liczba kolumn: {len(base_columns) + len(ATTRIBUTE_COLUMNS)}")
    print("Dodane kolumny:", ", ".join(name for name, _, _ in ATTRIBUTE_COLUMNS))
    print()

    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_str = escape_path(out_dir)
    row_group_size = row_group_mb * 1024 * 1024
    con.execute(
        f"COPY ({expanded_select}) TO '{out_str}' "
        f"(FORMAT 'parquet', CODEC {codec}, ROW_GROUP_SIZE {row_group_size}, "
        "PARTITION_BY (LogDate), OVERWRITE_OR_IGNORE true)"
    )
    print(f"Zapisano: {out_dir}")


if __name__ == "__main__":
    main()
