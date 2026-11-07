"""Step 7 - SQLite persistence so results survive and can be queried."""
import json
import sqlite3

from . import config as C


def connect():
    con = sqlite3.connect(C.DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS results(
        doc TEXT PRIMARY KEY, severity TEXT, fields TEXT, findings TEXT)""")
    return con


def save_results(results):
    con = connect()
    with con:
        con.executemany(
            "REPLACE INTO results VALUES (?,?,?,?)",
            [(r["doc"], r["severity"], json.dumps(r["fields"]),
              json.dumps(r["findings"])) for r in results],
        )
    con.close()


def load_results():
    con = connect()
    rows = con.execute("SELECT doc,severity,fields,findings FROM results").fetchall()
    con.close()
    return [{"doc": d, "severity": s, "fields": json.loads(f),
             "findings": json.loads(fd)} for d, s, f, fd in rows]
