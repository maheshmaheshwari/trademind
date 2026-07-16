"""
DBLogHandler → app_logs hypertable (api/logging_setup.py).

The handler is the durable log path for the HF Space (ephemeral filesystem);
these tests exercise the real handler against the real test-DB table.
"""
import logging

from api.logging_setup import DBLogHandler
from database.db import get_connection, release_connection, _execute


def _fetch_messages(logger_name: str):
    conn = get_connection()
    try:
        cur = _execute(
            conn,
            "SELECT level, message FROM app_logs WHERE logger = ? ORDER BY time",
            (logger_name,),
        )
        return cur.fetchall()
    finally:
        release_connection(conn)


class TestDBLogHandler:
    def test_records_persisted_to_app_logs(self):
        handler = DBLogHandler()
        lg = logging.getLogger("test.dblog.persist")
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
        try:
            lg.info("hello from test %s", 42)
            lg.warning("something odd")
        finally:
            lg.removeHandler(handler)
        handler.flush_to_db()

        rows = _fetch_messages("test.dblog.persist")
        assert ("INFO", "hello from test 42") in rows
        assert ("WARNING", "something odd") in rows

    def test_database_logger_records_skipped(self):
        # database.* records must never re-enter the DB handler (a DB outage
        # would otherwise feed its own error logs back into the failing DB).
        handler = DBLogHandler()
        lg = logging.getLogger("database.db")
        lg.addHandler(handler)
        try:
            lg.error("connection pool exhausted (should not be persisted)")
        finally:
            lg.removeHandler(handler)
        handler.flush_to_db()

        assert _fetch_messages("database.db") == []

    def test_debug_records_not_persisted(self):
        handler = DBLogHandler()
        lg = logging.getLogger("test.dblog.debug")
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)
        try:
            lg.debug("debug noise")
        finally:
            lg.removeHandler(handler)
        handler.flush_to_db()

        assert _fetch_messages("test.dblog.debug") == []
