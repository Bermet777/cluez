import os
from celery.loaders.app import AppLoader
from django import db


class CustomDjangoLoader(AppLoader):
    def on_task_init(self, task_id, task):
        """Called before every task."""
        print("Loader called")
        for conn in db.connections.all():
            conn.close_if_unusable_or_obsolete()

        # Calling db.close() on some DB connections will cause the inherited DB
        # conn to also get broken in the parent process so we need to remove it
        # without triggering any network IO that close() might cause.
        for c in db.connections.all():
            if c and c.connection:
                try:
                    os.close(c.connection.fileno())
                except (AttributeError, OSError, TypeError, db.InterfaceError):
                    pass
            try:
                c.close()
            except db.InterfaceError:
                pass
            except db.DatabaseError as exc:
                str_exc = str(exc)
                if 'closed' not in str_exc and 'not connected' not in str_exc:
                    raise
        super(CustomDjangoLoader, self).on_task_init(task_id, task)


