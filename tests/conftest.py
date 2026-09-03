import os
import tempfile
import uuid

os.environ["YYLAB_HOME"] = os.path.join(tempfile.gettempdir(), f"yylab_test_{uuid.uuid4().hex[:8]}")
# Ingest tests run the in-process path by default (monkeypatched adapters and
# constants cannot cross process boundaries); parallel tests opt in explicitly.
os.environ.setdefault("YYLAB_INGEST_WORKERS", "1")
