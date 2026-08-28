import os
import tempfile
import uuid

os.environ["YYLAB_HOME"] = os.path.join(tempfile.gettempdir(), f"yylab_test_{uuid.uuid4().hex[:8]}")
