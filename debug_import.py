import os
import sys

root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    import model.predictor
    print("SUCCESSFULLY IMPORTED model.predictor")
except Exception:
    import traceback
    with open("tmp_error.log", "w") as f:
        f.write(traceback.format_exc())
    print("LOGGED EXCEPTION TO tmp_error.log")
