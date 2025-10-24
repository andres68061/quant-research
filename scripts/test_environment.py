#!/usr/bin/env python3
"""
Test script to verify the quant environment is properly set up.
This script tests all the key packages we installed.
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("test_environment")


def test_imports():
    """Test that all key packages can be imported successfully."""
    logger.info("imports_start")

    try:
        import numpy as np
        logger.info("import_ok", extra={"package": "numpy", "version": np.__version__})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "numpy", "error": str(e)})
        return False

    try:
        import pandas as pd
        logger.info("import_ok", extra={"package": "pandas", "version": pd.__version__})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "pandas", "error": str(e)})
        return False

    try:
        import scipy
        logger.info("import_ok", extra={"package": "scipy", "version": scipy.__version__})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "scipy", "error": str(e)})
        return False

    try:
        import matplotlib.pyplot as plt
        logger.info("import_ok", extra={"package": "matplotlib", "version": plt.matplotlib.__version__})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "matplotlib", "error": str(e)})
        return False

    try:
        import seaborn as sns
        logger.info("import_ok", extra={"package": "seaborn", "version": sns.__version__})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "seaborn", "error": str(e)})
        return False

    try:
        import plotly
        logger.info("import_ok", extra={"package": "plotly", "version": plotly.__version__})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "plotly", "error": str(e)})
        return False

    try:
        import yfinance as yf
        logger.info("import_ok", extra={"package": "yfinance", "version": yf.__version__})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "yfinance", "error": str(e)})
        return False

    try:
        logger.info("import_ok", extra={"package": "pandas_datareader"})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "pandas_datareader", "error": str(e)})
        return False

    try:
        import talib  # noqa: F401
        logger.info("import_ok", extra={"package": "ta-lib"})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "ta-lib", "error": str(e)})
        return False

    try:
        import sklearn
        logger.info("import_ok", extra={"package": "scikit-learn", "version": sklearn.__version__})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "scikit-learn", "error": str(e)})
        return False

    try:
        import statsmodels
        logger.info("import_ok", extra={"package": "statsmodels", "version": statsmodels.__version__})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "statsmodels", "error": str(e)})
        return False

    try:
        import jupyter  # noqa: F401
        logger.info("import_ok", extra={"package": "jupyter"})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "jupyter", "error": str(e)})
        return False

    try:
        import ipykernel
        logger.info("import_ok", extra={"package": "ipykernel", "version": ipykernel.__version__})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "ipykernel", "error": str(e)})
        return False

    try:
        import requests
        logger.info("import_ok", extra={"package": "requests", "version": requests.__version__})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "requests", "error": str(e)})
        return False

    try:
        import tqdm
        logger.info("import_ok", extra={"package": "tqdm", "version": tqdm.__version__})
    except Exception as e:
        logger.exception("import_fail", extra={"package": "tqdm", "error": str(e)})
        return False

    return True


def test_basic_functionality():
    """Test basic functionality of key packages."""
    logger.info("basic_functionality_start")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        arr = np.array([1, 2, 3, 4, 5])
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.title('Test Plot')
        plt.close()

        logger.info("basic_functionality_ok", extra={"numpy_array": arr.tolist(), "df_rows": int(len(df))})
        return True
    except Exception as e:
        logger.exception("basic_functionality_fail", extra={"error": str(e)})
        return False


def main():
    logger.info(
        "env_test_start",
        extra={
            "python": sys.version,
            "cwd": os.getcwd(),
            "executable": sys.executable,
        },
    )

    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()

    if imports_ok and functionality_ok:
        logger.info("env_ready")
    else:
        logger.error("env_not_ready")

    return imports_ok and functionality_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
