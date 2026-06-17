import sys
import os

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# 尝试导入基本模块
try:
    import numpy as np
    print("Successfully imported numpy")
except Exception as e:
    print(f"Error importing numpy: {e}")

try:
    import torch
    print(f"Successfully imported torch, version: {torch.__version__}")
except Exception as e:
    print(f"Error importing torch: {e}")

try:
    from transformers import BertConfig, CLIPConfig, BertModel
    print("Successfully imported transformers")
except Exception as e:
    print(f"Error importing transformers: {e}")

try:
    from transformers.models.clip import CLIPProcessor
    print("Successfully imported CLIPProcessor")
except Exception as e:
    print(f"Error importing CLIPProcessor: {e}")

try:
    from models import *
    print("Successfully imported models")
except Exception as e:
    print(f"Error importing models: {e}")
    import traceback
    traceback.print_exc()

try:
    from processor import *
    print("Successfully imported processor")
except Exception as e:
    print(f"Error importing processor: {e}")
    import traceback
    traceback.print_exc()

try:
    from schedulers import *
    print("Successfully imported schedulers")
except Exception as e:
    print(f"Error importing schedulers: {e}")
    import traceback
    traceback.print_exc()

print("Test completed")