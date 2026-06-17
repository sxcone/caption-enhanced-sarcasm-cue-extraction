import sys

print(f"Python version: {sys.version}")

# 尝试导入transformers模块
try:
    import transformers
    print(f"Successfully imported transformers, version: {transformers.__version__}")
except Exception as e:
    print(f"Error importing transformers: {e}")
    import traceback
    traceback.print_exc()

print("Test completed")