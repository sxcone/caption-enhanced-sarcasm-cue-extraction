import sys

print(f"Python version: {sys.version}")

# 尝试导入transformers模块
try:
    from transformers import BertConfig, CLIPConfig, BertModel
    print("Successfully imported transformers")
except Exception as e:
    print(f"Error importing transformers: {e}")
    import traceback
    traceback.print_exc()

print("Test completed")