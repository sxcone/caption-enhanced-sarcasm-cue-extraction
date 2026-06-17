import sys
import traceback

print(f"Python version: {sys.version}")

# 尝试导入transformers模块
print("\n1. Testing transformers import...")
try:
    from transformers import BertConfig, CLIPConfig, BertModel
    print("Successfully imported transformers")
except Exception as e:
    print(f"Error importing transformers: {e}")
    traceback.print_exc()

# 尝试导入CLIPProcessor
print("\n2. Testing CLIPProcessor import...")
try:
    from transformers.models.clip import CLIPProcessor
    print("Successfully imported CLIPProcessor")
except Exception as e:
    print(f"Error importing CLIPProcessor: {e}")
    traceback.print_exc()

# 尝试导入CLIPModel
print("\n3. Testing CLIPModel import...")
try:
    from transformers.models.clip import CLIPModel
    print("Successfully imported CLIPModel")
except Exception as e:
    print(f"Error importing CLIPModel: {e}")
    traceback.print_exc()

print("\nTest completed")