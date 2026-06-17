import sys

print(f"Python version: {sys.version}")

# 尝试导入CLIPProcessor和CLIPModel
try:
    from transformers import CLIPProcessor, CLIPModel
    print("Successfully imported CLIPProcessor and CLIPModel")
except Exception as e:
    print(f"Error importing CLIPProcessor and CLIPModel: {e}")
    import traceback
    traceback.print_exc()

print("Test completed")