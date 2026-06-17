import sys
import os

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# 添加当前目录到Python路径
sys.path.append(os.getcwd())

# 尝试导入run.py中的函数
try:
    from run import parse_argument, get_logger
    print("Successfully imported functions from run.py")
    
    # 测试parse_argument函数
    print("\nTesting parse_argument...")
    args = parse_argument()
    print(f"Parsed arguments: {args}")
    
    # 测试get_logger函数
    print("\nTesting get_logger...")
    logger = get_logger(args)
    print("Logger initialized successfully")
    
    print("\nTest completed successfully")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nScript completed")