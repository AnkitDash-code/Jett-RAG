"""
Auto-start KoboldCpp embedding server for Qwen3-Embedding model.
Called automatically by main.py at startup.
"""
import subprocess
import sys
import os
import time

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to KoboldCpp executable (auto-detected relative to project)
KOBOLDCPP_PATH = os.path.join(SCRIPT_DIR, '..', 'LLM-Backend', 'koboldcpp.exe')
KOBOLDCPP_PATH = os.path.abspath(KOBOLDCPP_PATH)

# Path to embedding settings file (in same directory as this script)
SETTINGS_PATH = os.path.join(SCRIPT_DIR, 'embedding_settings.kcpps')
SETTINGS_PATH = os.path.abspath(SETTINGS_PATH)

def check_already_running():
    """Check if KoboldCpp embedding server is already running on port 5001."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 5001))
        sock.close()
        return result == 0
    except:
        return False

def main():
    # Check if already running
    if check_already_running():
        print("[OK] KoboldCpp embedding server already running on port 5001")
        return True
    
    if not os.path.exists(KOBOLDCPP_PATH):
        print(f"[ERROR] KoboldCpp executable not found: {KOBOLDCPP_PATH}")
        print("   Please download KoboldCpp and place it in the LLM-Backend folder")
        return False
    
    if not os.path.exists(SETTINGS_PATH):
        print(f"[ERROR] Settings file not found: {SETTINGS_PATH}")
        return False
    
    print(f"[STARTING] Launching KoboldCpp embedding server...")
    print(f"   Executable: {KOBOLDCPP_PATH}")
    print(f"   Settings:   {SETTINGS_PATH}")
    
    # Launch KoboldCpp as a background process (detached)
    # Use CREATE_NEW_CONSOLE on Windows to prevent blocking
    try:
        if sys.platform == 'win32':
            # On Windows, use CREATE_NEW_CONSOLE to show in separate window
            subprocess.Popen(
                [KOBOLDCPP_PATH, SETTINGS_PATH],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=os.path.dirname(KOBOLDCPP_PATH)
            )
        else:
            subprocess.Popen(
                [KOBOLDCPP_PATH, SETTINGS_PATH],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        
        print("[OK] KoboldCpp embedding server started in background")
        print("   Waiting for server to be ready...")
        
        # Wait for server to start (up to 30 seconds)
        for i in range(30):
            time.sleep(1)
            if check_already_running():
                print(f"[OK] Embedding server ready on port 5001 (took {i+1}s)")
                return True
            print(f"   Still waiting... ({i+1}s)")
        
        print("[WARNING] Server started but not responding yet. It may still be loading the model.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to start KoboldCpp: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
