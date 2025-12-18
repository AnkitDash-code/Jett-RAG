#!/usr/bin/env python
"""
Simple test runner script.
Run all tests: python tests/run_tests.py
Run specific test: python tests/run_tests.py test_auth
"""
import sys
import subprocess


def main():
    """Run pytest with appropriate options."""
    # Base pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",  # Verbose
        "--tb=short",  # Short tracebacks
        "-x",  # Stop on first failure
    ]
    
    # If specific test file provided
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if not test_name.startswith("test_"):
            test_name = f"test_{test_name}"
        if not test_name.endswith(".py"):
            test_name = f"{test_name}.py"
        cmd.append(f"tests/{test_name}")
    else:
        # Run all tests
        cmd.append("tests/")
    
    # Run pytest
    result = subprocess.run(cmd, cwd=str(__file__).rsplit("tests", 1)[0])
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
