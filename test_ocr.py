#!/usr/bin/env python3
"""
Test script for OCR API endpoints.

Tests:
1. Health check endpoint
2. File upload OCR with local WebP image
3. URL-based OCR with remote WebP image
"""

import os
import requests
import json
from pathlib import Path


# Configuration
API_BASE: str = "http://localhost:8000"
LOCAL_IMAGE: str = "Test Image/abc.webp"
REMOTE_IMAGE_URL: str = "https://www.all-ppt-templates.com/images/xgenerated-random-text.jpg.pagespeed.ic.VxwUZTXYws.webp"

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def print_header(title: str) -> None:
    """Print a formatted test header."""
    print(f"\n{CYAN}{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}{RESET}\n")


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓ {msg}{RESET}")


def print_error(msg: str) -> None:
    """Print error message."""
    print(f"{RED}✗ {msg}{RESET}")


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"{YELLOW}ℹ {msg}{RESET}")


def test_health_check() -> bool:
    """Test health check endpoint."""
    print_header("Test 1: Health Check")
    
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=10)
        resp.raise_for_status()
        print_success(f"Health endpoint returned: {resp.json()}")
        return True
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False


def test_file_upload() -> bool:
    """Test file upload OCR with local WebP image."""
    print_header("Test 2: File Upload OCR")
    
    local_path = Path(LOCAL_IMAGE)
    
    if not local_path.exists():
        print_error(f"Local image not found: {local_path}")
        print_info(f"Expected path: {local_path.absolute()}")
        return False
    
    try:
        print_info(f"Uploading file: {local_path}")
        print_info(f"File size: {local_path.stat().st_size} bytes")
        
        with open(local_path, "rb") as f:
            files = {"file": (local_path.name, f, "image/webp")}
            resp = requests.post(
                f"{API_BASE}/ocr/file",
                files=files,
                timeout=300  # Allow long timeout for Azure API
            )
        
        resp.raise_for_status()
        result = resp.json()
        
        print_success(f"Upload successful for: {result.get('filename', 'unknown')}")
        
        # Check for retry history (indicates throttling/retries occurred)
        if "_retry_history" in result:
            print_info(f"Retry history: {result['_retry_history']}")
        
        extracted_text = result.get("text", "")
        if extracted_text:
            print_success(f"OCR extracted {len(extracted_text)} characters")
            print(f"\n{CYAN}Extracted Text:{RESET}")
            print(f"{extracted_text[:500]}...")  # Print first 500 chars
        else:
            print_error("No text extracted from image")
        
        return True
    except Exception as e:
        print_error(f"File upload test failed: {e}")
        return False


def test_url_ocr() -> bool:
    """Test URL-based OCR with remote WebP image."""
    print_header("Test 3: URL-Based OCR")
    
    try:
        print_info(f"Testing URL: {REMOTE_IMAGE_URL}")
        
        payload = {"url": REMOTE_IMAGE_URL}
        resp = requests.post(
            f"{API_BASE}/ocr/url",
            json=payload,
            timeout=300  # Allow long timeout for Azure API
        )
        
        resp.raise_for_status()
        result = resp.json()
        
        print_success(f"URL OCR successful for: {REMOTE_IMAGE_URL}")
        
        # Check for retry history (indicates throttling/retries occurred)
        if "_retry_history" in result:
            print_info(f"Retry history: {result['_retry_history']}")
        
        extracted_text = result.get("text", "")
        if extracted_text:
            print_success(f"OCR extracted {len(extracted_text)} characters")
            print(f"\n{CYAN}Extracted Text:{RESET}")
            print(f"{extracted_text[:500]}...")  # Print first 500 chars
        else:
            print_error("No text extracted from URL image")
        
        return True
    except Exception as e:
        print_error(f"URL OCR test failed: {e}")
        return False


def test_ms_connection() -> bool:
    """Test MS Connection endpoint."""
    print_header("Test 4: MS Connection Test")
    
    try:
        resp = requests.get(f"{API_BASE}/test-connection", timeout=10)
        resp.raise_for_status()
        result = resp.json()
        print_success(f"MS Connection test returned: {result}")
        
        # Check if connection was successful
        if result.get("status") == "ok":
            print_success("Azure Vision API connection verified")
        elif result.get("status") == "error":
            print_error(f"Connection error: {result.get('error', 'Unknown error')}")
        
        return result.get("status") == "ok"
    except Exception as e:
        print_error(f"MS Connection test failed: {e}")
        return False


def main() -> None:
    """Run all tests."""
    print(f"\n{CYAN}OCR API Test Suite{RESET}")
    print(f"{CYAN}API Base: {API_BASE}{RESET}\n")
    
    tests = [
        ("Health Check", test_health_check),
        ("File Upload", test_file_upload),
        ("URL OCR", test_url_ocr),
        ("MS Connection", test_ms_connection),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print_error(f"Unexpected error in {name}: {e}")
            results[name] = False
    
    # Print summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = f"{GREEN}PASS{RESET}" if passed_test else f"{RED}FAIL{RESET}"
        print(f"  {name:.<50} {status}")
    
    print(f"\n{CYAN}Results: {passed}/{total} tests passed{RESET}\n")
    
    if passed == total:
        print_success("All tests passed!")
    else:
        print_error(f"{total - passed} test(s) failed")


if __name__ == "__main__":
    main()
