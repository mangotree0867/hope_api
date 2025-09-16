#!/usr/bin/env python3
"""
Test script to verify that the API endpoints work with optional authentication
Now using ANONYMOUS_USER_ID (0) for unauthenticated users instead of NULL
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_predict_video_without_auth():
    """Test /predict-video endpoint without authentication"""
    print("\n=== Testing /predict-video without authentication ===")

    # Create a dummy video file for testing
    with open("/tmp/test_video.mp4", "wb") as f:
        f.write(b"dummy video content")

    with open("/tmp/test_video.mp4", "rb") as f:
        files = {"file": ("test.mp4", f, "video/mp4")}
        response = requests.post(f"{BASE_URL}/predict-video", files=files)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
    return response.status_code != 401  # Should not return 401 anymore

def test_predict_sequence_without_auth():
    """Test /predict_sequence endpoint without authentication"""
    print("\n=== Testing /predict_sequence without authentication ===")

    payload = {
        "image_sequence": ["dGVzdA=="]  # Base64 encoded "test"
    }

    response = requests.post(
        f"{BASE_URL}/predict_sequence",
        json=payload
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}...")
    return response.status_code != 401  # Should not return 401 anymore

def test_predict_video_with_auth():
    """Test /predict-video endpoint with authentication"""
    print("\n=== Testing /predict-video with authentication ===")

    # First login to get a token (assuming test user exists)
    login_response = requests.post(
        f"{BASE_URL}/auth/login",
        json={"login_id": "test", "password": "test123"}
    )

    if login_response.status_code == 200:
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        with open("/tmp/test_video.mp4", "rb") as f:
            files = {"file": ("test.mp4", f, "video/mp4")}
            response = requests.post(
                f"{BASE_URL}/predict-video",
                files=files,
                headers=headers
            )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        return response.status_code in [200, 400, 500]  # Any non-auth error is fine
    else:
        print("Could not login with test user")
        return False

if __name__ == "__main__":
    print("Testing Optional Authentication for Prediction Endpoints")
    print("=" * 60)

    tests_passed = 0
    tests_total = 3

    try:
        if test_predict_video_without_auth():
            tests_passed += 1
            print("✅ Test passed: predict-video without auth")
        else:
            print("❌ Test failed: predict-video without auth")
    except Exception as e:
        print(f"❌ Test error: {e}")

    try:
        if test_predict_sequence_without_auth():
            tests_passed += 1
            print("✅ Test passed: predict_sequence without auth")
        else:
            print("❌ Test failed: predict_sequence without auth")
    except Exception as e:
        print(f"❌ Test error: {e}")

    try:
        if test_predict_video_with_auth():
            tests_passed += 1
            print("✅ Test passed: predict-video with auth")
        else:
            print("❌ Test failed: predict-video with auth")
    except Exception as e:
        print(f"❌ Test error: {e}")

    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    print("=" * 60)