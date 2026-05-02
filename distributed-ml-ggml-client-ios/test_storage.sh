#!/bin/bash

# Storage Server Integration Test Script
# Usage: ./test_storage.sh <IP> <PORT>

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <IP> <PORT>"
    exit 1
fi

IP=$1
PORT=$2
BASE_URL="http://$IP:$PORT"

# Track IDs to cleanup on the phone
CREATED_IDS=()

cleanup() {
    echo ""
    echo "--- Cleaning up ---"
    echo "Removing local temporary files..."
    rm -f small.bin large_10mb.bin downloaded_verify.bin dummy.bin
    
    if [ ${#CREATED_IDS[@]} -gt 0 ]; then
        echo "Attempting to remove test chunks from the device..."
        for id in "${CREATED_IDS[@]}"; do
            curl -s -X DELETE "$BASE_URL/chunk/$id" > /dev/null || true
        done
    fi
}

trap cleanup EXIT

echo "--- Starting Storage Server Tests for $BASE_URL ---"

# Helper for testing status codes
test_status() {
    local method=$1
    local path=$2
    local expected=$3
    local data_arg=$4
    
    echo -n "Testing $method $path (Expected $expected)... "
    
    local status
    if [ "$method" == "PUT" ]; then
        status=$(curl -s -o /dev/null -w "%{http_code}" -X PUT --data-binary "@$data_arg" "$BASE_URL$path")
    elif [ "$method" == "GET" ]; then
        status=$(curl -s -o /dev/null -w "%{http_code}" -X GET "$BASE_URL$path")
    elif [ "$method" == "DELETE" ]; then
        status=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL$path")
    fi

    if [ "$status" -eq "$expected" ]; then
        echo "PASS"
    else
        echo "FAIL (Got $status)"
        exit 1
    fi
}

# Core test sequence for a specific file
run_chunk_test() {
    local file_path=$1
    local label=$2
    
    echo "--- Testing with $label file ($file_path) ---"
    
    # Calculate SHA256
    local chunk_id
    if command -v shasum >/dev/null; then
        chunk_id=$(shasum -a 256 "$file_path" | awk '{print $1}')
    else
        chunk_id=$(sha256sum "$file_path" | awk '{print $1}')
    fi

    # 1. PUT chunk
    test_status "PUT" "/chunk/$chunk_id" 200 "$file_path"
    CREATED_IDS+=("$chunk_id")

    # 2. GET and Verify
    echo -n "Testing GET /chunk/$chunk_id and verifying content... "
    curl -s "$BASE_URL/chunk/$chunk_id" > downloaded_verify.bin
    if diff "$file_path" downloaded_verify.bin > /dev/null; then
        echo "PASS"
    else
        echo "FAIL (Content mismatch)"
        exit 1
    fi

    # 3. Check List
    echo -n "Verifying chunk in /chunks/list... "
    if curl -s "$BASE_URL/chunks/list" | grep -q "$chunk_id"; then
        echo "PASS"
    else
        echo "FAIL (Not in list)"
        exit 1
    fi

    # 4. DELETE chunk
    test_status "DELETE" "/chunk/$chunk_id" 200
    
    # Remove from cleanup list since it's already deleted
    for i in "${!CREATED_IDS[@]}"; do
        if [[ ${CREATED_IDS[i]} == "$chunk_id" ]]; then
            unset 'CREATED_IDS[i]'
        fi
    done
    
    # 5. Verify Deletion
    echo -n "Verifying deletion in /chunks/list... "
    if ! curl -s "$BASE_URL/chunks/list" | grep -q "$chunk_id"; then
        echo "PASS"
    else
        echo "FAIL (Still in list)"
        exit 1
    fi
}

# 1. Test /storage_info
echo -n "Testing GET /storage_info... "
INFO=$(curl -s "$BASE_URL/storage_info")
if [[ $INFO == *"total_space"* ]]; then
    echo "PASS"
else
    echo "FAIL (Invalid response: $INFO)"
    exit 1
fi

# 2. Small file test
echo "Small file test content $(date)" > small.bin
run_chunk_test "small.bin" "small (KB)"

# 3. Large file test (10MB)
echo -n "Generating 10MB random test file... "
dd if=/dev/urandom of=large_10mb.bin bs=1M count=10 status=none
echo "Done"
run_chunk_test "large_10mb.bin" "large (10MB)"

# 4. Healthcheck
echo -n "Testing GET /chunks/healthcheck... "
HEALTH=$(curl -s "$BASE_URL/chunks/healthcheck")
if [[ $HEALTH == *"healthy"* ]]; then
    echo "PASS"
else
    echo "FAIL (Unhealthy status: $HEALTH)"
    exit 1
fi

# 5. Test Error Cases
echo "Testing Error Cases:"
echo "dummy" > dummy.bin
test_status "PUT" "/chunk/0000000000000000000000000000000000000000000000000000000000000000" 400 "dummy.bin"
test_status "GET" "/chunk/ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff" 404

echo ""
echo "--- ALL TESTS PASSED SUCCESSFULLY ---"
