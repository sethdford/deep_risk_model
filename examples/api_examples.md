# Deep Risk Model API Examples

This document provides example requests for testing the Deep Risk Model API endpoints.

## Prerequisites

1. Replace `API_ENDPOINT` with your actual API Gateway endpoint URL
2. For Python examples, install the requests library:
```bash
pip install requests
```

## Health Check

### cURL
```bash
curl -X GET "${API_ENDPOINT}/health"
```

### Python
```python
import requests

response = requests.get(f"{API_ENDPOINT}/health")
print(response.json())
```

Expected response:
```json
{
    "status": "healthy",
    "version": "0.1.0"
}
```

## Generate Risk Factors

### cURL
```bash
curl -X POST "${API_ENDPOINT}/factors" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
      ],
      [
        [0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7],
        [0.8, 0.9, 1.0]
      ]
    ],
    "returns": [0.05, 0.07]
  }'
```

### Python
```python
import requests
import numpy as np

# Generate sample data
n_stocks = 2
seq_len = 3
n_features = 3

features = np.random.randn(n_stocks, seq_len, n_features).tolist()
returns = np.random.randn(n_stocks).tolist()

payload = {
    "features": features,
    "returns": returns
}

response = requests.post(
    f"{API_ENDPOINT}/factors",
    json=payload,
    headers={"Content-Type": "application/json"}
)

print("Response status:", response.status_code)
print("Response body:", response.json())
```

Expected response:
```json
{
    "factors": [
        [0.123, 0.456, 0.789],
        [0.234, 0.567, 0.890]
    ],
    "covariance": [
        [0.111, 0.222],
        [0.222, 0.333]
    ]
}
```

## Estimate Covariance Matrix

### cURL
```bash
curl -X POST "${API_ENDPOINT}/covariance" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
      ],
      [
        [0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7],
        [0.8, 0.9, 1.0]
      ]
    ],
    "returns": [0.05, 0.07]
  }'
```

### Python
```python
import requests
import numpy as np

# Generate sample data
n_stocks = 2
seq_len = 3
n_features = 3

features = np.random.randn(n_stocks, seq_len, n_features).tolist()
returns = np.random.randn(n_stocks).tolist()

payload = {
    "features": features,
    "returns": returns
}

response = requests.post(
    f"{API_ENDPOINT}/covariance",
    json=payload,
    headers={"Content-Type": "application/json"}
)

print("Response status:", response.status_code)
print("Response body:", response.json())
```

Expected response:
```json
[
    [0.111, 0.222],
    [0.222, 0.333]
]
```

## Error Handling Examples

### Invalid Input Shape
```python
import requests

# Invalid feature shape
payload = {
    "features": [[1, 2], [3, 4]],  # Missing sequence dimension
    "returns": [0.05, 0.07]
}

response = requests.post(
    f"{API_ENDPOINT}/factors",
    json=payload,
    headers={"Content-Type": "application/json"}
)

print("Response status:", response.status_code)
print("Response body:", response.json())
```

### Missing Content-Type Header
```bash
curl -X POST "${API_ENDPOINT}/factors" \
  -d '{"features": [[[1, 2, 3]]], "returns": [0.05]}'
```

## Testing Script

Here's a complete Python script to test all endpoints:

```python
import requests
import numpy as np
import json

def test_health_check(api_endpoint):
    print("\nTesting health check endpoint...")
    response = requests.get(f"{api_endpoint}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_generate_factors(api_endpoint):
    print("\nTesting generate factors endpoint...")
    
    # Generate sample data
    n_stocks = 2
    seq_len = 3
    n_features = 3
    
    features = np.random.randn(n_stocks, seq_len, n_features).tolist()
    returns = np.random.randn(n_stocks).tolist()
    
    payload = {
        "features": features,
        "returns": returns
    }
    
    response = requests.post(
        f"{api_endpoint}/factors",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_estimate_covariance(api_endpoint):
    print("\nTesting estimate covariance endpoint...")
    
    # Generate sample data
    n_stocks = 2
    seq_len = 3
    n_features = 3
    
    features = np.random.randn(n_stocks, seq_len, n_features).tolist()
    returns = np.random.randn(n_stocks).tolist()
    
    payload = {
        "features": features,
        "returns": returns
    }
    
    response = requests.post(
        f"{api_endpoint}/covariance",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def main():
    # Replace with your API endpoint
    api_endpoint = "YOUR_API_ENDPOINT"
    
    try:
        test_health_check(api_endpoint)
        test_generate_factors(api_endpoint)
        test_estimate_covariance(api_endpoint)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

To use the testing script:

1. Save it as `test_api.py`
2. Replace `YOUR_API_ENDPOINT` with your actual API endpoint
3. Run it:
```bash
python test_api.py
``` 