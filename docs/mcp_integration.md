# MCP Integration Documentation

## Overview
The DeepRiskModel integrates with the Financial Datasets MCP server to fetch real market data for risk factor modeling. This integration provides access to historical market data, financial statements, and other market information.

## Setup

### 1. MCP Server Configuration
```bash
# Clone the MCP server
git clone https://github.com/sethdford/mcp-server.git
cd mcp-server

# Install dependencies
uv venv
source .venv/bin/activate
uv add "mcp[cli]" httpx

# Set up environment variables
echo "FINANCIAL_DATASETS_API_KEY=971f2844-76b9-4f37-8f9e-fc933ab8c170" > .env

# Start the server
mcp server
```

### 2. DeepRiskModel Integration
The integration is handled through the `MCPClient` struct in `src/mcp_client.rs`:

```rust
use deep_risk_model::MCPClient;

// Initialize client
let client = MCPClient::new("http://localhost:8000".to_string());

// Fetch market data
let market_data = client.fetch_market_data(
    &symbols,
    start_date,
    end_date
).await?;
```

## Features

### Market Data Fetching
- Fetches historical market data for specified symbols
- Supports custom date ranges
- Handles multiple features and returns

### Data Conversion
- Automatically converts MCP data format to DeepRiskModel format
- Handles feature matrix construction
- Manages stock identifiers and timestamps

### Integration Example
```rust
use deep_risk_model::{DeepRiskModel, MCPClient, ModelConfig};
use chrono::{Duration, Utc};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize client
    let client = MCPClient::new("http://localhost:8000".to_string());
    
    // Define symbols
    let symbols = vec!["AAPL", "MSFT", "GOOGL"];
    
    // Set date range
    let end_date = Utc::now();
    let start_date = end_date - Duration::days(30);
    
    // Fetch and process data
    let market_data = client.fetch_market_data(&symbols, start_date, end_date).await?;
    
    // Use with DeepRiskModel
    let model = DeepRiskModel::new(config, device)?;
    model.train(&market_data).await?;
}
```

## Testing

### Unit Tests
```bash
cargo test --package deep_risk_model --lib tests::mcp_client
```

### Integration Tests
```bash
cargo test --package deep_risk_model --lib tests::e2e_tests
```

## Error Handling
The integration includes comprehensive error handling:
- Network errors during data fetching
- Data format conversion errors
- Invalid date ranges or symbols
- Missing or malformed data

## Performance Considerations
- Uses async/await for efficient I/O
- Supports batch processing of market data
- Handles large datasets efficiently
- Includes connection pooling

## Security
- API key management through environment variables
- Secure HTTPS communication
- Input validation and sanitization

## Limitations
- Requires running MCP server locally
- Limited to available market data
- Rate limiting based on API key
- Network dependency for data fetching

## Future Improvements
1. Add caching layer for frequently accessed data
2. Implement retry mechanisms for failed requests
3. Add support for real-time data streaming
4. Enhance error reporting and recovery
5. Add data validation and cleaning 