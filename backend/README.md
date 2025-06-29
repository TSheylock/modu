# SASOK Platform Backend

The backend service for the SASOK Intelligence Platform, providing AI processing, Web3 integration, and analytics capabilities.

## Features

- **AI Processing**
  - Emotion Analysis
  - NLP Processing
  - Real-time Response Generation
  - Model Learning & Updates

- **Web3 Integration**
  - Wallet Connection
  - Smart Contract Interaction
  - Transaction History
  - NFT Data Management

- **Analytics**
  - User Interaction Tracking
  - Emotion Data Analysis
  - Transaction Analytics
  - Platform Statistics

## Setup

1. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your configuration values
```

4. **Run Development Server**
```bash
uvicorn main:app --reload --port 8000
```

## API Documentation

### AI Processing Endpoints

#### POST /api/ai/emotion-analysis
Analyze emotions in text or image data.
```json
{
    "data": "text or base64 image",
    "type": "text|image"
}
```

#### POST /api/ai/nlp-process
Process text for intent classification and entity extraction.
```json
{
    "text": "user input text"
}
```

### Web3 Integration Endpoints

#### POST /api/web3/connect
Connect a wallet address.
```json
{
    "wallet_address": "0x..."
}
```

#### GET /api/web3/transactions/{address}
Get transaction history for an address.

### Analytics Endpoints

#### POST /api/analytics/log
Log user interaction data.
```json
{
    "user_id": "user123",
    "interaction_type": "click",
    "data": {}
}
```

#### GET /api/analytics/stats
Get platform statistics.

## Project Structure

```
backend/
├── main.py              # FastAPI application
├── ai_processor.py      # AI processing module
├── web3_handler.py      # Web3 integration module
├── analytics_manager.py # Analytics module
├── config.py           # Configuration management
├── requirements.txt    # Python dependencies
└── .env.example       # Environment variables template
```

## Development

### Adding New Features

1. Create new module in appropriate directory
2. Update main.py with new endpoints
3. Update configuration if needed
4. Add tests for new functionality
5. Update documentation

### Testing

```bash
pytest tests/
```

### Code Style

Follow PEP 8 guidelines and use type hints.

## Production Deployment

1. **Environment Setup**
   - Set up production environment variables
   - Configure secure database connection
   - Set up proper Web3 provider

2. **Security Considerations**
   - Enable CORS with specific origins
   - Set up rate limiting
   - Configure proper authentication

3. **Monitoring**
   - Set up logging
   - Configure performance monitoring
   - Enable error tracking

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Create pull request

## License

MIT License
