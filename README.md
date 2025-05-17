# Website Scraper Telegram Bot & API

This project provides both a Telegram bot and a REST API for scraping websites and querying content using AI.

## Features

- Scrape websites and store their content
- Ask questions about scraped content
- Interactive menu with buttons
- PDF processing support
- Vector database for efficient content retrieval
- REST API for programmatic access

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with the following variables:
   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   GROQ_API_KEY=your_groq_api_key
   ```

4. Get your Telegram Bot Token:
   - Message [@BotFather](https://t.me/botfather) on Telegram
   - Create a new bot using `/newbot`
   - Copy the token provided

5. Get your Groq API Key:
   - Sign up at [Groq](https://console.groq.com)
   - Generate an API key from your dashboard

## Usage

### Telegram Bot

1. Start the bot:
   ```bash
   python main.py
   ```

2. In Telegram, start a chat with your bot and use the following commands:
   - `/start` - Start the bot and show the main menu
   - `/help` - Show help information
   - `/scrape <url>` - Scrape a website (e.g., `/scrape https://example.com`)
   - `/ask <question>` - Ask a question about scraped content

### REST API

1. Start the API server:
   ```bash
   python api.py
   ```

2. The API will be available at `http://localhost:8000`

3. API Endpoints:

   - **POST /scrape**
     ```json
     {
       "url": "https://example.com",
       "max_depth": 4,
       "max_pages": 50
     }
     ```

   - **GET /scrape/{task_id}**
     - Check the status of a scraping task

   - **POST /query**
     ```json
     {
       "question": "What products does this website offer?",
       "brand_name": "optional_brand_name"
     }
     ```

   - **POST /process-pdf**
     - Process a PDF file and add its content to the vector database

   - **GET /health**
     - Health check endpoint

4. API Documentation:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## Example API Usage

```python
import requests

# Start scraping
response = requests.post("http://localhost:8000/scrape", json={
    "url": "https://example.com"
})
task_id = response.json()["task_id"]

# Check scraping status
status = requests.get(f"http://localhost:8000/scrape/{task_id}").json()

# Query content
answer = requests.post("http://localhost:8000/query", json={
    "question": "What products does this website offer?"
}).json()
```

## Example

1. Start the bot:
   ```
   /start
   ```

2. Scrape a website:
   ```
   /scrape https://example.com
   ```

3. Ask questions:
   ```
   /ask What products does this website offer?
   ```

## Notes

- The bot uses Groq's LLM for generating responses
- Content is stored in a vector database for efficient retrieval
- Scraping depth and page limits can be configured in the code
- PDF files in the same directory will be automatically processed
- The API supports asynchronous operations for long-running tasks 