"""
QOR Custom Tools — Add Your Own APIs
=======================================
This file shows how to connect QOR to ANY external data source.

The model calls these tools automatically when:
  1. It doesn't know the answer (low confidence)
  2. The question is about live/changing data
  3. Cached data is stale

After getting data from a tool, QOR:
  1. Answers the user with the fresh data
  2. Saves the data to memory
  3. Next time the same question comes → answers from memory (faster!)
  4. When memory gets stale → calls the tool again automatically

HOW TO ADD YOUR OWN TOOLS:
===========================
1. Write a function that takes a query string and returns a string
2. Register it with the confidence gate
3. That's it. QOR calls it automatically when needed.
"""

from qor.confidence import ConfidenceGate


def setup_example_tools(gate: ConfidenceGate):
    """
    Register example tools. Copy and modify these for your use case.
    """

    # =========================================================================
    # EXAMPLE 1: Database lookup (replace with your actual database)
    # =========================================================================
    def product_database(query: str) -> str:
        """Look up product info from your database."""
        # Replace this with actual database code:
        #   import sqlite3
        #   conn = sqlite3.connect("products.db")
        #   cursor = conn.execute("SELECT * FROM products WHERE name LIKE ?", (f"%{query}%",))
        #   results = cursor.fetchall()
        #   return format_results(results)

        # Demo data:
        products = {
            "laptop": "QOR Laptop Pro: 16GB RAM, 512GB SSD, $999",
            "phone": "QOR Phone X: 128GB, 6.5 inch screen, $699",
            "tablet": "QOR Tablet Air: 256GB, 10.9 inch, $449",
        }
        for key, val in products.items():
            if key in query.lower():
                return val
        return "Product not found in database"

    gate.tools.register(
        "product_db",
        "Look up product information from internal database",
        product_database,
        ["product", "inventory"]  # Categories this tool handles
    )

    # =========================================================================
    # EXAMPLE 2: REST API call
    # =========================================================================
    def company_api(query: str) -> str:
        """Call your company's internal API."""
        # Replace with actual API call:
        #   import requests
        #   response = requests.get(f"https://api.yourcompany.com/search?q={query}")
        #   return response.json()["answer"]

        return f"[Configure your API endpoint in custom_tools.py]"

    gate.tools.register(
        "company_api",
        "Search company internal knowledge base",
        company_api,
        ["company", "policy", "procedure"]
    )

    # =========================================================================
    # EXAMPLE 3: Weather API (free with OpenWeatherMap)
    # =========================================================================
    def weather_api(query: str) -> str:
        """Get weather data. Sign up at openweathermap.org for free API key."""
        # Replace YOUR_API_KEY:
        #   import requests
        #   API_KEY = "YOUR_API_KEY"
        #   city = extract_city(query)
        #   url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        #   data = requests.get(url).json()
        #   return f"Weather in {city}: {data['weather'][0]['description']}, {data['main']['temp']}°C"

        return "[Weather API not configured. Get free key at openweathermap.org]"

    gate.tools.register(
        "weather",
        "Get current weather for a location",
        weather_api,
        ["weather"]
    )

    # =========================================================================
    # EXAMPLE 4: News API
    # =========================================================================
    def news_api(query: str) -> str:
        """Get latest news. Uses DuckDuckGo search (no API key needed)."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=3))
                if results:
                    return "\n".join([
                        f"- {r['title']} ({r.get('date', 'recent')})" for r in results
                    ])
        except ImportError:
            pass
        return "[News search not available. Install: pip install duckduckgo-search]"

    gate.tools.register(
        "news_search",
        "Search for recent news articles",
        news_api,
        ["news"]
    )

    # =========================================================================
    # EXAMPLE 5: Calculator (always available, no API needed)
    # =========================================================================
    def calculator(query: str) -> str:
        """Simple math calculator."""
        import re
        # Extract math expression
        # Find patterns like "2+2", "100/5", "sqrt(16)", etc.
        expr = re.findall(r'[\d\.\+\-\*/\(\)\s]+', query)
        if expr:
            try:
                result = eval(expr[0].strip())  # Simple eval for math
                return f"Result: {result}"
            except Exception:
                pass
        return "Could not parse math expression"

    gate.tools.register(
        "calculator",
        "Perform mathematical calculations",
        calculator,
        ["math", "calculate"]
    )

    # =========================================================================
    # EXAMPLE 6: Document/File reader
    # =========================================================================
    def file_reader(query: str) -> str:
        """Read a specific file if mentioned in the query."""
        import os
        import re

        # Look for filenames in the query
        filenames = re.findall(r'[\w\-]+\.(?:txt|md|csv|json)', query)
        for fname in filenames:
            search_dirs = [".", "data", "knowledge", "learn", "documents"]
            for dir_path in search_dirs:
                full_path = os.path.join(dir_path, fname)
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()[:2000]  # First 2000 chars
                    return f"Contents of {fname}:\n{content}"

        return f"File not found"

    gate.tools.register(
        "file_reader",
        "Read contents of a local file",
        file_reader,
        ["file", "document"]
    )


# =========================================================================
# HOW TO ADD YOUR OWN TOOLS — Template
# =========================================================================
"""
def my_tool(query: str) -> str:
    '''
    Your tool function.
    - Takes: query string (the user's question)
    - Returns: string with the answer/data
    '''
    # Your code here — API call, database query, file read, anything
    result = call_my_api(query)
    return str(result)

# Register it:
gate.tools.register(
    "my_tool_name",           # Unique name
    "What this tool does",    # Description  
    my_tool,                  # The function
    ["keyword1", "keyword2"]  # What category of live data it handles
)

# That's it! QOR will automatically call your tool when:
# - User asks about something matching those categories
# - Model confidence is low
# - Cached data is stale

# The tool result gets:
# - Used to answer the current question
# - Saved to memory for future use
# - Refreshed automatically when data gets stale
"""


# =========================================================================
# QUICK START — Full setup example
# =========================================================================
def quick_start():
    """
    Complete example: set up QOR with zero-hallucination system.
    Copy this into your own script and modify.
    """
    import torch
    from qor.config import QORConfig
    from qor.model import QORModel
    from qor.tokenizer import QORTokenizer
    from qor.confidence import ConfidenceGate
    from qor.rag import QORRag

    # 1. Load your trained model
    config = QORConfig.small()
    tokenizer = QORTokenizer()
    tokenizer.load("tokenizer.json")
    config.model.vocab_size = tokenizer.vocab_size

    checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu",
                             weights_only=False)
    model = QORModel(config.model)
    model.load_state_dict(checkpoint["model_state"])

    # 2. Create confidence gate
    gate = ConfidenceGate(model, tokenizer)

    # 3. Add RAG knowledge base (optional)
    rag = QORRag()
    if os.path.exists("knowledge"):
        rag.add_folder("knowledge/")
        gate.set_rag(rag)

    # 4. Add custom tools (optional)
    setup_example_tools(gate)

    # 5. Start chatting with zero hallucination!
    gate.interactive_chat()


import os
if __name__ == "__main__":
    quick_start()
