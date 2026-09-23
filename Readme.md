AI Blackjack

A simple blackjack game in Python 3.12+ where one AI dealer deals cards to you and three (can be configured in code to add more) AI players: John (cautious personality), Sara (balanced personality) and Matt (risk-taker personality). It's built with LangChain and works with any OpenAI-compatible LLM endpoint, such as Ollama, LM Studio, OpenAI or OpenRouter.


Rules
- Cards are worth 2 to 11. Each player can take up to 3 cards.
- Only the dealer can draw cards, through its `deal_card` tool. Players ask the dealer in plain language.
- A total over 21 is a bust. The highest total of 21 or less wins, and ties are shared.


Setup
pip install -r requirements.txt
cp .env.example .env   # then edit LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
python blackjack.py


Playing
You go first. Talk to the dealer naturally, for example "deal me a card", "hit me again" or "I'll stand". Type `quit` to leave. Then the AI players take their turns, and the results table names the winner.

Files
- card.py holds draw_card(), the card-drawing function.
- blackjack.py holds the dealer agent, the AI players, the game loop and the results.
