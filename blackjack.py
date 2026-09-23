

import os
import sys
import json
import re

from dotenv import load_dotenv # For LLM config and API key

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from cards import draw_card

MAX_CARDS = 3
LIMIT = 21

AI_PLAYERS = {
    "John": ("cautious. you hate busting and usually stand once you reach 14 or more", 14),
    "Sara": ("balanced. you usually stand at 17 or more", 17),
    "Matt": ("a risk-taker. you like to push your luck and keep drawing until 19 or more", 19),
}

DEALER_PROMPT = f"""
You are the dealer in a simplifed Blackjack game.
Rules:
Card values are 2 to 11. Each player may hold at most {MAX_CARDS} cards. 
A total over {LIMIT} is a bust. The highest total of {LIMIT} or less wins.
Only you can draw cards, using your tools.
The player whose turn it is will talk to you:
- If they ask for a card in any wording (hit, deal me, another one etc.), call deal_card.
- If they want to stop (stand, stay, holder, I'm done etc.), call end_turn.
- Otherwise, answer briefly without calling a tool.
After a tool runs, announce the result to the table in one short, friendly sentence.
"""


hands: dict[str, list[int]] = {}
finished: set[str] = set()
turn = {"player": "", "dealt": False} # whose turn it is; one card per request


# Add up total score for the player
def total(name: str):
    return sum(hands[name])

# Flag as busted if a player's score crosses limit
def busted(name: str):
    return total(name) > LIMIT

# Move to the next player if ...
def turn_over(name: str):
    return name in finished or busted(name) or len(hands[name]) >= MAX_CARDS

@tool
def deal_card() -> str:
    """
    Draw the next card for player who turn it is. Use only when they ask for a card.
    """
    name = turn["player"]
    if turn["dealt"]:
        return "Refused: only one card per request."
    if turn_over(name):
        return f"Refused: {name} cannot take more cards (turn over, bust, or {MAX_CARDS} cards)."
    card = draw_card()
    hands[name].append(card)
    turn["dealt"] = True
    status = "BUST!" if busted(name) else ""
    return f"Dealth a {card} to {name}. Hand: {hands[name]}, total {total(name)}.{status}"

@tool
def end_turn() -> str:
    """
    End the current player's turn when they want to stand/stop taking cards
    """
    name = turn["player"]
    finished.add(name)
    return f"{name} stands with a total of {total(name)}."

TOOLS = {t.name: t for t in (deal_card, end_turn)}


def dealer_respond(dealer, name: str, message: str) -> str:
    """
    Pass a player's request to the dealer agent, running any tool calls it makes.
    dealer: LLM defined in main()
    name: player whose turn it is
    message: what the player said. e.g. deal me a card
    """

    turn.update(player=name, dealt=False)
    msgs = [
        SystemMessage(DEALER_PROMPT),
        HumanMessage(f"{name} (hand {hands[name]}, total {total(name)}) says: {message}"),
    ]

    for i in range(3):
        reply = dealer.invoke(msgs)
        if not reply.tool_calls:
            return reply.content.strip()
        msgs.append(reply)
        for call in reply.tool_calls:
            result = TOOLS[call["name"]].invoke(call["args"])
            msgs.append(ToolMessage(result, tool_call_id=call["id"]))
    return result


def player_decide(llm, name: str) -> tuple[str, str]:
    """
    Ask an AI player whether to hit or stand.
    Returns action, message to the dealer
    """
    style, threshold = AI_PLAYERS[name]
    if not hands[name]:
        return "hit", "Deal me my first card"
    
    others = ",".join(f"{n}: {total(n)}{' (bust)' if busted(n) else ''}"
                      for n in hands if n != name and turn_over(n)) or "none yet"

    prompt = f"""
    You are {name}, a Blackjack player who is {style}.
    Rules: cards are 2 - 11, at most {MAX_CARDS} cards, over {LIMIT} is bust, highest total of {LIMIT} or less wins.
    Your cards: {hands[name]} (total {total(name)}). You may take {MAX_CARDS - len(hands[name])} more 
    Finished players: {others}.
    Decide whether to hit or stand. Reply with JSON only:
    {{"action": "hit" or "stand", "message": "a short in-character line to the dealer"}}
    """

    text = llm.invoke(prompt).content
    try:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S) # extract thinking
        data_json_str = re.search(r"\{.*\}", text, re.S).group()
        data = json.loads(data_json_str)
        action, message = data["action"].strip().lower(), data["message"].strip()

        if action not in ("hit", "stand"):
            raise ValueError(action)
    except (AttributeError, KeyError, ValueError):
        action = "hit" if total(name) < threshold else "stand"
        message = "Hit me" if action == "hit" else "I will stand"
    return action, message


def play_turn(dealer, llm, name: str, is_user: bool) -> None:
    """
    Runs one players one full turn
    """

    print(f"\n --- {name}'s turn ---")
    misses = 0
    while not turn_over(name):
        if is_user:
            print(f"Your hand: {hands[name]} (total {total(name)}). "
                  f"Talk to the dealer, e.g. 'deal me a card' or 'I will stand'.")
            message = input(f"{name}>").strip()
            if not message:
                continue
            if message.lower() in ("quit", "exit"):
                sys.exit("Exiting")
        else:
            action, message = player_decide(llm, name)
            print(f"{name}: {message}")
        before = len(hands[name])
        print(f"Dealer: {dealer_respond(dealer, name, message)}")

        if not is_user:
            if action == "stand":
               finished.add(name)
            elif len(hands[name]) == before:
                misses += 1
                if misses >= 2:
                    # dealer is not dealing. Stop the loop
                    finished.add(name)
    outcome = f"Bust" if busted(name) else "done"
    print(f"{name} finishes with {hands[name]} = {total(name)} ({outcome})")


def show_results() -> None:
    """
    Summarize results
    """
    print(f"\n -------- Results --------")
    for name, cards in hands.items():
        status = f"Bust" if busted(name) else "OK"
        print(f" {name} {str(cards)} total {total(name)} {status}")
    valid = {n: total(n) for n in hands if not busted(n) and hands[n]}
    if not valid:
        print(f"\n No winner - everyone busted!")
        return
    best = max(valid.values())
    winners = [n for n, t in valid.items() if t == best]
    label = "Winner" if len(winners) == 1 else "It is a tie"
    print(f"\n{label}: {' & '.join(winners)} with {best}")

def build_llms():
    load_dotenv()
    base_url, model = os.getenv("LLM_BASE_URL"), os.getenv("LLM_MODEL")

    if not base_url or not model:
        sys.exit("Set LLM_BASE_URL, LLM_MODEL AND LLM_API_KEY in .env file")

    common = dict(base_url=base_url, model=model, api_key=os.getenv("LLM_API_KEY"), timeout = 120)
    dealer_llm = ChatOpenAI(temperature=0.2, **common)
    player_llm = ChatOpenAI(temperature=0.8, **common)

    try:
        player_llm.invoke("Reply with OK")
    except Exception as e:
        sys.exit(f"Could not reach the LLM at {base_url} (model '{model}'): {e}")
    return dealer_llm.bind_tools([deal_card, end_turn]), player_llm


def main():
    dealer, llm = build_llms()
    print(f"Welcome to Blackjack! Cards are 2-11, up to {MAX_CARDS} cards each, "
          f"closes to {LIMIT} without going over wins.")

    user = input("What is your name? ").strip() or "You"
    while user in AI_PLAYERS:
        user = input(f"{user} already exists. Pick another name: ").strip() or "You"
    print(f"Players: {user}, {', '.join(AI_PLAYERS)}.")

    for name in (user, *AI_PLAYERS):
        hands[name] = []
    play_turn(dealer, llm, user, is_user=True)
    for name in AI_PLAYERS:
        play_turn(dealer, llm, name, is_user=False)
    show_results()


if __name__ == "__main__":
    main()









