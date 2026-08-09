import os
import argparse
import json
from dotenv import load_dotenv

from harness.client import ArenaClient
from agents.router import route_question
from agents.models import get_base_url_and_key

def main():
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Valura AI Arena Runner")
    parser.add_argument(
        "--mode",
        choices=["practice", "qualifying", "final"],
        default="practice",
        help="Run mode (default: practice)"
    )
    args = parser.parse_args()

    # Get API key and base URL from environment
    base_url, api_key = get_base_url_and_key()
    
    print("=" * 60)
    print(f"Starting Valura AI Arena Runner in [{args.mode.upper()}] mode")
    print(f"Gateway URL: {base_url}")
    print(f"API Key: {api_key[:8]}...{api_key[-8:] if len(api_key) > 16 else ''}")
    print("=" * 60)

    # 1. Initialize API Client
    client = ArenaClient(base_url=base_url, api_key=api_key, mode=args.mode)

    # 2. Fetch and Cache Fresh Data Files locally (critical since each attempt generates a new book)
    os.makedirs("data", exist_ok=True)
    print("Downloading rules...")
    rules = client.get_rules()
    with open("data/rules.json", "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)
        
    print("Downloading client book...")
    book = client.get_book()
    with open("data/book.json", "w", encoding="utf-8") as f:
        json.dump(book, f, indent=2)
        
    print("Downloading market data...")
    market = client.get_market()
    with open("data/market.json", "w", encoding="utf-8") as f:
        json.dump(market, f, indent=2)
    print("All files downloaded and cached under data/ directory.")

    # 3. Post Roster
    # The scoring harness requires declaring the agents that will run
    roster = {
        "framework": "agno",
        "framework_version": "1.5.0",
        "agents": [
            {"role": "router", "name": "Routing Specialist", "model": "valura-fast"},
            {"role": "book_qa", "name": "Book QA Specialist", "model": "valura-fast"},
            {"role": "kyc_profile", "name": "KYC Profile Specialist", "model": "valura-fast"},
            {"role": "notes_desk", "name": "Notes Desk Specialist", "model": "valura-deep"},
            {"role": "market_desk", "name": "Market Desk Specialist", "model": "valura-fast"},
            {"role": "compliance", "name": "Compliance Specialist", "model": "valura-fast"}
        ]
    }
    print("Declaring agent roster to gateway...")
    client.post_roster(roster)
    print("Roster successfully declared.")

    # 4. Question Loop
    print("\nStarting question-answering loop...")
    
    while True:
        try:
            # Fetch the next unanswered question
            question_data = client.get_next_question()
            
            # If progress shows we are done, or if response is empty
            if not question_data:
                print("No question returned. Run might be finished.")
                break
                
            question_id = question_data.get("question_id")
            client_id = question_data.get("client_id")
            prompt = question_data.get("prompt")
            progress = question_data.get("progress", {})
            
            answered = progress.get("answered", 0)
            total = progress.get("total", 90)
            
            print("-" * 50)
            print(f"Question [{answered + 1}/{total}]: {question_id}")
            print(f"Client: {client_id}")
            print(f"Prompt: {prompt}")
            print("-" * 50)

            # Process the question through our multi-agent system
            response_payload = route_question(
                question_id=question_id,
                client_id=client_id,
                prompt=prompt
            )
            
            print(f"Answer: {response_payload.get('answer')}")
            print(f"Value: {response_payload.get('answer_value')}")
            print(f"Flags: {response_payload.get('flags')}")
            print(f"Agents: {response_payload.get('agents')}")
            print(f"Abstained: {response_payload.get('abstained')}, Refused: {response_payload.get('refused')}")
            
            # Post the answer
            answer_status = client.post_answer(response_payload)
            print(f"Submission status: {answer_status}")
            
            # If we reached the end of the total questions, stop
            if answered + 1 >= total:
                print("All questions in the stream processed!")
                break
                
        except Exception as e:
            print(f"Error in execution loop: {e}")
            # If it's a fatal gateway connection error or similar, wait a moment and try next
            time_to_sleep = 5
            print(f"Sleeping for {time_to_sleep}s before trying to reconnect...")
            import time
            time.sleep(time_to_sleep)

    # 5. Print final progress scorecard
    print("\n" + "=" * 60)
    print("Run Completed! Fetching final scorecard...")
    try:
        scorecard = client.get_me()
        print(json.dumps(scorecard, indent=2))
    except Exception as e:
        print(f"Could not retrieve scorecard: {e}")
    print("=" * 60)

if __name__ == "__main__":
    main()
