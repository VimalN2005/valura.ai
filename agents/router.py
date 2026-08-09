import re
import json
from decimal import Decimal
from typing import Dict, List, Optional, Any
from pydantic import ValidationError

from schema.response import AnswerResponse
from agents.models import get_model
from agents.specialists import book_qa, kyc_profile, notes_desk, market_desk, compliance
from tools.book_tools import run_context, reset_run_context, register_flag, UncoveredSymbolException

# Map role name to Agent instance
AGENT_MAP = {
    "book_qa": book_qa,
    "kyc_profile": kyc_profile,
    "notes_desk": notes_desk,
    "market_desk": market_desk,
    "compliance": compliance
}

def extract_date_from_prompt(prompt: str) -> Optional[str]:
    """Helper to extract an ISO date (YYYY-MM-DD) from the prompt."""
    match = re.search(r"\b(202\d-\d{2}-\d{2})\b", prompt)
    if match:
        return match.group(1)
    return None

def classify_intent_via_llm(prompt: str) -> List[str]:
    """Uses valura-fast to classify the question's required specialists."""
    from agno.agent import Agent
    classifier_agent = Agent(
        model=get_model("valura-fast"),
        description="Intent classifier for financial query router",
        instructions=[
            "You are the routing gateway for a financial assistant.",
            "Analyze the user's question and classify which specialist role(s) are required to answer it.",
            "Output only a JSON array of strings containing the roles, for example: [\"book_qa\"] or [\"book_qa\", \"market_desk\"]."
        ]
    )
    
    classification_prompt = f"""Analyze the user's question and classify which specialist role(s) are required.

Available roles:
- book_qa: for cash balances, transactions count, holdings, stock quantities, values, fee summaries, portfolio rebalance drifts.
- kyc_profile: for personal details, address, employment, DOB, PAN, bank account info, risk profile.
- notes_desk: for free-text notes, relationship manager logs, ops memos.
- market_desk: for instrument details (sectors, exchanges), historical close prices, and market news.
- compliance: for investment advice (recommendations, buy/sell suitability), target allocation suggestions, or questions trying to access multiple accounts/clients.

Question: {prompt}
Classified Roles:"""

    try:
        response = classifier_agent.run(classification_prompt)
        text = response.content.strip()
        # Clean markdown code blocks if any
        if text.startswith("```"):
            text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE)
        roles = json.loads(text.strip())
        if isinstance(roles, list):
            # Validate that they are valid roles
            valid_roles = [r for r in roles if r in AGENT_MAP]
            if valid_roles:
                return valid_roles
    except Exception as e:
        print(f"Error during intent classification: {e}")
    
    # Fallback heuristics if classification fails or LLM is down
    return []

def classify_intent_heuristics(prompt: str) -> List[str]:
    """Fallback keyword-based classifier."""
    prompt_lower = prompt.lower()
    
    # Personalised Advice & Scope
    if any(x in prompt_lower for x in ("should", "recommend", "advice", "suggest", "suitability", "good time to")):
        return ["compliance"]
        
    roles = []
    if any(x in prompt_lower for x in ("cash", "balance", "transaction", "deposit", "withdrawal", "fee", "dividend", "holding", "portfolio", "drift")):
        roles.append("book_qa")
    if any(x in prompt_lower for x in ("address", "pan", "kyc", "bank account", "dob", "birth", "employment", "risk profile")):
        roles.append("kyc_profile")
    if any(x in prompt_lower for x in ("note", "memo", "relationship manager")):
        roles.append("notes_desk")
    if any(x in prompt_lower for x in ("price", "sector", "industry", "news", "headline")):
        roles.append("market_desk")
        
    return roles if roles else ["book_qa"]

def generate_heuristic_answer(client_id: str, prompt: str) -> Optional[Dict[str, Any]]:
    """Generates a deterministic answer using Python tools if the LLM proxy is down."""
    prompt_lower = prompt.lower()
    
    # 1. Cash Balance
    if "cash balance" in prompt_lower or "current cash" in prompt_lower:
        from tools.book_tools import get_client_cash_balance
        res = get_client_cash_balance(client_id)
        val = res["cash_balance"]
        return {
            "answer": f"The current cash balance for client {client_id} is USD {val}.",
            "answer_value": val,
            "abstained": False,
            "refused": False,
            "reason": None,
            "citations": res["citations"],
            "confidence": 0.9,
            "flags": ["upstream_issue"],
            "agents": ["router", "book_qa"]
        }
        
    # 2. KYC / PAN / DOB / Address / Bank
    if "pan" in prompt_lower or "bank account" in prompt_lower or "kyc" in prompt_lower:
        from tools.book_tools import get_client_kyc
        res = get_client_kyc(client_id)
        if "pan" in prompt_lower:
            val = res["pan"]
            ans = f"The PAN for client {client_id} is {val}."
        elif "bank account" in prompt_lower or "account number" in prompt_lower:
            val = res["bank_account"]["account_number"]
            ans = f"The bank account number for client {client_id} is {val}."
        else:
            val = res["kyc_status"]
            ans = f"The KYC status for client {client_id} is {val}."
        return {
            "answer": ans,
            "answer_value": val,
            "abstained": False,
            "refused": False,
            "reason": None,
            "citations": res["citations"],
            "confidence": 0.9,
            "flags": ["upstream_issue"],
            "agents": ["router", "kyc_profile"]
        }
        
    # 3. Fees summary
    if "fee" in prompt_lower or "fees" in prompt_lower:
        from tools.book_tools import get_client_transactions
        year_match = re.search(r"\b(202\d)\b", prompt)
        year = year_match.group(1) if year_match else None
        
        res = get_client_transactions(client_id, limit=2000, type_filter="fee")
        total_fee = Decimal("0.00")
        fee_txs = []
        for t in res["transactions"]:
            if year and t["date"][:4] != year:
                continue
            total_fee += Decimal(t["amount_usd"])
            fee_txs.append(t["id"])
            
        val = str(total_fee.quantize(Decimal("0.01")))
        ans = f"Total platform fees charged to client {client_id}" + (f" in {year}" if year else "") + f" were USD {val}."
        
        # Citation threshold check
        citations = [client_id] if len(fee_txs) > 6 else fee_txs
        return {
            "answer": ans,
            "answer_value": val,
            "abstained": False,
            "refused": False,
            "reason": None,
            "citations": citations,
            "confidence": 0.9,
            "flags": ["upstream_issue"],
            "agents": ["router", "book_qa"]
        }

    return None

def extract_answer_value_via_llm(answer_text: str, question: str) -> Optional[str]:
    """Uses valura-fast to parse the single target value/date/number from the text."""
    from agno.agent import Agent
    parser_agent = Agent(
        model=get_model("valura-fast"),
        description="Value extractor parser",
        instructions=["Extract the single final figure, count, or date. Output ONLY the extracted value as a string (or the word 'null')."]
    )
    prompt = f"""Given the following user question and the specialist's answer, extract the single figure, count, or date that answers the question.

Output format rules:
- Money/currency values: format as a clean decimal string with 2 decimal places, no currency symbol, no thousands separator (e.g. "71.88", "15386.78", "0.00").
- Counts/Integers: format as a plain integer string (e.g. "3").
- Dates: format as an ISO date string (e.g. "2025-09-14").
- If the question does not have a single numeric/date/count answer (e.g. a general question or yes/no), output "null".
- Output ONLY the extracted value as a string (or the word "null"), with no extra spaces, quotes, or sentences.

Question: {question}
Answer: {answer_text}
Extracted Value:"""

    try:
        response = parser_agent.run(prompt)
        val = response.content.strip().replace('"', '').replace("'", "")
        if val.lower() == "null" or not val:
            return None
        return val
    except Exception:
        return None

def extract_answer_value_heuristics(answer_text: str) -> Optional[str]:
    """Fallback regex extractor for value/number/date."""
    # Try to find a decimal number (like 12345.67)
    dec_match = re.search(r"\b\d+\.\d{2}\b", answer_text)
    if dec_match:
        return dec_match.group(0)
    # Try to find an ISO date (like 2025-09-14)
    date_match = re.search(r"\b(202\d-\d{2}-\d{2})\b", answer_text)
    if date_match:
        return date_match.group(0)
    # Try to find integer count
    int_match = re.search(r"\b\d+\b", answer_text)
    if int_match:
        return int_match.group(0)
    return None

# --- Main Route Endpoint ---

def route_question(question_id: str, client_id: str, prompt: str) -> Dict[str, Any]:
    """
    Main entry point for processing a question.
    Sets up context, runs intent classification, dispatches agents, handles failures and constraints.
    """
    as_at = extract_date_from_prompt(prompt)
    reset_run_context(client_id, as_at)
    
    # 1. Check for compliance advice directly in the prompt
    prompt_lower = prompt.lower()
    is_advice = any(x in prompt_lower for x in ("should", "recommend", "advice", "suggest", "suitability", "good time to"))
    
    # Check for cross-client queries
    other_cli_match = re.search(r"\b(cli_\d{4})\b", prompt)
    is_cross_client = other_cli_match is not None and other_cli_match.group(1) != client_id
    
    if is_advice or is_cross_client:
        # Direct compliance refusal
        reason = "Compliance refusal: personalised investment advice is not permitted." if is_advice else "Compliance refusal: accessing other client records is strictly forbidden."
        return {
            "question_id": question_id,
            "answer": "",
            "answer_value": None,
            "abstained": False,
            "refused": True,
            "reason": reason,
            "citations": [],
            "confidence": 1.0,
            "flags": [],
            "agents": ["router", "compliance"]
        }

    # 2. Intent Classification
    roles = []
    llm_error = False
    try:
        roles = classify_intent_via_llm(prompt)
    except Exception as e:
        # LLM proxy is down or rate-limited
        print(f"LLM error during classification: {e}")
        llm_error = True
        
    if not roles:
        roles = classify_intent_heuristics(prompt)

    # 3. Fallback Heuristics on LLM failures (Blackout)
    if llm_error:
        heuristic_res = generate_heuristic_answer(client_id, prompt)
        if heuristic_res:
            heuristic_res["question_id"] = question_id
            return heuristic_res
        else:
            # Decline honestly
            return {
                "question_id": question_id,
                "answer": "",
                "answer_value": None,
                "abstained": True,
                "refused": False,
                "reason": "Upstream service blackout: LLM proxy is currently unavailable.",
                "citations": [],
                "confidence": 0.0,
                "flags": ["upstream_issue"],
                "agents": ["router"]
            }

    # 4. Run Specialist Agents
    agent_outputs = []
    agent_path = ["router"]
    
    try:
        for role in roles:
            agent = AGENT_MAP[role]
            agent_path.append(role)
            # Run the agent
            run_res = agent.run(prompt)
            agent_outputs.append(run_res.content)
    except UncoveredSymbolException as exc:
        # Uncovered symbol caught at python tool layer
        return {
            "question_id": question_id,
            "answer": "",
            "answer_value": None,
            "abstained": True,
            "refused": False,
            "reason": f"No price or market data is available for symbol {exc.symbol} (uncovered symbol).",
            "citations": [],
            "confidence": 1.0,
            "flags": [],
            "agents": agent_path
        }
    except Exception as e:
        # Handle other tool failures or proxy errors
        print(f"Exception during specialist execution: {e}")
        # Try heuristics if possible
        heuristic_res = generate_heuristic_answer(client_id, prompt)
        if heuristic_res:
            heuristic_res["question_id"] = question_id
            return heuristic_res
        else:
            # Check if it was a rate limit / proxy error
            register_flag("upstream_issue")
            return {
                "question_id": question_id,
                "answer": "",
                "answer_value": None,
                "abstained": True,
                "refused": False,
                "reason": f"Upstream execution error: {str(e)}",
                "citations": [],
                "confidence": 0.0,
                "flags": ["upstream_issue"],
                "agents": agent_path
            }

    # 5. Synthesize Combined Answer if multiple specialists ran
    if len(agent_outputs) > 1:
        from agno.agent import Agent
        synthesizer_agent = Agent(
            model=get_model("valura-fast"),
            description="Synthesis Specialist",
            instructions=[
                "You are a financial synthesizer.",
                "Combine the findings from different specialists into a single, cohesive, professional natural language answer.",
                "Do not perform any arithmetic yourself. Merely synthesize the findings."
            ]
        )
        synthesis_prompt = f"""Combine the following findings from different specialists into a single, cohesive, professional natural language answer to the question: "{prompt}".

Findings:
{chr(10).join(agent_outputs)}

Answer:"""
        try:
            res = synthesizer_agent.run(synthesis_prompt)
            final_answer = res.content.strip()
        except Exception:
            # Fallback join
            final_answer = " ".join(agent_outputs)
    else:
        final_answer = agent_outputs[0] if agent_outputs else ""

    # 6. Extract Answer Value
    answer_value = None
    try:
        answer_value = extract_answer_value_via_llm(final_answer, prompt)
    except Exception:
        pass
        
    if not answer_value:
        answer_value = extract_answer_value_heuristics(final_answer)

    # 7. Collect Citations and Flags from Context
    citations = list(run_context.citations)
    # Apply Citation Compression Rule (> 6 records -> cite client_id)
    if len(citations) > 6:
        citations = [client_id]
        
    flags = list(run_context.flags)
    
    # 8. Post-Process validation against schema
    response_dict = {
        "question_id": question_id,
        "answer": final_answer,
        "answer_value": answer_value,
        "abstained": False,
        "refused": False,
        "reason": None,
        "citations": citations,
        "confidence": 0.95,
        "flags": flags,
        "agents": agent_path
    }
    
    try:
        # Validate using Pydantic model to ensure absolute schema compliance
        validated = AnswerResponse(**response_dict)
        return validated.model_dump()
    except ValidationError as ve:
        # If Pydantic validation fails, return safe default response
        print(f"Validation error in response dict: {ve}")
        return {
            "question_id": question_id,
            "answer": "",
            "answer_value": None,
            "abstained": True,
            "refused": False,
            "reason": f"Contract validation error: {str(ve)}",
            "citations": [],
            "confidence": 0.0,
            "flags": ["upstream_issue"],
            "agents": agent_path
        }
