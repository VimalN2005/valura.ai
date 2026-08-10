import os
import json
import threading
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any

# Path to the data files
BOOK_PATH = "data/book.json"
MARKET_PATH = "data/market.json"

# --- Thread-Safe Run Context ---
class RunContext(threading.local):
    def __init__(self):
        self.citations = set()
        self.flags = set()
        self.client_id = None
        self.as_at = None

run_context = RunContext()

def reset_run_context(client_id: str, as_at: Optional[str] = None):
    """Re-initialize the thread-local context for a new question run."""
    run_context.citations = set()
    run_context.flags = set()
    run_context.client_id = client_id
    run_context.as_at = as_at

def enforce_client_scope(client_id: str):
    """Enforce that the requested client_id matches the scoped client_id."""
    if run_context.client_id and client_id != run_context.client_id:
        raise PermissionError(
            f"Cross-client data access denied. Question is scoped to {run_context.client_id}, but tool requested {client_id}."
        )

def get_resolved_client_id(client_id: str) -> str:
    """Helper to enforce scope and return the actual client ID (resolving names/IDs if needed)."""
    if not run_context.client_id:
        return client_id
        
    resolved_id = client_id
    if not client_id.startswith("cli_"):
        try:
            import re
            book = load_json_file(BOOK_PATH)
            scoped_client = next((c for c in book["clients"] if c["id"] == run_context.client_id), None)
            if scoped_client:
                # Helper to compare names removing spaces/punctuation
                clean_name = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
                
                # 1. Check Name
                if clean_name(client_id) == clean_name(scoped_client["name"]):
                    return run_context.client_id
                    
                # 2. Check Accounts
                for acc in scoped_client.get("accounts", []):
                    if client_id == acc.get("id") or client_id == acc.get("broker_ref"):
                        return run_context.client_id
                        
                # 3. Check KYC ID or PAN
                kyc = scoped_client.get("kyc", {})
                if client_id == kyc.get("id") or client_id == kyc.get("pan"):
                    return run_context.client_id
                    
                # 4. Check Suitability reviews
                for rev in scoped_client.get("suitability_reviews", []):
                    if client_id == rev.get("id"):
                        return run_context.client_id
                        
                # 5. Check transactions
                for tx in scoped_client.get("transactions", []):
                    if client_id == tx.get("id") or client_id == tx.get("symbol"):
                        return run_context.client_id
                        
                # 6. Check notes
                for note in scoped_client.get("notes", []):
                    if client_id == note.get("id"):
                        return run_context.client_id
        except Exception:
            pass
            
    if resolved_id != run_context.client_id:
        raise PermissionError(
            f"Cross-client data access denied. Question is scoped to {run_context.client_id}, but tool requested {client_id}."
        )
        
    return resolved_id

def register_citations(citations: List[str]):
    """Register citations in the run context."""
    for c in citations:
        if c:
            run_context.citations.add(str(c))

def register_flag(flag: str):
    """Register a flag in the run context."""
    run_context.flags.add(flag)


# --- Utility Functions ---

def load_json_file(filepath: str) -> Any:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Required data file {filepath} not found. Run the harness/download_starter.py script first.")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def quantize_decimal(val: Decimal) -> str:
    """Formats decimal to 2 decimal places with no thousands separator."""
    return str(val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def mask_value(val: str) -> str:
    """Masks a string, returning four asterisks followed by the last 4 characters."""
    if not val:
        return ""
    last_four = val[-4:] if len(val) >= 4 else val
    return f"****{last_four}"

# Special Exceptions
class UncoveredSymbolException(Exception):
    def __init__(self, symbol: str):
        super().__init__(f"No price or market data is available for symbol {symbol} (uncovered symbol).")
        self.symbol = symbol

class ClientNotFoundException(Exception):
    def __init__(self, client_id: str):
        super().__init__(f"Client {client_id} not found in the client book.")
        self.client_id = client_id


# --- Deterministic Data Queries ---

def get_client_kyc(client_id: str) -> Dict[str, Any]:
    """
    Retrieve client KYC details with strict masking.
    Checks for risk profile conflicts between KYC and Suitability Reviews.
    """
    client_id = get_resolved_client_id(client_id)
    
    book = load_json_file(BOOK_PATH)
    client = next((c for c in book["clients"] if c["id"] == client_id), None)
    if not client:
        raise ClientNotFoundException(client_id)

    kyc = client.get("kyc", {})
    kyc_id = kyc.get("id")
    raw_pan = kyc.get("pan", "")
    raw_account = kyc.get("bank_account", {}).get("account_number", "")
    
    masked_pan = mask_value(raw_pan)
    masked_bank_account = mask_value(raw_account)
    
    bank_info = kyc.get("bank_account", {})
    masked_bank_info = {
        "bank": bank_info.get("bank"),
        "account_number": masked_bank_account,
        "ifsc": bank_info.get("ifsc")
    }

    # Detect Risk Profile Conflict using as_at from context if set
    as_at = run_context.as_at
    reviews = client.get("suitability_reviews", [])
    if as_at:
        reviews = [r for r in reviews if r["date"] <= as_at]
    
    conflict = False
    conflict_details = None
    citations = [kyc_id] if kyc_id else []

    kyc_risk = kyc.get("risk_profile")
    
    if reviews:
        reviews.sort(key=lambda x: x["date"], reverse=True)
        latest_review = reviews[0]
        review_risk = latest_review.get("risk_profile")
        
        if kyc_risk and review_risk and kyc_risk != review_risk:
            conflict = True
            conflict_details = f"KYC records show risk profile as '{kyc_risk}', but suitability review {latest_review['id']} dated {latest_review['date']} records risk profile as '{review_risk}'."
            citations.append(latest_review["id"])
            register_flag("conflict")

    register_citations(citations)

    return {
        "client_id": client_id,
        "name": client["name"],
        "kyc_status": kyc.get("kyc_status"),
        "pan": masked_pan,
        "date_of_birth": kyc.get("date_of_birth"),
        "address": kyc.get("address"),
        "annual_income_band": kyc.get("annual_income_band"),
        "risk_profile": kyc_risk,
        "employment": kyc.get("employment"),
        "bank_account": masked_bank_info,
        "conflict": conflict,
        "conflict_details": conflict_details,
        "citations": citations
    }

def get_client_accounts(client_id: str) -> Dict[str, Any]:
    """Retrieve client accounts information."""
    client_id = get_resolved_client_id(client_id)
    
    book = load_json_file(BOOK_PATH)
    client = next((c for c in book["clients"] if c["id"] == client_id), None)
    if not client:
        raise ClientNotFoundException(client_id)
        
    citations = [client_id]
    register_citations(citations)
    
    return {
        "client_id": client_id,
        "accounts": client.get("accounts", []),
        "citations": citations
    }

def get_client_cash_balance(client_id: str) -> Dict[str, Any]:
    """
    Deterministically calculates client cash balance.
    Respects run_context.as_at filter.
    """
    client_id = get_resolved_client_id(client_id)
    
    book = load_json_file(BOOK_PATH)
    client = next((c for c in book["clients"] if c["id"] == client_id), None)
    if not client:
        raise ClientNotFoundException(client_id)

    as_at = run_context.as_at or book["meta"]["as_of"]

    cash = Decimal("0.00")
    relevant_tx_ids = []
    
    for t in client.get("transactions", []):
        if t["date"] > as_at:
            continue
            
        relevant_tx_ids.append(t["id"])
        ttype = t["type"]
        
        if ttype == "deposit":
            cash += Decimal(t["amount_usd"])
        elif ttype in ("withdrawal", "fee"):
            cash -= Decimal(t["amount_usd"])
        elif ttype == "buy":
            cash -= Decimal(t["net_usd"])
        elif ttype == "sell":
            cash += Decimal(t["net_usd"])
        elif ttype == "dividend":
            cash += Decimal(t["net_usd"])

    register_citations(relevant_tx_ids)

    return {
        "client_id": client_id,
        "as_at": as_at,
        "cash_balance": quantize_decimal(cash),
        "citations": relevant_tx_ids,
        "transactions_count": len(relevant_tx_ids)
    }

def get_client_transactions(client_id: str, limit: int = 10, type_filter: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve filtered and sorted transaction list."""
    client_id = get_resolved_client_id(client_id)
    
    book = load_json_file(BOOK_PATH)
    client = next((c for c in book["clients"] if c["id"] == client_id), None)
    if not client:
        raise ClientNotFoundException(client_id)

    as_at = run_context.as_at or book["meta"]["as_of"]

    txs = []
    for t in client.get("transactions", []):
        if t["date"] > as_at:
            continue
        if type_filter and t["type"] != type_filter:
            continue
        txs.append(t)

    # Sort transactions descending by date and then by id descending
    txs.sort(key=lambda x: (x["date"], x["id"]), reverse=True)
    
    result_txs = txs[:limit]
    tx_ids = [t["id"] for t in result_txs]
    
    register_citations(tx_ids)

    return {
        "client_id": client_id,
        "as_at": as_at,
        "transactions": result_txs,
        "total_count": len(txs),
        "citations": tx_ids
    }

def get_client_notes(client_id: str) -> Dict[str, Any]:
    """Retrieve client notes and memos up to run_context.as_at."""
    client_id = get_resolved_client_id(client_id)
    
    book = load_json_file(BOOK_PATH)
    client = next((c for c in book["clients"] if c["id"] == client_id), None)
    if not client:
        raise ClientNotFoundException(client_id)

    as_at = run_context.as_at or book["meta"]["as_of"]

    notes = []
    citations = []
    for n in client.get("notes", []):
        if n["date"] > as_at:
            continue
        notes.append(n)
        citations.append(n["id"])

    register_citations(citations)

    return {
        "client_id": client_id,
        "notes": notes,
        "citations": citations
    }

def get_client_holdings_and_drift(client_id: str) -> Dict[str, Any]:
    """
    Reconstructs client holdings, values them, checks coverage,
    and calculates drift against target allocations.
    """
    client_id = get_resolved_client_id(client_id)
    
    book = load_json_file(BOOK_PATH)
    market = load_json_file(MARKET_PATH)
    
    client = next((c for c in book["clients"] if c["id"] == client_id), None)
    if not client:
        raise ClientNotFoundException(client_id)

    as_at = run_context.as_at or book["meta"]["as_of"]

    # 1. Reconstruct holding quantities from transactions
    holdings = {}
    relevant_tx_ids = []
    
    for t in client.get("transactions", []):
        if t["date"] > as_at:
            continue
        relevant_tx_ids.append(t["id"])
        ttype = t["type"]
        if ttype == "buy":
            symbol = t["symbol"]
            holdings[symbol] = holdings.get(symbol, Decimal("0.00")) + Decimal(t["quantity"])
        elif ttype == "sell":
            symbol = t["symbol"]
            holdings[symbol] = holdings.get(symbol, Decimal("0.00")) - Decimal(t["quantity"])

    # Filter out near-zero/zero positions
    holdings = {sym: q for sym, q in holdings.items() if q > Decimal("0.0001")}

    # 2. Value holdings & check coverage
    covered_symbols = set(market["meta"]["covered_symbols"])
    uncovered_holdings = [sym for sym in holdings.keys() if sym not in covered_symbols]
    
    if uncovered_holdings:
        raise UncoveredSymbolException(uncovered_holdings[0])

    # Calculate Cash Balance
    cash_res = get_client_cash_balance(client_id)
    cash = Decimal(cash_res["cash_balance"])

    stock_values = {}
    for sym, q in holdings.items():
        prices = market["prices"].get(sym, [])
        valid_prices = [p for p in prices if p["date"] <= as_at]
        if not valid_prices:
            price = Decimal("0.00")
        else:
            valid_prices.sort(key=lambda x: x["date"], reverse=True)
            price = Decimal(valid_prices[0]["close"])
        stock_values[sym] = q * price

    total_stocks = sum(stock_values.values(), Decimal("0.00"))
    total_portfolio = cash + total_stocks

    # 3. Target Allocations & Drift
    reviews = client.get("suitability_reviews", [])
    valid_reviews = [r for r in reviews if r["date"] <= as_at]
    target_alloc = {}
    review_id = None
    
    if valid_reviews:
        valid_reviews.sort(key=lambda x: x["date"], reverse=True)
        latest_review = valid_reviews[0]
        review_id = latest_review["id"]
        target_alloc = {sym: Decimal(pct) for sym, pct in latest_review["target_allocation_pct"].items()}

    # Calculate Actual Allocations
    actual_alloc = {}
    drift = {}
    if total_portfolio > Decimal("0.00"):
        actual_alloc = {sym: (val / total_portfolio) * Decimal("100.00") for sym, val in stock_values.items()}
        actual_alloc["CASH"] = (cash / total_portfolio) * Decimal("100.00")
    else:
        actual_alloc["CASH"] = Decimal("0.00")

    # Compute drift = actual% - target%
    all_symbols = set(target_alloc.keys()) | {sym for sym in stock_values.keys()}
    for sym in all_symbols:
        target = target_alloc.get(sym, Decimal("0.00"))
        actual = actual_alloc.get(sym, Decimal("0.00"))
        drift[sym] = actual - target

    # Formatted output versions
    holdings_formatted = {sym: str(q) for sym, q in holdings.items()}
    stock_values_formatted = {sym: quantize_decimal(v) for sym, v in stock_values.items()}
    target_alloc_formatted = {sym: quantize_decimal(v) for sym, v in target_alloc.items()}
    actual_alloc_formatted = {sym: quantize_decimal(v) for sym, v in actual_alloc.items()}
    drift_formatted = {sym: quantize_decimal(v) for sym, v in drift.items()}

    # Citations
    citation_pool = []
    if review_id:
        citation_pool.append(review_id)
    citation_pool.extend(relevant_tx_ids)
    
    register_citations(citation_pool)

    return {
        "client_id": client_id,
        "as_at": as_at,
        "cash_balance": quantize_decimal(cash),
        "holdings": holdings_formatted,
        "stock_values": stock_values_formatted,
        "total_stocks": quantize_decimal(total_stocks),
        "total_portfolio": quantize_decimal(total_portfolio),
        "target_allocation": target_alloc_formatted,
        "actual_allocation": actual_alloc_formatted,
        "drift": drift_formatted,
        "citations": citation_pool
    }

def get_market_instrument(symbol: str) -> Dict[str, Any]:
    """Retrieve instrument details from market data."""
    market = load_json_file(MARKET_PATH)
    covered_symbols = set(market["meta"]["covered_symbols"])
    if symbol not in covered_symbols:
        raise UncoveredSymbolException(symbol)
        
    inst = next((i for i in market["instruments"] if i["symbol"] == symbol), None)
    citations = [symbol]
    register_citations(citations)
    
    return {
        "symbol": symbol,
        "instrument": inst,
        "citations": citations
    }

def get_market_price(symbol: str) -> Dict[str, Any]:
    """Retrieve market price for a symbol on or before run_context.as_at."""
    market = load_json_file(MARKET_PATH)
    covered_symbols = set(market["meta"]["covered_symbols"])
    if symbol not in covered_symbols:
        raise UncoveredSymbolException(symbol)
        
    as_at = run_context.as_at or market["meta"]["as_of"]

    prices = market["prices"].get(symbol, [])
    valid_prices = [p for p in prices if p["date"] <= as_at]
    
    if not valid_prices:
        raise ValueError(f"No price record found for {symbol} on or before {as_at}")

    valid_prices.sort(key=lambda x: x["date"], reverse=True)
    latest_price = valid_prices[0]
    
    citations = [symbol]
    register_citations(citations)
    
    return {
        "symbol": symbol,
        "date": latest_price["date"],
        "price": latest_price["close"],
        "citations": citations
    }

def get_market_news(symbol: str) -> Dict[str, Any]:
    """Retrieve market news for a symbol up to run_context.as_at."""
    market = load_json_file(MARKET_PATH)
    covered_symbols = set(market["meta"]["covered_symbols"])
    if symbol not in covered_symbols:
        raise UncoveredSymbolException(symbol)
        
    as_at = run_context.as_at or market["meta"]["as_of"]

    news_items = []
    citations = []
    for n in market.get("news", []):
        if n["symbol"] == symbol and n["date"] <= as_at:
            news_items.append(n)
            citations.append(n["id"])

    register_citations(citations)

    return {
        "symbol": symbol,
        "news": news_items,
        "citations": citations
    }
