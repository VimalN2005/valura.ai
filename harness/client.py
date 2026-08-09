import os
import time
import httpx
from typing import Dict, Any, Optional

class ArenaClient:
    """
    Client wrapper for interacting with the Valura AI Arena API.
    Handles rate-limiting (429) retries, authentication, and HTTP headers.
    """
    def __init__(self, base_url: str, api_key: str, mode: str = "practice"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.mode = mode
        
        # Configure httpx client with standard headers
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=60.0,
            verify=False  # Disable SSL verification to prevent issues with custom sandbox certs
        )

    def request(self, method: str, path: str, json_data: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        params = {"mode": self.mode}
        
        retries = 6
        backoff = 1.0
        
        for attempt in range(retries):
            try:
                response = self.client.request(method, url, params=params, json=json_data)
                
                # Handle Transient Rate Limiting (429)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        sleep_time = float(retry_after) if retry_after else backoff
                    except ValueError:
                        sleep_time = backoff
                    print(f"[HTTP 429] Rate limited. Sleeping for {sleep_time}s before retry...")
                    time.sleep(sleep_time + 0.2)  # sleep with a tiny safety buffer
                    backoff *= 1.5
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    continue
                print(f"[HTTP Error] {method} {path} returned status {e.response.status_code}: {e.response.text}")
                raise e
            except httpx.RequestError as e:
                print(f"[Request Error] Attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    raise e
                time.sleep(backoff)
                backoff *= 2.0
                
        raise Exception(f"Max retries exceeded for {method} {path}")

    def get_rules(self) -> Dict[str, Any]:
        return self.request("GET", "v1/rules")

    def get_book(self) -> Dict[str, Any]:
        return self.request("GET", "v1/book")

    def get_market(self) -> Dict[str, Any]:
        return self.request("GET", "v1/market")

    def post_roster(self, roster: Dict[str, Any]) -> Dict[str, Any]:
        return self.request("POST", "v1/roster", json_data=roster)

    def get_next_question(self) -> Dict[str, Any]:
        return self.request("GET", "v1/next")

    def post_answer(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        return self.request("POST", "v1/answer", json_data=answer)

    def get_me(self) -> Dict[str, Any]:
        return self.request("GET", "v1/me")
