import os
from agno.models.openai import OpenAIChat

def get_base_url_and_key() -> tuple[str, str]:
    """Retrieve the base URL and API key from environment variables."""
    # The server runs either with standard VALURA_API_KEY or the Docker-specified ASSESSMENT_KEY
    api_key = os.getenv("ASSESSMENT_KEY") or os.getenv("VALURA_API_KEY")
    if not api_key:
        # Fallback for practice mode if not set in environment (since user provided it, we can fallback to it)
        api_key = "vlr_9MgCCpiMEKBico7RT_N2aqRP-Du-cYOO"

    # The target gateway URL
    base_url = os.getenv("ASSESSMENT_URL") or "https://ai-arena.twocc.in"
    
    # Clean trailing slash from base_url
    if base_url.endswith("/"):
        base_url = base_url[:-1]
        
    return base_url, api_key

def get_model(model_name: str = "valura-fast") -> OpenAIChat:
    """
    Returns an Agno OpenAIChat client pointing to the Valura proxy.
    Supported model names: 'valura-fast' or 'valura-deep'.
    """
    base_url, api_key = get_base_url_and_key()
    
    # OpenAI client expects base_url to end with /v1
    openai_base_url = f"{base_url}/llm/v1"
    
    return OpenAIChat(
        id=model_name,
        api_key=api_key,
        base_url=openai_base_url
    )
