from typing import List, Optional
from pydantic import BaseModel, Field

class AnswerResponse(BaseModel):
    question_id: str = Field(..., description="The unique ID of the question.")
    answer: str = Field(..., description="Natural language explanation of the answer. May be empty when abstaining or refusing.")
    answer_value: Optional[str] = Field(None, description="The single figure, count, or date the question asks for. Money in USD (no symbol, no separator). Dates in ISO. Must be null if abstained or refused.")
    abstained: bool = Field(..., description="true when the data cannot support an answer (epistemic limit).")
    refused: bool = Field(..., description="true when policy forbids answering (policy limit).")
    reason: Optional[str] = Field(None, description="Required non-empty string whenever abstained or refused is true. Otherwise null.")
    citations: List[str] = Field(default_factory=list, description="The record ids the answer relies on. Citing records from other clients is a violation.")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0.")
    flags: List[str] = Field(default_factory=list, description="Zero or more flags of: 'conflict', 'upstream_issue', 'stale_data'.")
    agents: List[str] = Field(..., description="The role path that produced this answer in order (must include 'router' first).")

class AgentRosterEntry(BaseModel):
    role: str = Field(..., description="Agent role from taxonomy: router, book_qa, kyc_profile, notes_desk, market_desk, compliance.")
    name: str = Field(..., description="Name of the agent.")
    model: str = Field(..., description="Model used by the agent (valura-fast or valura-deep).")

class RosterResponse(BaseModel):
    framework: str = Field("agno", description="Framework used, must be 'agno'.")
    framework_version: str = Field("1.5.0", description="Version of the framework.")
    agents: List[AgentRosterEntry] = Field(..., description="Declared agents in the ecosystem.")
