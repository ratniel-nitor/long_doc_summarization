from typing import List, Optional, Dict
from pydantic import BaseModel, Field

system_prompt = """
# Summary of [Document Title]

## Introduction
- **Purpose**: Briefly describe the purpose of the document.
- **Parties Involved**: List the parties involved in the agreement.

## Key Terms and Definitions
- **Term 1**: Definition and explanation.
- **Term 2**: Definition and explanation.

## Main Provisions
- **Obligations**: Outline the main obligations of each party.
- **Rights**: Describe the rights granted to each party.

## Financial Terms
- **Payment Terms**: Summarize the payment structure and any financial obligations.

## Duration and Termination
- **Duration**: State the length of the agreement.
- **Termination Conditions**: Explain how and when the agreement can be terminated.

## Additional Clauses
- **Confidentiality**: Briefly describe any confidentiality obligations.
- **Dispute Resolution**: Outline the process for resolving disputes.

## Conclusion
- **Summary**: Provide a brief conclusion summarizing the overall intent and key points of the document.
"""

class Party(BaseModel):
    name: str
    role: str # E.g., "Complainant", "Accused", "Witness"  
    representation: Optional[str] # Lawyer's name

class Citation(BaseModel):
    """Represents a citation within the document."""
    text: str
    source: Optional[str]  # E.g., "AIR 1970 SC 123"


class Fact(BaseModel):
    """Represents a factual statement presented in the document."""
    text: str
    source: Optional[str] # E.g., "Witness Testimony," "Exhibit A"
    verified: Optional[bool]
    related_to_points: Optional[List[int]] # Relate to Points for Determination

class Argument(BaseModel):
    """Represents a legal argument or point made in the document."""
    side: str  # E.g., "Plaintiff", "Defendant", "Court"
    text: str
    supporting_evidence: Optional[List[str]]  # IDs of Facts or Citations
    related_arguments: Optional[List[str]] # IDs of other related Arguments


class LegalPrecedent(BaseModel):
    case_name: str  
    citation: Optional[Citation]
    relevance: str # Brief description of why this precedent is relevant


class Statute(BaseModel):
    name: str
    section: Optional[str]
    citation: Optional[Citation]
    relevance: str


class Judgment(BaseModel):  # Enhanced
    text: str
    reasoning: Optional[str]
    sentences: List[str] # Add sentences imposed

class PointForDetermination(BaseModel): # From Rule 1(i)
    text: str # Text of point to determine
    decision: Optional[str]
    reasoning: Optional[str] # Reasoning behind decision on this specific point

class DocumentChunkExtraction(BaseModel): # Enhanced
    chunk_id: int 
    parties: List[Party] # Add parties involved in this chunk
    points_for_determination: List[PointForDetermination] # Add points from Rule 1(i)
    arguments: List[Argument] = []
    facts: List[Fact] = []
    citations: List[Citation] = []
    legal_precedents: List[LegalPrecedent] = []
    statutes: List[Statute] = []
    judgement: Optional[Judgment] 

class DocumentExtraction(BaseModel): # (No change)
    chunks: List[DocumentChunkExtraction]