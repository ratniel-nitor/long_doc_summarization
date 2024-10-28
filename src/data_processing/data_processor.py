from llama_index.core import Document
from src.data_processing.data_loader import read_pdf_file
from src.model_handlers.model_loaders import ModelInitializer
from pydantic import BaseModel, Field, ValidationError
from typing import List, Any
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json

MODEL_NAME = "gemini-1.5-flash-002"

system_prompt = """
You are an AI assistant specialized in analyzing legal documents. You can identify entities, their properties and dependencies in the document. The information extracted will be used to generate a comprehensive summary of the document and must therefore be as accurate and information rich as possible.
"""

# Configure logging
logger.add("data_processor.log", rotation="5 MB", level="DEBUG")

class Property(BaseModel):
    key: str
    value: str
    resolved_absolute_value: str

class Entity(BaseModel):
    id: int
    subquote_string: List[str] = Field(
        ..., description="Correctly resolved value of the entity"
    )
    entity_title: str
    properties: List[Property] = Field(
        ..., description="List of properties of the entity"
    )
    dependencies: List[int] = Field(
        ...,
        description="List of entity IDs that depend on this entity or relies on it to resolve it",
    )

class DocumentExtraction(BaseModel):
    entities: List[Entity] = Field(..., description="List of entities in the document")


# class LegalEntityType(Enum):
#     CITATION = "citation"
#     ARGUMENT = "argument"
#     FACT = "fact"
#     LEGAL_PRECEDENT = "legal_precedent"
#     STATUTE = "statute"
#     JUDGMENT = "judgment"
#     PARTY = "party"
#     COURT = "court"
#     DATE = "date"
#     RULING_PROVISION = "ruling_provision"
#     PENALTY = "penalty"
#     OTHER = "other"


# class LegalEntity(BaseModel):
#     type: LegalEntityType = Field(description="The type of legal entity being extracted.")
#     text: str = Field(description="The actual text content of the entity.")
#     metadata: Optional[Dict] = Field(
#         default=None,
#         description="""Optional metadata for the entity.  Examples:
#         - CITATION: {"source": "AIR 1970 SC 123"}
#         - ARGUMENT: {"side": "Plaintiff", "supporting_facts": [1, 2]}  (where 1 and 2 are chunk_ids of supporting FACTs)
#         - PARTY: {"role": "Defendant", "representation": "John Smith"}
#         - RULING_PROVISION: {"statute": "Indian Penal Code 377", "section": "2(v)"}
#         - PENALTY: {"type": "Imprisonment", "duration": "10 years"} 
#         """,
#     )


# class DocumentChunkExtraction(BaseModel):
#     chunk_id: int = Field(description="Unique identifier for the text chunk.")
#     entities: List[LegalEntity] = Field(
#         default=[], description="List of legal entities extracted from this chunk."
#     )


# class DocumentExtraction(BaseModel):
#     chunks: List[DocumentChunkExtraction] = Field(
#         description="Extracted information from all document chunks."
#     )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ValueError, TypeError, AttributeError, Exception)),
    before_sleep=lambda retry_state: logger.info(f"Retrying due to {retry_state.outcome.exception()}")
)
def get_extracted_doc(client: Any, doc_text: str, previous_extractions: List[Entity]) -> DocumentExtraction:
    if previous_extractions:
        context = f"Previously extracted entities: {previous_extractions}\n\n" 
    else:
        context = ""

    prompt = f"{context}\nExtract entities from the following text: {doc_text}"
    
    logger.debug(f"Sending request to LLM with prompt: {prompt[:500]}...")  # Log first 500 chars of prompt
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_model=DocumentExtraction,
        )
        logger.debug(f"Received response from LLM: {[e.entity_title for e in response.entities]}")
        return response
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Invalid response format: {e}. Retrying...")
        raise

def process_document(documents: List[Document]) -> List[DocumentExtraction]:
    logger.info(f"Starting to process {len(documents)} documents")
    generated_data = []
    client = ModelInitializer.initialize_gemini(model_name=MODEL_NAME, use_instructor=True)

    for i, doc in enumerate(documents):
        logger.info(f"Processing document {i+1}/{len(documents)}")
        try:
            extracted_doc = get_extracted_doc(client, doc.text, generated_data)
            if isinstance(extracted_doc, DocumentExtraction):
                generated_data.append(extracted_doc)
            else:
                logger.warning(f"Unexpected response format for document {i+1}. Attempting to parse...")
                # Attempt to parse the response if it's not in the expected format
                parsed_doc = DocumentExtraction(entities=[])
                if isinstance(extracted_doc, dict) and 'entities' in extracted_doc:
                    for entity_data in extracted_doc['entities']:
                        try:
                            entity = Entity(**entity_data)
                            parsed_doc.entities.append(entity)
                        except Exception as e:
                            logger.error(f"Error parsing entity: {str(e)}")
                generated_data.append(parsed_doc)
        except Exception as e:
            logger.error(f"Error processing document {i+1}: {str(e)}")

    logger.info(f"Finished processing {len(documents)} documents")
    return generated_data

if __name__ == "__main__":
    from pathlib import Path

    DATA_DIR = Path("D:\\NIPL2093\\work\\long_doc_summarization\\data\\")
    TEST_FILE = DATA_DIR / "37657_2017_1_1501_39247_Judgement_19-Oct-2022_149pg.pdf"

    logger.add("output.log", rotation="10 MB")

    documents = read_pdf_file(TEST_FILE)
    processed_data = process_document(documents[:3])  # Process first 3 pages
    logger.info(f"Generated output: {json.dumps([doc.dict() for doc in processed_data], indent=2)}")
