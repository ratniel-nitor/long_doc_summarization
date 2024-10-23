import json
from typing import List
from pathlib import Path
from loguru import logger
from llama_index.core.node_parser import SentenceSplitter
from src.data_processing.data_loader import read_pdf_file
from src.model_handlers.model_loaders import ModelInitializer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    DocumentSummaryIndex,
    get_response_synthesizer,
    Document,
    Settings,
)
from datetime import datetime  # Add this import at the top
logger.info("Imports completed")

# Configure logging
logger.add("doc_summ_index.log", rotation="10 MB")

# Define constants
DATA_DIR = Path("D:\\NIPL2093\\work\\long_doc_summarization\\data\\")
TEST_FILE = "37657_2017_1_1501_39247_Judgement_19-Oct-2022_149pg.pdf"
OUTPUT_DIR = DATA_DIR / "outputs"

# models 
MODEL_LLM = "gemini-1.5-flash-002"
MODEL_EMBEDDING = "BAAI/bge-small-en-v1.5"

final_summary_query = """
You're an AI bot summarizing Indian court judgments and orders for legal professionals. Include all key legal details. Summarize the entire document, covering these points:

1. Case Basics: Briefly describe the case type, parties involved (complainant, accused, etc.), and the main legal issues.  Start by listing the parties clearly.
2. Factual Background: Concisely present the relevant facts of the dispute.
3. Legal Arguments: Summarize each side's key arguments, noting undisputed points. Include specific legal provisions (with article/section numbers) used.
4. Court's Analysis: Explain the court's reasoning on each point, highlighting evidence interpretation and legal principles applied. State the court's decision on each point.
5. Decision and Outcome: Specify the court's final order and directives. In criminal cases, detail the offense, sentence, fines, or compensation.
6. Other Relevant Information: Include details about acquittals and release orders, costs and compensation, compounding of offenses, and compensation to the accused for frivolous complaints.

NOTE:
- Refer to individuals (accused, witnesses) by their names and assigned numbers, if available.
- When referencing individuals (accused, witnesses), use their designated numbers/nomenclature followed by their names (if needed, with the number in brackets).
- Refer to exhibits by their exhibit number and brief description.
- If the document is not a court judgment or order, or does not contain any legal information, return "The provided text is not a court judgment or order."

DO NOT entail what is not mentioned in the document. 
"""

def main():
    test_file_path = DATA_DIR / TEST_FILE

    logger.info(f"Starting to process file: {test_file_path}")
    documents: List[Document] = read_pdf_file(str(test_file_path))

    if documents:
        logger.success(f"Successfully read the PDF file: {test_file_path}")
        
        llm = ModelInitializer.initialize_gemini(model_name=MODEL_LLM, use_llamaindex=True)
        Settings.llm = llm
        
        embed_model = HuggingFaceEmbedding(model_name=MODEL_EMBEDDING)
        Settings.embed_model = embed_model
        
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        
        # # Log the chunks created from the input PDF file
        # import random
        # random_index = random.randint(0, len(documents) - 1)
        # logger.info(f"Randomly selected document index: {random_index}")
        # chunks = splitter.split_text(documents[random_index])
        # for i, chunk in enumerate(chunks):
        #     logger.info(f"Chunk {i + 1}:")
        #     # logger.info(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)
        #     logger.info(chunk)
        #     logger.info("-" * 50)
        
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=True,
            # streaming=True,
        )
        
        doc_summary_index = DocumentSummaryIndex.from_documents(
            documents,
            transformations=[splitter],
            response_synthesizer=response_synthesizer,
            show_progress=True,
            embed_model=Settings.embed_model,
        )
        
        # you can pass one docuemnt (embedded) at once, to get document summary here
        # doesn't provide summary for the whole document at once 
        # summary = doc_summary_index.get_document_summary(TEST_FILE)
        
        # logger.info("Document Summary sample:")
        # logger.info(summary)

        query_engine = doc_summary_index.as_query_engine()
        summary = query_engine.query(final_summary_query)
        logger.info("Document Summary sample:")
        logger.info(summary)

        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f"{TEST_FILE.split('.')[0]}_summary.json"
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "file_name": TEST_FILE,
                "model": {
                    "llm": MODEL_LLM,
                    "embedding": MODEL_EMBEDDING
                },
                "prompt": final_summary_query,
                "summary": str(summary)
            }, f, indent=2, ensure_ascii=False)
            f.write('\n')
        
        # Persist the index
        index_storage_path = OUTPUT_DIR / f"{TEST_FILE.split('.')[0]}_index"
        doc_summary_index.storage_context.persist(str(index_storage_path))
        
        logger.info(f"Summary saved to: {output_file}")
        logger.info(f"Index persisted to: {index_storage_path}")
    else:
        logger.error(f"Failed to read the PDF file: {test_file_path}")

if __name__ == "__main__":
    main()
