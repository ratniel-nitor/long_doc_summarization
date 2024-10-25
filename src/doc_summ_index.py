import json
from typing import List
from pathlib import Path
from loguru import logger
from llama_index.core.node_parser import SentenceSplitter
from src.data_processing.data_loader import read_pdf_file
from src.model_handlers.model_loaders import ModelInitializer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core import (
    DocumentSummaryIndex,
    get_response_synthesizer,
    Document,
    Settings,
)
from datetime import datetime

logger.info("Imports completed")

logger.add("doc_summ_index.log", rotation="10 MB")

# Define constants
DATA_DIR = Path("D:\\NIPL2093\\work\\long_doc_summarization\\data\\")
TEST_FILE = "37657_2017_1_1501_39247_Judgement_19-Oct-2022_149pg.pdf"
OUTPUT_DIR = DATA_DIR / "outputs"

# models
MODEL_LLM = "gemini-1.5-flash-002"
MODEL_EMBEDDING = "BAAI/bge-small-en-v1.5"
# MODEL_EMBEDDING = "jxm/cde-small-v1"

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

IMPORTANT GUIDELINES:
1. ONLY use information explicitly present in the provided text
2. Do NOT make assumptions or inferences
3. If specific information is not found, state "Information not found in the document"
4. Use direct quotes where possible
5. Include specific case numbers, dates, and names exactly as they appear 
"""


def generate_subsection_summary(
    nodes, section_name: str, response_synthesizer: BaseSynthesizer
) -> str:
    """Generate a summary from a set of nodes for a specific section."""
    logger.info(f"Generating summary for section: {section_name}")
    try:
        texts = [
            str(node.node.text) if hasattr(node, "node") else str(node.text)
            for node in nodes
        ]
        logger.debug(f"Processing {len(texts)} text segments for {section_name}")
        for i, text in enumerate(texts):
            logger.debug(f"Node {i} text: {text}")

        # combined_text = " ".join(texts)
        summary_query = f"Based on the following text, provide a concise but detailed summary of {section_name}"

        logger.debug(f"Generating summary with query length: {len(summary_query)}")

        # List[str] is sent as text for synthesis
        summary = response_synthesizer.synthesize(summary_query, nodes)

        logger.success(f"Successfully generated summary for {section_name}")
        return str(summary)
    except Exception as e:
        logger.exception(f"Error generating summary for {section_name}: {str(e)}")
        return f"Error generating {section_name} summary"


def main():
    test_file_path = DATA_DIR / TEST_FILE
    logger.info(f"Starting document processing pipeline for: {test_file_path}")

    try:
        documents: List[Document] = read_pdf_file(str(test_file_path))
        logger.info(f"Successfully loaded PDF with {len(documents)} document(s)")

        if documents:
            logger.info("Initializing models and configurations")
            Settings.llm = ModelInitializer.initialize_gemini(
                model_name=MODEL_LLM, use_llamaindex=True
            )
            # Settings.llm = ModelInitializer.initialize_groq()
            logger.debug(f"Initialized LLM: {MODEL_LLM}")

            Settings.embed_model = HuggingFaceEmbedding(model_name=MODEL_EMBEDDING)
            logger.debug(f"Initialized embedding model: {MODEL_EMBEDDING}")

            splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

            response_synthesizer: BaseSynthesizer = get_response_synthesizer(
                response_mode="tree_summarize",
                # response_mode="refine",
                use_async=True,
                summary_template=PromptTemplate(
                    final_summary_query, prompt_type=PromptType.SUMMARY
                ),
                # streaming=True,
            )

            doc_summary_index = DocumentSummaryIndex.from_documents(
                documents,
                transformations=[splitter],
                response_synthesizer=response_synthesizer,
                show_progress=True,
                embed_model=Settings.embed_model,
            )

            doc_retriever = doc_summary_index.as_retriever()

            summaries = {
                "case_basics": generate_subsection_summary(
                    doc_retriever.retrieve(
                        "Describe the case type, parties involved (complainant, accused, etc.), and the main legal issues. Start by listing the parties clearly. If any information is not available, state 'information not available'."
                    ),
                    "case basics (including case type, parties involved, and main legal issues)",
                    response_synthesizer,
                ),
                "factual_background": generate_subsection_summary(
                    doc_retriever.retrieve(
                        "Summarize the factual background of the dispute. If any information is not available, state 'information not available'."
                    ),
                    "factual background of the case",
                    response_synthesizer,
                ),
                "legal_arguments": generate_subsection_summary(
                    doc_retriever.retrieve(
                        "Summarize each side's key arguments, noting undisputed points. Include specific legal provisions (with article/section numbers) used. If any information is not available, state 'information not available'."
                    ),
                    "legal arguments from all sides",
                    response_synthesizer,
                ),
                "court_analysis": generate_subsection_summary(
                    doc_retriever.retrieve(
                        "Explain the court's reasoning on each point, highlighting evidence interpretation and legal principles applied. If any information is not available, state 'information not available'."
                    ),
                    "court's analysis and reasoning",
                    response_synthesizer,
                ),
                "decision_outcome": generate_subsection_summary(
                    doc_retriever.retrieve(
                        "Specify the court's final order and directives. In criminal cases, detail the offense, sentence, fines, or compensation. Include details about acquittals and release orders, costs and compensation, compounding of offenses, and compensation to the accused for frivolous complaints. If any information is not available, state 'information not available'."
                    ),
                    "court's final decision and directives",
                    response_synthesizer,
                ),
                "other_relevant": generate_subsection_summary(
                    doc_retriever.retrieve(
                        "Include details about acquittals and release orders, costs and compensation, compounding of offenses, and compensation to the accused for frivolous complaints. If any information is not available, state 'information not available'."
                    ),
                    "other relevant information",
                    response_synthesizer,
                ),
            }

            # Generate the final summary at once for the whole pdf
            query_engine = doc_summary_index.as_query_engine()
            summary = query_engine.query(final_summary_query)
            logger.info("Document Summary sample:")
            logger.info(summary)

            # Save the structured summary
            timestamp = datetime.now().isoformat()
            output_file = OUTPUT_DIR / f"{TEST_FILE.split('.')[0]}_summary.json"
            logger.info(f"Saving summary to: {output_file}")
            try:
                with open(output_file, "a", encoding="utf-8") as f:
                    json.dump(
                        {
                            "timestamp": timestamp,
                            "file_name": TEST_FILE,
                            "model": {"llm": MODEL_LLM, "embedding": MODEL_EMBEDDING},
                            "structured_summary": summaries,
                            "prompt": final_summary_query,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                    f.write("\n")
                logger.success(f"Successfully saved summary to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save summary to file: {str(e)}")

            # Persist the index
            index_storage_path = OUTPUT_DIR / f"{TEST_FILE.split('.')[0]}_index"
            logger.info(f"Persisting index to: {index_storage_path}")
            try:
                doc_summary_index.storage_context.persist(str(index_storage_path))
                logger.success("Successfully persisted index")
            except Exception as e:
                logger.error(f"Failed to persist index: {str(e)}")

    except Exception as e:
        logger.exception(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
