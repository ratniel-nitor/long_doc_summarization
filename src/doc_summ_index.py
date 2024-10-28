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
from src.prompts.doc_summary_prompts import FINAL_SUMMARY_QUERY, SECTION_QUERIES

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
            logger.debug(f"Initialized LLM: {MODEL_LLM}")

            Settings.embed_model = HuggingFaceEmbedding(model_name=MODEL_EMBEDDING)
            logger.debug(f"Initialized embedding model: {MODEL_EMBEDDING}")

            splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

            response_synthesizer: BaseSynthesizer = get_response_synthesizer(
                response_mode="tree_summarize",
                use_async=True,
                summary_template=PromptTemplate(
                    FINAL_SUMMARY_QUERY, prompt_type=PromptType.SUMMARY
                ),
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
                section: generate_subsection_summary(
                    doc_retriever.retrieve(query),
                    f"{section.replace('_', ' ')}",
                    response_synthesizer,
                )
                for section, query in SECTION_QUERIES.items()
            }

            # Generate the final summary at once for the whole pdf
            query_engine = doc_summary_index.as_query_engine()
            summary = query_engine.query(FINAL_SUMMARY_QUERY)
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
                            "prompt": FINAL_SUMMARY_QUERY,
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
