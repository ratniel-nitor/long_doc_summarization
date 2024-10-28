from loguru import logger
from pathlib import Path
from src.data_processing.data_loader import read_pdf_file
from src.data_processing.data_processor import process_document
import json

# Configure logging
logger.add("app.log", rotation="10 MB")

# Define constants
DATA_DIR = Path("D:\\NIPL2093\\work\\long_doc_summarization\\data\\")
TEST_FILE = "37657_2017_1_1501_39247_Judgement_19-Oct-2022_149pg.pdf"
OUTPUT_DIR = DATA_DIR / "outputs"

def main():
    # Construct the full path to the test file
    test_file_path = DATA_DIR / TEST_FILE

    # Log the start of the process
    logger.info(f"Starting to process file: {test_file_path}")

    # Read the PDF file
    document_content = read_pdf_file(str(test_file_path))

    if document_content:
        logger.success(f"Successfully read the PDF file: {test_file_path}")
        
        # Process the document content
        # processed_data = process_document(document_content)
        processed_data = process_document(document_content[:50])

        # Log the processed data
        logger.info("Processed document data:")
        for idx, extraction in enumerate(processed_data, 1):
            logger.info(f"Extraction {idx}:")
            for entity in extraction.entities:
                logger.info(f"  Entity: {entity.entity_title}")
                logger.info(f"    ID: {entity.id}")
                logger.info(f"    Subquote: {entity.subquote_string}")
                logger.info("    Properties:")
                for prop in entity.properties:
                    logger.info(f"      {prop.key}: {prop.value}")
                logger.info(f"    Dependencies: {entity.dependencies}")
                logger.info("---")
        
        # Create the outputs directory if it doesn't exist
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save the processed data as a JSON file
        output_file = OUTPUT_DIR / f"{TEST_FILE.stem}_processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([extraction.dict() for extraction in processed_data], f, indent=2, ensure_ascii=False)
    else:
        logger.error(f"Failed to read the PDF file: {test_file_path}")

if __name__ == "__main__":
    main()
