import os
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, Document
from loguru import logger
from typing import List
from tqdm import tqdm
import re

# constants
DATA_DIR = Path("D:\\NIPL2093\\work\\long_doc_summarization\\data\\")
TEST_FILE = Path("37657_2017_1_1501_39247_Judgement_19-Oct-2022_149pg.pdf")

# Load env variables
load_dotenv()

# Configure logging
logger.add("app.log", rotation="5 MB")


def preprocess_citations(text: str) -> str:
    # Split the text into lines
    lines = text.split('\n')
    processed_lines = []
    citations = []
    
    for line in lines:
        # Remove standalone page numbers
        if re.match(r'^\s*\d+\s*$', line):
            continue
        
        # Identify and collect citations
        if re.match(r'^\d+\s+\(', line):
            citations.append(line.strip())
            continue
        
        # Remove page numbers from the end of lines
        # might remove some important part of content, thus commented
        # line = re.sub(r'\s+\d+\s*$', '', line)
        
        processed_lines.append(line)
    
    # Join the processed lines
    processed_text = '\n'.join(processed_lines)
    
    # Add collected citations at the end
    if citations:
        processed_text += '\n\nCitations:\n' + '\n'.join(citations)
    
    return processed_text


def read_pdf_file(file_path: str | Path, use_llama_parse: bool = False) -> List[Document]:
    """
    Read a single PDF file and return its content as a list of Document objects.

    Args:
    file_path (str): Path to the PDF file.
    use_llama_parse (bool): Whether to use LlamaParse for parsing.

    Returns:
    List[Document]: List of Document objects containing the PDF content.
    """
    if use_llama_parse:
        parser = LlamaParse(result_type="markdown")
        file_extractor = {".pdf": parser}
    logger.info(f"Attempting to read PDF file: {file_path}")

    try:
        if use_llama_parse:
            documents = SimpleDirectoryReader(
                input_files=[file_path], file_extractor=file_extractor
            ).load_data()
        else:
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        if documents:
            # Apply preprocessing to each document
            for i, doc in tqdm(enumerate(documents), desc="Pre-processing documents"):
                documents[i].text = preprocess_citations(doc.text)
            
            logger.success(f"Successfully read and preprocessed PDF file: {file_path}")
            logger.debug(f"Number of documents: {len(documents)}")
            logger.debug(f"First document sample: {documents[0].text[:100]}")

            # Save the markdown content to a file
            output_dir = DATA_DIR / "converted_markdown"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{Path(file_path).stem}.md"

            with open(output_file, "w", encoding="utf-8") as f:
                for doc in documents:
                    f.write(doc.text + "\n\n")

            logger.info(f"Saved markdown content to: {output_file}")
            return documents
        else:
            logger.warning(f"PDF file is empty: {file_path}")
            return []
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {str(e)}")
        return []


def read_pdf_directory(directory_path):
    """
    Recursively read all PDF files in a directory and return their content in markdown format.

    Args:
    directory_path (str): Path to the directory containing PDF files.

    Returns:
    dict: A dictionary where keys are file paths and values are the content in markdown format.
    """
    logger.info(f"Attempting to read PDF files from directory: {directory_path}")
    parser = LlamaParse(result_type="markdown")
    file_extractor = {".pdf": parser}

    pdf_contents = {}

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                logger.info(f"Processing PDF file: {file_path}")
                try:
                    documents = SimpleDirectoryReader(
                        input_files=[file_path], file_extractor=file_extractor
                    ).load_data()
                    if documents:
                        pdf_contents[file_path] = documents[0].text
                        logger.success(f"Successfully read PDF file: {file_path}")
                    else:
                        logger.warning(f"PDF file is empty: {file_path}")
                except Exception as e:
                    logger.error(f"Error reading PDF file {file_path}: {str(e)}")

    logger.info(f"Finished reading PDF files from directory: {directory_path}")
    return pdf_contents


# testing
if __name__ == "__main__":
    test_file_path = DATA_DIR / TEST_FILE
    test_content = read_pdf_file(test_file_path)
    print(test_content)
