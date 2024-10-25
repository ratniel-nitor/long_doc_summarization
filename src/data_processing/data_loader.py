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
    """
    Extract citations from text and append them in a structured format at the end.
    
    Args:
        text (str): Input text containing citations
        
    Returns:
        str: Processed text with citations extracted and appended at the end
    """
    # Split text into lines
    lines = text.split('\n')
    
    citations = []
    main_text_lines = []
    citation_started = False
    current_citation = []
    
    for line in lines:
        # Check if line starts citation block
        if re.match(r'^\s*\d+\s+', line) and (len(main_text_lines) > 0 and main_text_lines[-1].strip() == ''):
            citation_started = True
            
        if citation_started:
            # If it's a new citation (starts with number)
            if re.match(r'^\s*\d+\s+', line):
                # Save previous citation if exists
                if current_citation:
                    citations.append(' '.join(current_citation))
                # Start new citation
                current_citation = [line.strip()]
            else:
                # Continue previous citation
                if line.strip():
                    current_citation.append(line.strip())
        else:
            main_text_lines.append(line)
    
    # Add last citation if exists
    if current_citation:
        citations.append(' '.join(current_citation))
    
    # Format citations
    formatted_citations = []
    for citation in citations:
        # Extract citation number and text
        match = re.match(r'^\s*(\d+)\s+(.+)$', citation)
        if match:
            num, text = match.groups()
            formatted_citations.append(f"{num}: {text}")
    
    # Combine main text and formatted citations
    result = '\n'.join(main_text_lines).strip()
    if formatted_citations:
        result += '\n\nCitations:\n' + '\n'.join(formatted_citations) + '\n\n'
    
    return result


def preprocess_page_number_prefix(text: str) -> str:
    # Remove page number pattern from beginning of text
    # Matches pattern like "' 4 \n", "' 124 \n" etc.
    # Where there's a single quote, space, page number, space, and newline
    if text.startswith("PART"):
        # Remove patterns like "PART I & II  \n4 \n" or "PART XII  \n78 \n"
        return re.sub(r"^PART.*?\n\d+\s+\n\s*", '', text)
    elif text.startswith(" "):
        return re.sub(r"^\s+\d+\s+\n", '', text)
    return text


# def preprocess_add_ellipsis(text: str) -> str:
#     """
#     Add ellipsis at the beginning of text if it doesn't start with a capital letter
#     and doesn't match the pattern of a new point/section start.
    
#     Args:
#         text (str): Input text to process
        
#     Returns:
#         str: Processed text with ellipsis added if needed
#     """
#     try:
#         # Return unchanged if empty
#         if not text or not text.strip():
#             return text
            
#         # Check if text starts with capital letter
#         if text[0].isupper():
#             return text
            
#         # Pattern for point/section start:
#         # <number/letter/roman numerals/small case roman letters within brackets><period><space><capital_letter>
#         point_start_pattern = r'^(?:\d+|\([a-zA-Z0-9ivxlcdmIVXLCDM]+\))\.\s[A-Z]'
#         if re.match(point_start_pattern, text):
#             return text
            
#         # Add ellipsis if conditions not met
#         return f"... {text}"
#     except Exception as e:
#         logger.error(f"Error in preprocess_add_ellipsis: {str(e)}")
#         return text  # Return original text in case of error


def read_pdf_file(file_path: str | Path, use_llama_parse: bool = False, return_chunks: bool = False) -> List[Document]:
    """
    Read a single PDF file and return its content as a list of Document objects.

    Args:
    file_path (str): Path to the PDF file.
    use_llama_parse (bool): Whether to use LlamaParse for parsing.
    return_chunks (bool): If True, returns chunked nodes using SentenceWindowNodeParser.
                         If False, returns original documents.

    Returns:
    List[Document]: List of Document objects containing the PDF content,
                   or List of chunked nodes if return_chunks is True.
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
                documents[i].text = preprocess_page_number_prefix(doc.text)
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

            if return_chunks:
                from llama_index.core.node_parser import SentenceWindowNodeParser
                node_parser = SentenceWindowNodeParser.from_defaults(
                    window_size=3,
                    window_metadata_key="window",
                    original_text_metadata_key="original_text",
                )
                nodes = node_parser.get_nodes_from_documents(documents)
                return nodes
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
    # Save content to text file
    output_file = DATA_DIR / f"{TEST_FILE.stem}.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in test_content:
                f.write(doc.text)
        logger.success(f"Successfully saved content to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save content to file: {str(e)}")
    print(test_content)
