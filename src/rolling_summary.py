import time
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional
from google.api_core.exceptions import ResourceExhausted
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.base.llms.types import CompletionResponse

from src.data_processing.data_loader import read_pdf_file
from src.model_handlers.model_loaders import ModelInitializer

# Configure logger
logger.add("rolling_summary.log", rotation="10 MB")

# System prompt for legal document summarization
LEGAL_SUMMARY_PROMPT = """
You are an AI assistant helping legal professionals summarize court judgments and orders. Your task is to provide a detailed, 
comprehensive summary that captures all key legal details. Focus on the following aspects (including but not limited to):

1. Case Basics:
   - Case type and jurisdiction
   - Parties involved (with clear designations)
   - Main legal issues presented

2. Factual Background:
   - Chronological sequence of relevant events upon which the case is built
   - Key disputed facts
   - Undisputed facts

3. Legal Arguments:
   - Arguments presented by each party
   - Specific legal provisions cited (with section/article numbers)
   - Precedents relied upon

4. Court's Analysis:
   - Court's reasoning on each major point
   - Interpretation of evidence and legal principles
   - Treatment of precedents

5. Decision and Outcome:
   - Final orders and directives
   - Relief granted or denied
   - Any specific conditions or timelines set

6. Other Relevant Information:
   - Costs and compensation details
   - Interim orders
   - Special observations by the court

Important Guidelines:
- DO NOT include any information that is not mentioned in the current chunk.
- AVOID redundanct usage of the phrase "The court has ..."
- Maintain precise legal terminology
- Include all statute references with section numbers in brackets where applicable as seen in the original document
- Cite case laws in proper format
- Preserve numerical details, dates and timelines
- Note any dissenting opinions
- Highlight any new legal principles established

Previous Summary Context: {prev_summary}

Based on the above context and the current portion of the judgment, provide a comprehensive summary that builds upon and 
integrates with the previous summary while adding new details from the current portion.
"""

class RollingSummary:
    def __init__(
        self, 
        model_name: str = "gemini-1.5-flash",
        chunk_size: int = 512,
        chunk_overlap: int = 20
    ):
        """
        Initialize the rolling summary processor
        
        Args:
            model_name: Name of the model to use
            chunk_size: Size of chunks to process at once
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.model = ModelInitializer.initialize_gemini(
            model_name=model_name,
            use_llamaindex=True
        )
        # self.model = ModelInitializer.initialize_groq(
        #     model_name="llama-3.1-70b-versatile",
        #     use_llamaindex=True
        # )
        # self.model = ModelInitializer.initialize_cerebras(
        #     model_name="llama3.1-70b",
        #     use_llamaindex=True
        # )
        
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.token_usage_data = []
        logger.info(f"Initialized RollingSummary with model {model_name}")

    # @retry(
    #     wait=wait_fixed(1),
    #     stop=stop_after_attempt(3),
    #     retry=retry_if_exception_type(ResourceExhausted)
    # )
    def generate_rolling_summary(
        self,
        current_chunk: str,
        previous_summary: Optional[str] = None
    ) -> str:
        """
        Generate summary incorporating previous context
        
        Args:
            current_chunk: Current text chunk to summarize
            previous_summary: Previous summary to build upon
            
        Returns:
            Updated comprehensive summary
        """
        try:
            # Construct the prompt
            prompt = LEGAL_SUMMARY_PROMPT.format(
                prev_summary=previous_summary if previous_summary else "No previous summary available."
            )
            
            # Add the current chunk to analyze
            full_prompt = f"{prompt}\n\nCurrent text to analyze:\n{current_chunk}"
            
            # Get completion from model
            logger.debug(f"Generating summary for chunk of length {len(current_chunk)}")
            response: CompletionResponse = self.model.complete(
                prompt=full_prompt,
                formatted=True
            )
            logger.debug(f"Response object: {repr(response)}")
            
            if not response or not response.text:
                logger.warning("Received empty response from model")
                return previous_summary if previous_summary else ""
                
            logger.debug("Successfully generated summary")

            # Store token usage data: Gemini
            if response.raw and response.raw.get('usage_metadata'):
                usage = response.raw['usage_metadata']
                self.token_usage_data.append({
                    'prompt_tokens': usage.get('prompt_token_count', 0),
                    'candidates_token_count': usage.get('candidates_token_count', 0),
                    'total_tokens': usage.get('total_token_count', 0)
                })
                logger.debug(f"Token usage - Prompt: {usage.get('prompt_token_count', 0)}, Candidates: {usage.get('candidates_token_count', 0)}, Total: {usage.get('total_token_count', 0)}")

            # Store token usage data: Groq
            # if response.raw and hasattr(response.raw, 'usage'):
            #     usage = response.raw.usage
            #     self.token_usage_data.append({
            #         'prompt_tokens': usage.prompt_tokens,
            #         'completion_tokens': usage.completion_tokens,
            #         'total_tokens': usage.total_tokens
            #     })
            #     logger.debug(f"Token usage - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")

            # Store token usage data: Cerebras
            # if response.raw and hasattr(response.raw, 'usage'):
            #     usage = response.raw.usage
            #     self.token_usage_data.append({
            #         'prompt_tokens': usage.prompt_tokens,
            #         'completion_tokens': usage.completion_tokens,
            #         'total_tokens': usage.total_tokens
            #     })
            #     logger.debug(f"Token usage - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")


            # Log first few and last few sentences of the generated summary
            sentences = response.text.split('. ')
            first_sentences = '. '.join(sentences[:3]) + '.'
            last_sentences = '. '.join(sentences[-3:]) + '.'
            logger.info("Generated summary preview:")
            logger.info(f"{first_sentences}...{last_sentences}")

            return response.text

        except ResourceExhausted as e:
            logger.warning(f"Rate limit exceeded, retrying: {str(e)}")
            raise  # Will be caught by retry decorator
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise

    def process_document(self, file_path: str | Path) -> tuple[str, int, int]:
        """
        Process entire document using rolling summary technique
        
        Args:
            file_path: Path to the document
            
        Returns:
            Tuple of (final comprehensive summary, number of pages, number of chunks)
        """
        logger.info(f"Starting document processing: {file_path}")
        
        # Load document
        documents = read_pdf_file(file_path)
        if not documents:
            logger.error("No documents loaded")
            raise ValueError("No documents loaded")
            
        num_pages = len(documents)
        logger.info(f"Loaded {num_pages} document segments")
        
        # Split into chunks
        chunks = self.splitter.get_nodes_from_documents(documents)
        num_chunks = len(chunks)
        logger.info(f"Split into {num_chunks} chunks")
        
        # Process chunks sequentially
        current_summary = None
        for i, chunk in enumerate(chunks, 1):
            try:
                logger.info(f"Processing chunk {i}/{num_chunks}")
                current_summary = self.generate_rolling_summary(
                    current_chunk=chunk.text,
                    previous_summary=current_summary
                )
                logger.info(f"Successfully processed chunk {i}")
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                raise
                
        logger.success("Completed document processing")
        return current_summary, num_pages, num_chunks

def main():
    # Define constants from doc_summ_index.py
    DATA_DIR = Path("D:\\NIPL2093\\work\\long_doc_summarization\\data\\")
    TEST_FILE = "37657_2017_1_1501_39247_Judgement_19-Oct-2022_149pg.pdf"
    OUTPUT_DIR = DATA_DIR / "outputs"
    MODEL_LLM = "gemini-1.5-flash-002"

    # Create rolling summary output directory
    rolling_summary_dir = OUTPUT_DIR / "rolling_summaries"
    rolling_summary_dir.mkdir(parents=True, exist_ok=True)

    # Initialize RollingSummary
    logger.info("Initializing RollingSummary processor")
    processor = RollingSummary(model_name=MODEL_LLM)

    try:
        # Start timing
        start_time = time.time()

        # Process document
        test_file_path = DATA_DIR / TEST_FILE
        summary, num_pages, num_chunks = processor.process_document(test_file_path)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"Took {minutes} minutes and {seconds} seconds to process {num_pages} total pages ({num_chunks} chunks)."

        # Add timing information to summary
        summary_with_time = f"{summary}\n\n{time_str}"

        # Save output
        output_file = rolling_summary_dir / f"{TEST_FILE.split('.')[0]}_rolling_summary.txt"
        logger.info(f"Saving rolling summary to: {output_file}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary_with_time)
            
        # Save token usage analysis
        token_df = pd.DataFrame(processor.token_usage_data)
        token_df['total_tokens'] = token_df['prompt_tokens'] + token_df['completion_tokens']
        token_analysis_file = rolling_summary_dir / "tokens_analysis.csv"
        token_df.to_csv(token_analysis_file, index=False)
        logger.info(f"Saved token usage analysis to {token_analysis_file}")
            
        logger.success(f"Successfully saved rolling summary to {output_file}")
        logger.info(time_str)

    except Exception as e:
        logger.exception(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
