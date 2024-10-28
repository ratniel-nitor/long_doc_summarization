from datetime import datetime
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional, Literal, Union
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.base.llms.types import CompletionResponse
from pydantic import BaseModel, Field
import json

from src.data_processing.data_loader import read_pdf_file
from src.model_handlers.model_loaders import ModelInfo, ModelInitializer
from src.prompts.legal_prompts import LEGAL_SUMMARY_PROMPT

# Configure logger
logger.add("rolling_summary.log", rotation="10 MB")

class SummaryProgress(BaseModel):
    """Track document processing progress"""
    file_name: str
    total_items: int
    current_index: int = 0
    current_summary: str = ""
    completed: bool = False
    processing_type: Literal["chunk", "document"] = "document"
    start_time: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    def model_dump_json(self, *args, **kwargs) -> str:
        """Convert to JSON string"""
        return json.dumps(
            {
                "file_name": self.file_name,
                "total_items": self.total_items,
                "current_index": self.current_index,
                "current_summary": self.current_summary,
                "completed": self.completed,
                "processing_type": self.processing_type,
                "start_time": self.start_time.isoformat(),
                "last_updated": self.last_updated.isoformat()
            },
            indent=2
        )

    @classmethod
    def model_validate_json(cls, json_str: str) -> "SummaryProgress":
        """Create from JSON string"""
        data = json.loads(json_str)
        data["start_time"] = datetime.fromisoformat(data["start_time"])
        data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)

class RollingSummary:
    def __init__(
        self, 
        model_name: str = "gemini-1.5-flash-002",
        api_key: Optional[str] = None,
        output_dir: Optional[Path] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 20
    ):
        """Initialize the rolling summary processor"""
        # Initialize model and get provider info
        self.model, self.provider = ModelInitializer.initialize_model(
            model_name=model_name,
            api_key=api_key
        )
        
        # Get token field and mapping for this provider
        self.token_field, self.token_mapping = ModelInfo.get_token_info(self.provider)
        
        self.output_dir = output_dir or Path("data/outputs/rolling_summaries")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Initialize sentence splitter
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.token_usage_data = []  # List of dicts instead of TokenUsage objects
        logger.info(f"Initialized RollingSummary with {self.provider.value} model: {model_name}")

    def _save_summary(self, file_name: str, summary: str, summary_type: str) -> None:
        """Save summary to text file"""
        try:
            summary_file = self.output_dir / f"{file_name}_{summary_type}_summary.txt"
            summary_file.write_text(summary, encoding='utf-8')  # Add encoding here
            logger.info(f"Saved {summary_type} summary to {summary_file}")
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
            raise

    def _save_progress(self, progress: SummaryProgress, file_path: Path) -> None:
        """Save progress to JSON file"""
        try:
            # Use consistent file naming
            progress_file = self.output_dir / f"{progress.file_name}_{progress.processing_type}_progress.json"
            progress.last_updated = datetime.now()
            progress_file.write_text(progress.model_dump_json(indent=2), encoding='utf-8')  # Add encoding here
            
            # Also save current summary as intermediate result
            self._save_summary(
                progress.file_name,
                progress.current_summary,
                f"{progress.processing_type}_intermediate"
            )
            
            logger.debug(f"Saved progress and intermediate summary for {progress.file_name}")
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
            raise

    def _save_token_analysis(self, file_name: str) -> None:
        """Save token usage analysis to CSV"""
        try:
            if not self.token_usage_data:
                logger.warning("No token usage data to save")
                return

            token_df = pd.DataFrame(self.token_usage_data)
            
            # Convert numpy values to native Python types
            stats = {
                'total_prompt_tokens': int(token_df['prompt_tokens'].sum()),
                'total_completion_tokens': int(token_df['completion_tokens'].sum()),
                'total_tokens': int(token_df['total_tokens'].sum()),
                'avg_tokens_per_request': float(token_df['total_tokens'].mean()),
                'num_requests': int(len(token_df))
            }
            
            # Save detailed token usage
            token_analysis_file = self.output_dir / f"{file_name}_token_analysis.csv"
            token_df.to_csv(token_analysis_file, index=False)
            
            # Save summary statistics
            stats_file = self.output_dir / f"{file_name}_token_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
                
            logger.info(f"Saved token analysis to {token_analysis_file}")
            logger.info(f"Saved token statistics to {stats_file}")
            logger.info(f"Total tokens used: {stats['total_tokens']:,}")
            
        except Exception as e:
            logger.error(f"Error saving token analysis: {e}")
            raise

    def process_document(
        self, 
        file_path: Union[str, Path],
        processing_type: Literal["chunk", "document"] = "document",
        resume: bool = True
    ) -> SummaryProgress:
        """
        Process document using rolling summary technique
        
        Args:
            file_path: Path to the document
            processing_type: Whether to process by chunks or full documents
            resume: Whether to resume from last saved progress
            
        Returns:
            SummaryProgress object containing progress information
        """
        file_path = Path(file_path)
        logger.info(f"Starting document processing: {file_path} using {processing_type} method")
        
        # Initialize or load progress
        progress = None
        if resume:
            try:
                progress_file = self.output_dir / f"{file_path.stem}_{processing_type}_progress.json"
                if progress_file.exists():
                    progress = SummaryProgress.model_validate_json(progress_file.read_text())
                    if progress.completed:
                        logger.info("Found completed summary, returning existing progress")
                        return progress
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
        
        # Load document and optionally split into chunks
        documents = read_pdf_file(file_path)
        if not documents:
            raise ValueError(f"No documents loaded from {file_path}")
        
        # Split into chunks if requested
        if processing_type == "chunk":
            documents = self.splitter.get_nodes_from_documents(documents)
            logger.info(f"Split documents into {len(documents)} chunks")
        
        # Initialize new progress if needed
        if not progress:
            progress = SummaryProgress(
                file_name=file_path.stem,
                total_items=len(documents),
                current_index=0,
                processing_type=processing_type
            )
        
        try:
            # Process items sequentially
            for i in range(progress.current_index, len(documents)):
                logger.info(f"Processing {processing_type} {i+1}/{len(documents)}")
                
                current_text = (
                    documents[i].text if processing_type == "document" 
                    else documents[i].text
                )
                
                new_summary = self.generate_rolling_summary(
                    current_text=current_text,
                    previous_summary=progress.current_summary
                )
                
                # Update progress and save intermediate results
                progress.current_index = i + 1
                progress.current_summary = new_summary
                self._save_progress(progress, file_path)
                
            # Mark as completed and save final results
            progress.completed = True
            self._save_progress(progress, file_path)
            
            # Save final summary
            self._save_summary(
                progress.file_name,
                progress.current_summary,
                progress.processing_type
            )
            
            # Save token analysis
            self._save_token_analysis(f"{file_path.stem}_{processing_type}")
            
            logger.success(f"Completed processing {file_path.name} using {processing_type} method")
            return progress
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            # Save progress and current state even on error
            if progress:
                self._save_progress(progress, file_path)
                self._save_token_analysis(f"{file_path.stem}_{processing_type}")
            raise

    def generate_rolling_summary(
        self,
        current_text: str,
        previous_summary: Optional[str] = None
    ) -> str:
        """Generate summary incorporating previous context"""
        try:
            prompt = LEGAL_SUMMARY_PROMPT.format(
                prev_summary=previous_summary if previous_summary else "No previous summary available."
            )
            
            full_prompt = f"{prompt}\n\nCurrent text to analyze:\n{current_text}"
            
            response: CompletionResponse = self.model.complete(
                prompt=full_prompt,
                formatted=True
            )
            
            if not response or not response.text:
                logger.warning("Received empty response from model")
                return previous_summary if previous_summary else ""

            # Handle token usage using provider-specific mapping
            if response.raw and response.raw.get(self.token_field):
                usage_data = response.raw[self.token_field]
                token_usage = {
                    "prompt_tokens": usage_data.get(self.token_mapping["prompt_tokens"], 0),
                    "completion_tokens": usage_data.get(self.token_mapping["completion_tokens"], 0),
                    "total_tokens": usage_data.get(self.token_mapping["total_tokens"], 0),
                    "provider": self.provider.value,
                    "timestamp": datetime.now().isoformat()
                }
                self.token_usage_data.append(token_usage)
                logger.debug(f"Token usage - Prompt: {token_usage['prompt_tokens']}, Completion: {token_usage['completion_tokens']}, Total: {token_usage['total_tokens']}")

            return response.text

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise

def main():
    """Test the implementation"""
    DATA_DIR = Path("data")
    TEST_FILE = "37657_2017_1_1501_39247_Judgement_19-Oct-2022_149pg.pdf"
    MODEL_LLM = "gemini-1.5-flash-002"
    PROCESSING_TYPE = "document"  # or "chunk" - single explicit choice

    processor = RollingSummary(model_name=MODEL_LLM)

    try:
        logger.info(f"Testing {PROCESSING_TYPE}-based summary")
        progress = processor.process_document(
            DATA_DIR / TEST_FILE,
            processing_type=PROCESSING_TYPE,
            resume=True
        )
        
        if progress.completed:
            logger.info(f"Successfully processed document using {PROCESSING_TYPE} method")
            logger.info(f"Processing time: {(datetime.now() - progress.start_time).total_seconds():.1f} seconds")
            logger.info(f"Total items processed: {progress.total_items}")
            
            # Save final summary
            summary_file = processor.output_dir / f"{progress.file_name}_{PROCESSING_TYPE}_summary.txt"
            summary_file.write_text(progress.current_summary, encoding='utf-8')
            logger.info(f"Saved {PROCESSING_TYPE} summary to {summary_file}")

    except Exception as e:
        logger.exception(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
