import json

def read_jsonl(file_path):
    summaries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                summaries.append(json.loads(line))
    return summaries

# Test the function
if __name__ == "__main__":
    file_path = r"D:\NIPL2093\work\long_doc_summarization\data\outputs\37657_2017_1_1501_39247_Judgement_19-Oct-2022_149pg_summary.json"
    
    try:
        summaries = read_jsonl(file_path)
        print(f"Successfully read {len(summaries)} entries from the JSONL file")
        
        # Print the first summary to verify the content (if exists)
        if summaries:
            print("\nFirst entry preview:")
            print(json.dumps(summaries[0], indent=2))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
