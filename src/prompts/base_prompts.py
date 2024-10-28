SYSTEM_PROMPT_TEMPLATE = """
You are an AI assistant specialized in {domain}. Your task is to {task_description}.

Input Context: {context}

Please provide your response following these guidelines:
{guidelines}
"""

SUMMARY_BASE_TEMPLATE = """
Provide a comprehensive summary of the following content, focusing on these key aspects:

{aspects}

Previous Context: {prev_summary}

Current Content to Analyze: {current_text}

Guidelines:
{guidelines}
"""
