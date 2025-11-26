"""
LLM-based log classifier module.

This module provides intelligent log classification and parsing using Groq's LLM models.
It can detect various log formats and extract structured data from them.
"""

import json
import csv
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import (
    LLM_CLASSIFIER_MODEL,
    LLM_PARSER_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    BATCH_SIZE,
    MAX_SAMPLE_LINES,
    CSV_OUTPUT_PATH
)


# Classification prompt template
CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a log file classification expert. Analyze the following log samples and:
1. Identify the log type (Kernel, Syslog, DMESG, OVS, Apache, Nginx, Application, JSON, Docker, Custom, etc.)
2. Provide confidence score (0-100)
3. List all detected fields/columns (timestamp, severity, host, message, etc.)
4. Describe the log format structure

You MUST respond with ONLY valid JSON, no additional text. Use this exact format:
{
  "log_type": "type_name",
  "confidence": 95,
  "detected_fields": ["field1", "field2", "field3"],
  "format_description": "Brief description of the log format"
}"""),
    ("user", "Analyze these log samples and classify them:\n\n{log_samples}")
])

# Schema extraction prompt
SCHEMA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a log parsing expert. Analyze the log samples and extract the column schema.
Identify all distinct fields present in the logs such as: timestamp, date, time, hostname, severity, 
log_level, process, pid, message, module, facility, etc.

You MUST respond with ONLY a JSON array of field names, no additional text. Example:
["timestamp", "hostname", "severity", "message"]"""),
    ("user", "Extract the schema from these log samples:\n\n{log_samples}")
])

# Parsing prompt template
PARSING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a log parsing expert. Parse the given log lines into structured CSV format.

Log format description: {format_description}
Expected fields (CSV columns): {fields}

For each log line, extract the values for each field. If a field is not present in a line, use empty string.
Output ONLY the CSV data rows (no headers), one row per log line.
Use comma as delimiter. Wrap values containing commas in double quotes.
Do not include any explanation, just the CSV rows."""),
    ("user", "Parse these log lines:\n\n{log_lines}")
])


def classify_log_type(log_sample: str, groq_api_key: str) -> dict:
    """
    Classify log type using LLM.
    
    Args:
        log_sample: Sample log lines (first few lines from log file)
        groq_api_key: API key for Groq
        
    Returns:
        dict: {
            'log_type': str,  # e.g., "Kernel", "Syslog", "Custom"
            'confidence': float,
            'detected_fields': list,  # e.g., ['timestamp', 'severity', 'message']
            'format_description': str
        }
    """
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=LLM_CLASSIFIER_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        # Format the prompt
        prompt = CLASSIFICATION_PROMPT.format_messages(log_samples=log_sample)
        
        # Get response from LLM
        response = llm.invoke(prompt)
        
        # Parse JSON response
        response_text = response.content.strip()
        
        # Try to extract JSON from the response
        result = _parse_json_response(response_text)
        
        if result:
            return {
                'log_type': result.get('log_type', 'Custom'),
                'confidence': float(result.get('confidence', 50)),
                'detected_fields': result.get('detected_fields', ['message']),
                'format_description': result.get('format_description', 'Unknown format')
            }
        
    except Exception as e:
        print(f"LLM classification error: {e}")
    
    # Return default if classification fails
    return {
        'log_type': 'Custom',
        'confidence': 0,
        'detected_fields': ['message'],
        'format_description': 'Could not determine format'
    }


def extract_log_schema(log_sample: str, groq_api_key: str) -> list:
    """
    Extract the schema/structure of the log using LLM.
    
    Args:
        log_sample: Sample log lines
        groq_api_key: API key for Groq
        
    Returns:
        list: Column names (e.g., ['timestamp', 'host', 'severity', 'message'])
    """
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=LLM_CLASSIFIER_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        prompt = SCHEMA_PROMPT.format_messages(log_samples=log_sample)
        response = llm.invoke(prompt)
        
        response_text = response.content.strip()
        
        # Try to parse as JSON array
        result = _parse_json_response(response_text)
        
        if isinstance(result, list) and len(result) > 0:
            return result
            
    except Exception as e:
        print(f"Schema extraction error: {e}")
    
    # Return default schema
    return ['timestamp', 'message']


def llm_based_parser(log_file_path: str, classification_result: dict, 
                     groq_api_key: str, csv_output_path: str = CSV_OUTPUT_PATH) -> bool:
    """
    Use LLM to parse log file into structured CSV.
    
    Args:
        log_file_path: Path to the log file
        classification_result: Result from classify_log_type()
        groq_api_key: API key for Groq
        csv_output_path: Path for output CSV file
        
    Returns:
        bool: Success status
    """
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=LLM_PARSER_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        fields = classification_result.get('detected_fields', ['message'])
        format_description = classification_result.get('format_description', '')
        
        # Read log file
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        # Write CSV with headers
        with open(csv_output_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(fields)  # Write header
            
            # Process in batches
            for i in range(0, len(lines), BATCH_SIZE):
                batch = lines[i:i + BATCH_SIZE]
                batch_text = ''.join(batch)
                
                if not batch_text.strip():
                    continue
                
                # Get LLM to parse this batch
                prompt = PARSING_PROMPT.format_messages(
                    format_description=format_description,
                    fields=', '.join(fields),
                    log_lines=batch_text
                )
                
                response = llm.invoke(prompt)
                parsed_rows = _parse_csv_response(response.content, len(fields))
                
                for row in parsed_rows:
                    csv_writer.writerow(row)
        
        return True
        
    except Exception as e:
        print(f"LLM parsing error: {e}")
        return False


def get_log_sample(file_path: str, num_lines: int = MAX_SAMPLE_LINES) -> str:
    """
    Get a sample of log lines from a file for classification.
    
    Args:
        file_path: Path to the log file
        num_lines: Number of lines to sample
        
    Returns:
        str: Sample log lines
    """
    sample_lines = []
    try:
        with open(file_path, 'r') as f:
            for _ in range(num_lines):
                line = f.readline()
                if not line:
                    break
                sample_lines.append(line)
    except Exception as e:
        print(f"Error reading log sample: {e}")
    
    return ''.join(sample_lines)


def _parse_json_response(response_text: str):
    """
    Parse JSON from LLM response, handling potential formatting issues.
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Parsed JSON object or None
    """
    # First try direct parsing
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in the response
    # Look for JSON object
    json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Look for JSON array
    array_match = re.search(r'\[[^\[\]]*\]', response_text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass
    
    return None


def _parse_csv_response(response_text: str, expected_columns: int) -> list:
    """
    Parse CSV rows from LLM response.
    
    Args:
        response_text: Raw response text from LLM
        expected_columns: Expected number of columns
        
    Returns:
        list: List of row lists
    """
    rows = []
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip if it looks like a header or explanation
        if line.lower().startswith(('here', 'the ', 'csv', 'output', 'parsed')):
            continue
        
        # Parse the CSV line
        try:
            # Use csv module for proper parsing
            import io
            reader = csv.reader(io.StringIO(line))
            row = next(reader, None)
            
            if row:
                # Pad or trim to expected columns
                if len(row) < expected_columns:
                    row.extend([''] * (expected_columns - len(row)))
                elif len(row) > expected_columns:
                    row = row[:expected_columns]
                rows.append(row)
        except Exception:
            # Try simple split as fallback
            parts = line.split(',')
            if len(parts) >= expected_columns:
                rows.append(parts[:expected_columns])
            elif len(parts) > 0:
                parts.extend([''] * (expected_columns - len(parts)))
                rows.append(parts)
    
    return rows
