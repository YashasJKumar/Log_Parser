# Upgrade Notes: LLM-Based Log Classification

## Overview

This upgrade replaces the rigid regex-based log classification system with an intelligent LLM-based approach using Groq Cloud. The log parser can now handle ANY log format, not just the 4 predefined types.

## What Changed

### New Files
- **`config.py`**: Centralized configuration for LLM models and parsing parameters
- **`llm_classifier.py`**: Main LLM classification and parsing logic
- **`UPGRADE_NOTES.md`**: This documentation file

### Modified Files
- **`parser.py`**: Integrated LLM classifier with regex fallback
- **`helper_functions.py`**: Enhanced `file_parser()` to use LLM classification
- **`main.py`**: Added UI elements to display detected log structure

## How the New LLM Classifier Works

### Classification Flow
```
User uploads log → LLM Classification → Type Detection + Schema Extraction → 
    → If known type: Use regex parser (fast)
    → If custom type: Use LLM parser
→ CSV output → RAG
```

### Key Functions

#### `classify_log_type(log_sample, groq_api_key)`
- Sends first 10 lines of log file to LLM
- Returns log type, confidence score, detected fields, and format description
- Uses Llama 3.3 70B for accurate classification

#### `llm_based_parser(log_file_path, classification_result, groq_api_key, csv_output_path)`
- Parses entire log file using LLM
- Processes logs in batches of 50 lines for efficiency
- Uses Llama 3.1 8B (faster model) for parsing

#### `parse_log(input_file_path, use_llm=True, groq_api_key=None)`
- Enhanced main parsing function
- Uses LLM classification first, falls back to regex if needed
- Returns classification details for UI display

## Performance Comparison

| Aspect | Regex-Based | LLM-Based |
|--------|-------------|-----------|
| Classification Speed | ~1ms | ~500ms-2s |
| Parsing Speed | Very fast | Slower (batched) |
| Format Support | 4 types only | Unlimited |
| Accuracy | Pattern-dependent | Context-aware |
| Edge Case Handling | Poor | Good |

## Fallback Behavior

1. **LLM Available + Known Type**: Uses regex for parsing (fastest)
2. **LLM Available + Unknown Type**: Uses LLM for parsing
3. **LLM Unavailable**: Falls back to regex-only (original behavior)
4. **LLM Fails Mid-Parse**: Graceful error handling, reports failure

## API Usage Considerations (Groq Free Tier)

### Rate Limits
- Classification: ~1 API call per file upload
- Parsing: ~1 API call per 50 log lines
- Stay within Groq free tier limits

### Cost Optimization
- Only first 10 lines sent for classification
- Batch processing (50 lines) for parsing
- Uses smaller model (8B) for parsing
- Uses larger model (70B) only for classification

## Supported Log Types

### Built-in Regex Support
- Kernel logs
- DMESG logs
- OVS (Open vSwitch) logs
- Syslog

### LLM-Detected (New)
- Apache/Nginx access logs
- Application logs (JSON format)
- Docker container logs
- Custom application logs
- Multi-line stack traces
- Any structured log format

## Configuration Options

Edit `config.py` to customize:

```python
# Model selection
LLM_CLASSIFIER_MODEL = "llama-3.3-70b-versatile"
LLM_PARSER_MODEL = "llama-3.1-8b-instant"

# Performance tuning
BATCH_SIZE = 50  # Lines per parsing batch
MAX_SAMPLE_LINES = 10  # Lines for classification

# Output
CSV_OUTPUT_PATH = "./parsed_log_data.csv"
```

## Migration Notes

### Backward Compatibility
- Existing functionality preserved
- Original regex parsers still available as fallback
- Same CSV output format for RAG compatibility

### Breaking Changes
- `parse_log()` now returns 3 values instead of 2
- Use `parse_log_legacy()` for original 2-value return

## Troubleshooting

### LLM Classification Failed
- Check Groq API key is valid
- Check internet connectivity
- Will automatically fall back to regex

### Slow Parsing
- Large files processed in batches
- Consider increasing `BATCH_SIZE` for faster processing
- Trade-off: larger batches may hit token limits

### Low Confidence Score
- Score < 30% triggers unknown format handling
- Try providing more log samples (increase `MAX_SAMPLE_LINES`)
