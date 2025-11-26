# Configuration file for LLM-based log parsing

# LLM Configuration
LLM_CLASSIFIER_MODEL = "llama-3.3-70b-versatile"  # Best for classification
LLM_PARSER_MODEL = "llama-3.1-8b-instant"  # Faster for parsing
LLM_TEMPERATURE = 0.1  # Low temperature for consistent parsing
LLM_MAX_TOKENS = 2048

# Parsing Configuration
BATCH_SIZE = 50  # Process logs in batches for efficiency
MAX_SAMPLE_LINES = 10  # Lines to sample for classification
CSV_OUTPUT_PATH = "./parsed_log_data.csv"

# Classification threshold - confidence score below this triggers fallback
CONFIDENCE_THRESHOLD = 30

# Keywords to skip when parsing CSV response (explanatory text from LLM)
CSV_SKIP_KEYWORDS = ('here', 'the ', 'csv', 'output', 'parsed')

# Known log types (for reference, LLM can detect more)
KNOWN_LOG_TYPES = [
    "Kernel",
    "DMESG",
    "OVS",
    "Syslog",
    "Apache",
    "Nginx",
    "Application",
    "JSON",
    "Docker",
    "Custom"
]
