"""
LLM Prompt Templates and Constants

This module contains all the prompt templates used for LLM inference.
"""

# ============================================================================
# EXTRACTION PROMPTS
# ============================================================================

# System prompt - keep it short and direct
SYSTEM_PROMPT_EXTRACT = """You extract occupation titles and skills from postings. You respond ONLY in this format:
Title: [occupation name]
Skills: [skill1, skill2, skill3]

Do not write anything else. Do not acknowledge. Just extract and respond in the format."""

# User prompt - direct and simple
USER_PROMPT_EXTRACT_PRIMARY = """Extract from this posting:

{text}

Remember: respond ONLY with:
Title: [occupation name]
Skills: [skill1, skill2, skill3]"""

# ============================================================================
# PARSING PATTERNS
# ============================================================================

# Title extraction patterns (regex)
TITLE_PATTERNS = [
    r"[*\s]*title[:\s]+([^\n]+)",
    r"[*\s]*occupation title[:\s]+([^\n]+)",
    r"[*\s]*position[:\s]+([^\n]+)",
]

# Skills extraction patterns (regex)
SKILLS_PATTERNS = [
    r"[*\s]*skills?[:\s]+([^\n]+)",
    r"[*\s]*required skills?[:\s]+([^\n]+)",
    r"[*\s]*technical skills?[:\s]+([^\n]+)",
]

# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

# Phrases that indicate LLM returned acknowledgment instead of extraction
NON_RESPONSE_PHRASES = [
    "okay, i understand",
    "i will extract",
    "i'll extract",
    "let me extract",
    "here's what i'll do"
]

# Invalid title values to reject
INVALID_TITLES = ['unknown', 'unknown position', 'n/a', 'none']

# Minimum response length (characters)
MIN_RESPONSE_LENGTH = 10

# Minimum skill length (characters)
MIN_SKILL_LENGTH = 2
