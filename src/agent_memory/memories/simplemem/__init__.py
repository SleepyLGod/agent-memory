"""SimpleMem-style built-in memory policy."""

from .policy import (
    SIMPLEMEM_EXTRACTION_PROMPT,
    SimpleMemMemory,
    WINDOW_SIZE,
    WINDOW_SLIDE,
)
from .policy_enhanced import SimpleMemMemoryEnhanced
from .storage import (
    SIMPLEMEM_BGE_M3,
    SIMPLEMEM_ENHANCED_NEO4J_STATEMENTS,
    SIMPLEMEM_NEO4J_SCHEMA,
    SIMPLEMEM_NEO4J_STATEMENTS,
)

__all__ = [
    "SIMPLEMEM_BGE_M3",
    "SIMPLEMEM_ENHANCED_NEO4J_STATEMENTS",
    "SIMPLEMEM_EXTRACTION_PROMPT",
    "SIMPLEMEM_NEO4J_SCHEMA",
    "SIMPLEMEM_NEO4J_STATEMENTS",
    "SimpleMemMemory",
    "SimpleMemMemoryEnhanced",
    "WINDOW_SIZE",
    "WINDOW_SLIDE",
]
