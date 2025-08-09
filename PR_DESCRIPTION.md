# Pull Request: Fix Pre-load embedding model to eliminate 10-20s delays

**Branch:** `fix/preload-embedding-model`  
**Target:** `main`  
**PR URL:** https://github.com/TheRealClodius/Memory-Signal/pull/new/fix/preload-embedding-model

## Problem
The memory add operation was experiencing **10-20 second delays** when the short-term memory became full (typically on the 11th addition with default capacity of 10). 

### Root Cause
- The embedding model (~90MB) was being loaded lazily from HuggingFace
- This download/initialization happened during the first embedding generation
- Users experienced this delay mid-operation, causing poor UX

## Solution
Pre-load the embedding model during MemoryOS initialization (server startup) instead of waiting for first use.

### Changes Made
- Added `_preload_embedding_model()` method to `Memoryos` class in `memoryos-mcp/memoryos/memoryos.py`
- Calls pre-loading during `__init__` to load model at server startup
- Generates a dummy embedding to trigger model download/initialization

### Code Changes
```python
# In __init__ method (line 136):
# Pre-load embedding model to avoid runtime delays
self._preload_embedding_model()

# New method (lines 138-156):
def _preload_embedding_model(self):
    """
    Pre-loads the embedding model during initialization to avoid delays during first use.
    This prevents the 10-20 second delay that occurs when embeddings are first needed.
    """
    try:
        from .utils import get_embedding
    except ImportError:
        from utils import get_embedding
    
    print(f"Pre-loading embedding model: {self.embedding_model_name}...")
    # Generate a dummy embedding to trigger model loading
    dummy_text = "Initialization test"
    _ = get_embedding(
        dummy_text, 
        model_name=self.embedding_model_name,
        use_cache=False,  # Don't cache this dummy embedding
        **self.embedding_model_kwargs
    )
    print(f"✅ Embedding model pre-loaded successfully")
```

## Performance Impact

### Before
- Memory operations 1-10: ~0.3s each
- Memory operation 11: **10-20 seconds** (model loading)
- Memory operations 12+: ~0.3s each

### After
- Server startup: +0.36s (one-time cost)
- **ALL memory operations: <1 second**
- Consistent performance across all operations

## Testing
Tested with profiling script that measures:
1. Server initialization time (includes pre-loading)
2. 12 consecutive memory additions
3. Verified no delays during user operations

### Test Results
```
Server initialization time: 0.36s (includes model pre-loading)
Memory  1: 0.352s
Memory  2: 0.180s
Memory  3: 0.285s
Memory  4: 0.368s
Memory  5: 0.406s
Memory  6: 0.222s
Memory  7: 0.314s
Memory  8: 0.280s
Memory  9: 0.257s
Memory 10: 0.561s
Memory 11: 0.267s  <- No delay here anymore!
Memory 12: 0.348s
```

## Impact
This change moves the model loading delay from runtime (when users are waiting) to server initialization (when no one is waiting), providing a much better user experience.

## Files Changed
- `memoryos-mcp/memoryos/memoryos.py` - Added pre-loading logic