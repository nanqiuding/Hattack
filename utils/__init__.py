from .utils import (
    set_seed,
    to_tensor,
    write_json,
    read_json
)

# Optional API clients are imported only by scripts that use them.
def __getattr__(name):
    if name == "GPTScorer":
        from .scorer import GPTScorer
        return GPTScorer
    if name == "KeywordMatcher":
        from .matcher import KeywordMatcher
        return KeywordMatcher
    if name in ("ImageDescriptionGeneratorOpen", "ImageDescriptionGeneratorClose"):
        from . import generator
        return getattr(generator, name)
    raise AttributeError(name)
