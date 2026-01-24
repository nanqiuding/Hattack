from .clip_extractor import (
    ClipExtractor,
    ClipB16Extractor,
    ClipB32Extractor,
    LaionExtractor,
)
from .FOA_extractor import (
    EnsembleExtractor,
    EnsembleLoss,
    EnsembleExtractor_OT_FOA,
    EnsembleLoss_OT_FOA
)
from .FOA_global_extractor import (
    EnsembleExtractor_global,
    EnsembleLoss_global
)
from .FOA_middle_extractor import (
    EnsembleExtractor_middle,
    EnsembleLoss_middle,
    EnsembleExtractor_middle_non_local,
    EnsembleLoss_middle_non_local
)
from .FOA_random_extractor import (
    EnsembleExtractor_random,
    EnsembleLoss_random,
    EnsembleExtractor_random_non_local,
    EnsembleLoss_random_non_local,
)

BACKBONE_MAP: dict[str, ClipExtractor] = {
    "B16": ClipB16Extractor,
    "B32": ClipB32Extractor,
    "Laion": LaionExtractor,
}