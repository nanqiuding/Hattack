from .base_trainer import BaseTrainer
from .dataset import ImageFolderDataset
from .FOA import FOA_trainer
from .FOA_global import FOA_global_trainer
from .dhra import DHRATrainer
from .m_attack import MAttackTrainer
from .FOA_inter_30_intra import FOA_inter_30_intra_trainer

TRAINER_MAP: dict[str, type[BaseTrainer]] = {
    "DHRA": DHRATrainer,
    "DHRA_NIPS1000": DHRATrainer,
    "M_ATTACK_NIPS100": MAttackTrainer,
    "M_ATTACK_NIPS1000": MAttackTrainer,
    "FOA_test": FOA_trainer,
    "FOA": FOA_trainer,
    "FOA_global": FOA_global_trainer,
    "FOA_inter-30-intra": FOA_inter_30_intra_trainer
}