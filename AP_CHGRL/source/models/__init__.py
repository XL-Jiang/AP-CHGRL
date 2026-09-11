from .AP_CHGRL import AP_CHGRL
from omegaconf import DictConfig


def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).cuda()#ComBrainTF(config)
