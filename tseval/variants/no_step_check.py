from tseval import init_config as tseval_init_config, evaluate as tseval_evaluate
from tseval.metric import ERR_FACT_MISMATCH, ERR_STEP_MISSING, ERR_STEP_REVERSAL
from baselines.ref_checker import init_config as rc_init_config, evaluate as rc_evaluate

def init_config(model, args):
    tseval_init_config(model, args)
    rc_init_config(model, args)

def evaluate(*args, **kwargs):
    score, extra = tseval_evaluate(*args, **kwargs)
    errors = set([item[0] for item in extra['reason']]) - set([ERR_STEP_MISSING, ERR_STEP_REVERSAL])
    if not errors:
        return rc_evaluate(*args, **kwargs)
    return score, extra
