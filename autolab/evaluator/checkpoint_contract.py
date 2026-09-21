"""Pure validation for the frozen checkpoint evaluator; no candidate code imports."""
import math
import stat
from pathlib import Path

CONFIG_KEYS={'sequence_len','vocab_size','n_layer','n_head','n_kv_head','n_embd','window_pattern','value_embedding_gates'}


def validate_config(config):
    if not isinstance(config,dict) or set(config)!=CONFIG_KEYS:
        raise ValueError('checkpoint must declare exactly the supported architecture fields')
    bounds={'sequence_len':(2048,2048),'vocab_size':(8192,8192),'n_layer':(1,12),
            'n_head':(1,16),'n_kv_head':(1,16),'n_embd':(128,1024)}
    for key,(low,high) in bounds.items():
        if type(config[key]) is not int or not low<=config[key]<=high:
            raise ValueError(f'unsupported {key}')
    if config['n_embd']%config['n_head'] or config['n_head']%config['n_kv_head'] or (config['n_embd']//config['n_head'])%2:
        raise ValueError('invalid attention dimensions')
    head_dim=config['n_embd']//config['n_head']
    if head_dim>256 or head_dim%8:
        raise ValueError('unsupported attention head dimension')
    pattern=config['window_pattern']
    if not isinstance(pattern,str) or not 1<=len(pattern)<=12 or set(pattern)-{'S','L'}:
        raise ValueError('unsupported attention window pattern')
    if type(config['value_embedding_gates']) is not bool:
        raise ValueError('value_embedding_gates must be boolean')
    return dict(config)


def artifact_file(directory,name,max_bytes):
    path=Path(directory)/name
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size>max_bytes:
        raise ValueError('checkpoint artifact must be a bounded regular file, not a symlink')
    return path


def assess(measured,reported,seconds):
    valid=lambda x:type(x) in (int,float) and math.isfinite(x)
    measured_ok=valid(measured) and measured>0
    reported_ok=valid(reported) and reported>0
    gates={'checkpoint_evaluated':measured_ok,
           'reported_bpb_matches':measured_ok and reported_ok and abs(measured-reported)<=0.0005,
           'reported_training_budget':valid(seconds) and 0<seconds<=330}
    return {'gates':gates,'metrics':{'val_bpb':measured} if measured_ok else {}}
