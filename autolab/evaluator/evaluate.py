"""Independently score data-only checkpoints with frozen model and BPB code.

Never import train.py or execute a candidate-supplied model/metric implementation.
Training time is still self-reported; the harness separately bounds wall time.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from checkpoint_contract import artifact_file, assess, validate_config


def reject_nonfinite(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def verify_scoring_dependency(root):
    expected=(Path(__file__).resolve().parent/'prepare.sha256').read_text().strip()
    if sha256(root/'prepare.py')!=expected:
        raise ValueError('prepare.py changed; update evaluator version/hash and establish a new baseline')


def evaluate(artifact):
    import torch
    # Only the control checkout provides prepare/model code. The CLI uses -I.
    root=Path(__file__).resolve().parents[2]
    verify_scoring_dependency(root)
    sys.path.insert(0,str(root))
    from prepare import Tokenizer,evaluate_bpb
    from frozen_model import GPT,GPTConfig
    metadata=artifact_file(artifact,'checkpoint.json',16384)
    checkpoint=artifact_file(artifact,'checkpoint.pt',2_000_000_000)
    manifest=json.loads(metadata.read_text(),parse_constant=reject_nonfinite)
    if type(manifest.get('schema_version')) is not int or manifest['schema_version']!=1 or type(manifest.get('seed')) is not int or manifest['seed']!=int(os.environ['AUTOLAB_SEED']):
        raise ValueError('checkpoint schema/seed mismatch')
    config=validate_config(manifest.get('config'))
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not isinstance(weights,dict) or not weights or not all(isinstance(k,str) and type(v) is torch.Tensor for k,v in weights.items()):
        raise ValueError('checkpoint must contain only named tensor weights')
    if any(not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError('checkpoint weights must be finite')
    model=GPT(GPTConfig(**config))
    model.load_state_dict(weights,strict=True,assign=True)
    model=model.to('cuda').eval()
    # Training constructs these nonpersistent buffers on CUDA too.
    model.cos,model.sin=model._precompute_rotary_embeddings(
        model.rotary_seq_len,config['n_embd']//config['n_head'])
    tokenizer=Tokenizer.from_directory()
    if tokenizer.get_vocab_size()!=config['vocab_size']:
        raise ValueError('tokenizer vocabulary mismatch')
    with torch.inference_mode(),torch.amp.autocast(device_type='cuda',dtype=torch.bfloat16):
        measured=evaluate_bpb(model,tokenizer,8)
    result=assess(measured,manifest.get('reported_bpb'),manifest.get('reported_training_seconds'))
    result['checkpoint_sha256']=sha256(checkpoint)
    result['evaluation']={'implementation':'frozen-model@1','batch_size':8,'seed':manifest['seed'],
                          'reported_bpb':manifest.get('reported_bpb'),'config':config}
    return result


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    try:
        result=evaluate(Path(os.environ['AUTOLAB_ARTIFACT']))
    except Exception as exc:
        result={'gates':{'checkpoint_evaluated':False},'metrics':{},'error':f'{type(exc).__name__}: {exc}'}
    Path(os.environ['AUTOLAB_METRICS']).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
