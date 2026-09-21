import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).parents[1]


def contract():
    spec=importlib.util.spec_from_file_location('contract',ROOT/'autolab/evaluator/checkpoint_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def config():
    return dict(sequence_len=2048,vocab_size=8192,n_layer=4,n_head=2,n_kv_head=2,n_embd=256,window_pattern='SSSL',value_embedding_gates=True)


def test_config_accepts_explicit_frozen_architecture():
    assert contract().validate_config(config())==config()


@pytest.mark.parametrize('key,value',[('n_layer',10000),('vocab_size',20),('n_embd',257),('n_head',0),('window_pattern',''),('value_embedding_gates','false'),('n_layer',True)])
def test_config_rejects_unsafe_or_unsupported_values(key,value):
    data=config();data[key]=value
    with pytest.raises(ValueError):contract().validate_config(data)


def test_fabricated_bpb_cannot_become_score():
    result=contract().assess(1.2,0.01,300.0)
    assert result['metrics']['val_bpb']==1.2
    assert result['gates']['reported_bpb_matches'] is False


@pytest.mark.parametrize('measured,reported,seconds',[(float('nan'),1.,300.),(1.,float('nan'),300.),(1.,1.,float('nan')),(1.,1.,-1.),(1.,1.,500.)])
def test_invalid_measurements_fail_closed(measured,reported,seconds):
    assert not all(contract().assess(measured,reported,seconds)['gates'].values())


def test_checkpoint_cannot_escape_artifacts(tmp_path):
    outside=tmp_path/'secret';outside.write_text('x')
    artifacts=tmp_path/'artifacts';artifacts.mkdir()
    (artifacts/'checkpoint.pt').symlink_to(outside)
    with pytest.raises(ValueError,match='regular|symlink'):
        contract().artifact_file(artifacts,'checkpoint.pt',100)


@pytest.mark.parametrize('width,heads',[(1024,1),(540,5)])
def test_attention_dimensions_fit_flash_kernel(width,heads):
    data=config();data.update(n_embd=width,n_head=heads,n_kv_head=heads)
    with pytest.raises(ValueError,match='head dimension'):
        contract().validate_config(data)


def test_scoring_dependency_is_pinned():
    import hashlib
    expected=(ROOT/'autolab/evaluator/prepare.sha256').read_text().strip()
    assert hashlib.sha256((ROOT/'prepare.py').read_bytes()).hexdigest()==expected


def test_changed_scoring_dependency_fails_before_import(tmp_path):
    import sys
    sys.path.insert(0,str(ROOT/'autolab/evaluator'))
    try:
        import evaluate
        (tmp_path/'prepare.py').write_text('raise AssertionError("must never import me")')
        with pytest.raises(ValueError,match='prepare.py changed'):
            evaluate.verify_scoring_dependency(tmp_path)
    finally:
        sys.path.pop(0)
