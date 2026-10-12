from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
P=Path(__file__).with_name('runner.py')
spec=importlib.util.spec_from_file_location('erasure_runner',P); assert spec and spec.loader
r=importlib.util.module_from_spec(spec); sys.modules[spec.name]=r; spec.loader.exec_module(r)

def test_three_way_folds():
    report=r.validate_folds(); assert set(report)=={'question_grouped','template_grouped'}
    for scheme in report:
        assert len(report[scheme])==5
        assert all((x['eraser_fit_n'],x['probe_fit_n'],x['test_n'])==(120,40,40) for x in report[scheme])

def test_role_rotation():
    for scheme in ('question_grouped','template_grouped'):
        ts=[]; ps=[]; count=np.zeros(100,int)
        for f in range(5):
            e,p,t,g=r.make_split_indices(scheme,f)
            assert set(e).isdisjoint(p) and set(e).isdisjoint(t) and set(p).isdisjoint(t)
            assert set(np.r_[e,p,t])==set(range(200)); assert len(np.unique(g))>=4
            ts += t[t<100].tolist(); ps += p[p<100].tolist(); count[e[e<100]] += 1
        assert sorted(ts)==list(range(100)); assert sorted(ps)==list(range(100)); assert np.all(count==3)

def test_synthetic_three_way():
    x=r.synthetic_verification(); assert x['eraser_fit_rows']==2000 and x['probe_fit_rows']==800 and x['test_rows']==800
    assert x['raw_balanced_accuracy']>=.95 and x['leace_balanced_accuracy']<=.60 and x['inlp_r4_balanced_accuracy']<=.60 and x['random_rowspace_r1_balanced_accuracy']>=.85

def test_leace_fit_properties():
    rng=np.random.default_rng(7); y=np.repeat([0,1],100); x=rng.normal(size=(200,24)); x[y==1,0]+=3
    fn,info=r.fit_leace_binary(x,y); z=fn(x)
    assert np.linalg.norm(z[y==1].mean(0)-z[y==0].mean(0))<1e-10
    assert abs(info['right']@info['left']-1)<1e-10
    assert np.linalg.norm(fn(z)-z)/np.linalg.norm(z)<1e-12

def test_conceptor_not_invertible():
    rng=np.random.default_rng(9); x=rng.normal(size=(80,32)); fn,info=r.fit_conceptor_not(x)
    assert 0 < info['min_eigenvalue'] <= info['max_eigenvalue'] < 1.0
    assert info['condition_number']>=1 and np.isfinite(fn(x)).all()

def test_inlp_projection():
    rng=np.random.default_rng(11); y=np.repeat([0,1],100); x=rng.normal(size=(200,20)); x[y==1,:3]+=1.5
    bases,fit=r.fit_inlp_bases(x,y,8); b=bases[8]
    assert fit['max_fit_iterations']>0
    assert np.max(np.abs(b@b.T-np.eye(len(b))))<1e-10
    z=r.apply_basis_projection(x,b); assert np.linalg.norm(r.apply_basis_projection(z,b)-z)/np.linalg.norm(z)<1e-10

def test_random_control_is_in_training_rowspace():
    rng=np.random.default_rng(13); x=rng.normal(size=(30,80))
    b=r.random_rowspace_basis(x,8,123)
    _,support=r.row_space_basis(x)
    residual=b-b@support@support.T
    assert np.linalg.norm(residual)<1e-10
    assert np.max(np.abs(b@b.T-np.eye(8)))<1e-10


def test_fast_metrics_match_sklearn_with_ties():
    y=np.asarray([0,1,0,1,0,1,0,1])
    p=np.asarray([.1,.9,.4,.4,.7,.8,.2,.7])
    m=r.fast_binary_metrics(y,p); pred=(p>=.5).astype(int)
    assert abs(m['log_loss']-log_loss(y,p,labels=[0,1]))<1e-14
    assert abs(m['balanced_accuracy']-balanced_accuracy_score(y,pred))<1e-14
    assert abs(m['roc_auc']-roc_auc_score(y,p))<1e-14
    assert abs(m['brier']-brier_score_loss(y,p))<1e-14
