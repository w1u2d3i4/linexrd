"""Conditional expert tree with 13 label-free inference features."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

C = 230


def softmax(z):
    e=np.exp(z-z.max(1,keepdims=True));return e/e.sum(1,keepdims=True)


def normalize(p):
    p=np.asarray(p,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=C or not len(p) or not np.isfinite(p).all() or (p<0).any() or (p.sum(1)<=0).any():
        raise ValueError('Expected finite nonempty (N,230) probabilities')
    return p/p.sum(1,keepdims=True)


def validate_inputs(payload,k):
    # Intentionally reads ONLY the three inference arrays.
    p=normalize(payload['p']);s=np.asarray(payload['sim'],dtype=np.float64);ny=np.asarray(payload['neighbor_y'])
    if s.ndim!=2 or len(s)!=len(p) or s.shape[1]<k or ny.shape!=s.shape or not np.isfinite(s).all():
        raise ValueError('Invalid aligned similarities/neighbor labels or too few neighbors')
    if not np.issubdtype(ny.dtype,np.integer) or ((ny<0)|(ny>=C)).any():raise ValueError('Bad neighbor labels')
    if (np.diff(s,axis=1)>1e-6).any():raise ValueError('Provide descending clean verified cosines and aligned labels')
    if ((s < -1.00001)|(s > 1.00001)).any():raise ValueError('Expected cosine similarities in [-1,1]')
    return dict(p=p,sim=s[:,:k],neighbor_y=ny[:,:k].astype(np.int64))


def labels(payload,n):
    y=np.asarray(payload['y'])
    if y.shape!=(n,) or not np.issubdtype(y.dtype,np.integer) or ((y<0)|(y>=C)).any():raise ValueError('Invalid SG labels')
    return y.astype(np.int64)


def knn_vote(d,temp):
    weights=softmax(d['sim']/temp);v=np.zeros_like(d['p'])
    np.add.at(v,(np.arange(len(v))[:,None],d['neighbor_y']),weights)
    return v


def tree_prob(model,x):
    if model['estimator'] is None:return np.full(len(x),model['constant'])
    return model['estimator'].predict_proba(x[:,:model['feature_count']])[:,1]


def mix_tree(model,d,v,win_prob):
    # win_prob estimates kNN winning conditional on informative disagreement.
    alpha=win_prob.copy()
    if model['mapping']!='soft':alpha=(win_prob>=float(model['mapping'])).astype(float)
    same=d['p'].argmax(1)==v.argmax(1);alpha[same]=1.
    threshold=model['benefit_threshold']
    if threshold is not None:alpha[(1-2*win_prob)<=threshold]=1.
    return (1-alpha[:,None])*d['p']+alpha[:,None]*v,alpha


def features(d,v):
    p=d['p'];s=d['sim'];k=d.get('active_k',s.shape[1]);s=s[:,:k]
    def stats(q):
        top=np.sort(q,axis=1)[:,-2:]
        return [top[:,1],top[:,1]-top[:,0],-(q*np.log(np.maximum(q,1e-12))).sum(1)/np.log(C)]
    pc=p.argmax(1);vc=v.argmax(1);row=np.arange(len(p))
    columns=stats(p)+stats(v)+[(pc==vc).astype(float),s.max(1),s.mean(1),s.std(1)]
    columns += [s[:,0]-s[:,1] if k>1 else np.zeros(len(s)),v[row,pc],p[row,vc]]
    return np.column_stack(columns)


def fit(payload, k=4, temperature=0.03, seed=0, depth=3, iterations=200,
        mapping="hard", benefit_threshold=0.1):
    """Fit one configured tree on labeled clean calibration examples.

    The target is a kNN win among disagreements with exactly one correct
    expert. No labels are read by predict(). No candidate search is performed.
    """
    if k < 1 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Require positive k and temperature")
    if mapping not in {"hard", "soft"}:
        raise ValueError("mapping must be hard or soft")
    if benefit_threshold is not None and not np.isfinite(benefit_threshold):
        raise ValueError("benefit threshold must be finite or None")
    d = validate_inputs(payload, k)
    y = labels(payload, len(d['p']))
    v = knn_vote(d, temperature)
    pc, vc = d['p'].argmax(1), v.argmax(1)
    keep = (pc != vc) & ((pc == y) | (vc == y))
    target = (vc[keep] == y[keep]).astype(int)
    estimator, constant = None, None
    if len(np.unique(target)) < 2:
        constant = float(target[0]) if len(target) else 0.5
    else:
        estimator = HistGradientBoostingClassifier(
            max_depth=depth, max_iter=iterations, learning_rate=0.05,
            l2_regularization=1.0, min_samples_leaf=20,
            early_stopping=False, random_state=seed,
        ).fit(features(d, v)[keep], target)
    return dict(K=k, temperature=temperature, estimator=estimator,
                constant=constant, feature_count=13,
                mapping="soft" if mapping == "soft" else 0.5,
                benefit_threshold=benefit_threshold)


def predict(model, payload):
    """Return fused probabilities, kNN probabilities, and retrieval weights."""
    d = validate_inputs(payload, model['K'])
    v = knn_vote(d, model['temperature'])
    r = tree_prob(model, features(d, v))
    q, alpha = mix_tree(model, d, v, r)
    return q, v, alpha
