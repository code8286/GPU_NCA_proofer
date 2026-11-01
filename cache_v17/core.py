"""Retain numerical updates, codec/scorers and fixed competence criteria."""
from overfit_v15.core import training_step,reference_probe,competent,first_crossing


def validate_nested(train,common,original_probe):
    ids=lambda rows:{r['id'] for r in rows}
    if len(train)!=64 or len(common)!=16 or len(ids(train))!=64 or len(ids(common))!=16:
        raise ValueError('Require exactly 64/16 unique training/common examples')
    if not ids(common)<=ids(train)<=set(original_probe):
        raise ValueError('Preserve outcome-blind nested cache identities')
    for values,count in ((train,16),(common,4)):
        counts={(task,n):sum(r['task']==task and r['graph']['n']==n for r in values)
                for task in ('bfs_distances','dijkstra_distances') for n in (8,10)}
        if set(counts.values())!={count}:
            raise ValueError('Both tasks and sizes require balanced exact coverage')
