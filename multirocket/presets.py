"""Reviewable 30k presets, including an explicit legacy reproduction mode.

The recommended name is changed only after the corrected review's adoption
gates pass. The base MultiRocket constructor remains available for ablations.
"""
from multirocket.multirocket import MultiRocket

COMMON=dict(num_features=30000,memory_efficient=True,prediction_batch_size=256,
            optimized_transform=True,fused_prediction=True)
PRESETS={
    'previous30':{**COMMON,'classifier':'fast_ridge','first_difference_fix':True},
    'candidate30':{**COMMON,'classifier':'auto_ridge','first_difference_fix':True,
                   'ridge_feature_limit_bytes':693*1024**2},
    'legacy30':{**COMMON,'classifier':'fast_ridge','first_difference_fix':False},
}
RECOMMENDED_30K='previous30'


def make_multirocket(preset=None,random_state=None,**overrides):
    """Build the reviewed default, a fixed backend, or the exact legacy mode."""
    preset=RECOMMENDED_30K if preset is None else preset
    if preset not in PRESETS:
        raise ValueError('Unknown preset; choose '+', '.join(PRESETS))
    options={**PRESETS[preset],'random_state':random_state,**overrides}
    model=MultiRocket(**options)
    model.preset_name_=preset
    return model
