from .basic import (BasePolicy, 
                    Epsilon_Deterministic_policy, 
                    Epsilon_argmax_policy, 
                    Gaussian_policy, 
                    Noisy_naf_policy,
                    Categorical_policy,
                    Beta_policy)

REGISTRY = {}

REGISTRY["random"] = BasePolicy
REGISTRY["deterministic"] = Epsilon_Deterministic_policy
REGISTRY["argmax"] = Epsilon_argmax_policy
REGISTRY["gaussian"] = Gaussian_policy
REGISTRY["naf"] = Noisy_naf_policy
REGISTRY["categorical"] = Categorical_policy
REGISTRY["beta"] = Beta_policy
