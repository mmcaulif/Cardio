from .basic import (BasePolicy, 
                    WhitenoiseDeterministic, 
                    EpsilonArgmax, 
                    Gaussian, 
                    NoisyNaf,
                    Categorical,
                    Beta)

REGISTRY = {}

REGISTRY["random"] = BasePolicy
REGISTRY["whitenoise"] = WhitenoiseDeterministic
REGISTRY["pinknoise"] = None
REGISTRY["ounoise"] = None
REGISTRY["argmax"] = EpsilonArgmax
REGISTRY["gaussian"] = Gaussian
REGISTRY["naf"] = NoisyNaf
REGISTRY["categorical"] = Categorical
REGISTRY["beta"] = Beta
