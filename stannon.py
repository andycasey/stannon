
from pystan import StanModel



model = stan.StanModel(file="cannon.stan")


model.optimizing(data=None)