# GenericMixture and the Component protocol

`GenericMixture(components=[...])` is a mixture of user-supplied components. A component is either an object that
implements the `Component` protocol (`log_probability` / `summarize` / `from_summaries`, the pomegranate 1.x API, so
`batch_size` works for free) or a `torch.nn.Module` with learnable parameters and a `log_prob(X)` method, whose M-step is
a generalised EM step (L-BFGS on the expected complete-data log-likelihood, accepted only on improvement).

```python
import torch
from tgmm import GenericMixture

class Exponential:                      # a protocol component: three methods, no base class
    def __init__(self, rate): self.rate, self.sw, self.sx = torch.tensor(rate), 0.0, 0.0
    def log_probability(self, X): return torch.log(self.rate) - self.rate * X[:, 0]
    def summarize(self, X, w): self.sw += w.sum(); self.sx += (w * X[:, 0]).sum()
    def from_summaries(self):
        self.rate = self.sw / self.sx; self.sw = self.sx = 0.0
    def sample(self, n): return torch.distributions.Exponential(self.rate).sample((n, 1))
    def n_parameters(self): return 1

mixture = GenericMixture([Exponential(0.5), Exponential(4.0)]).fit(X)
mixture.components_[0].rate, mixture.weights_
```

See the [design
note](https://github.com/adriansousapoza/tgmm/blob/main/docs/superpowers/specs/2026-10-15-v05-component-protocol-design.md)
for the contract, the decisions (no data-driven initialisation, `n_init=1`, which algorithms are supported) and the
limits (no SAEM or online EM: they blend additive statistics, which opaque `summarize` state is not).

::: tgmm.mixtures.generic.GenericMixture
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: tgmm.mixtures.protocol
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
