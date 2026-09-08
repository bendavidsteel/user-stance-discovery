"""jax configuration, imported before anything in the package creates an array.

float32 is not survivable in the smoother. The flag has to be set before the
first array exists, not merely before the first fit, and this package builds
quadrature nodes at import time -- so every module that touches jax imports
this one first, and the guarantee no longer depends on import order.

Importing jax does not initialise a backend; creating an array does. That is
why the pure-polars modules here import neither.
"""

import jax

jax.config.update('jax_enable_x64', True)
