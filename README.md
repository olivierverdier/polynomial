# Simple Polynomial class in Python/numpy

## Usage

```python
from polynomial import *

p = Polynomial([2.,3.])
p(2.) # value of 2 + 3X at 2., i.e., 8

p2 = 2 + 3*X

p == p2 # True

p.zeros() # list of zeros
```

## Testing

Run:

```bash
python test_polynomial.py
```
