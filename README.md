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

Check out the [associated notebook](https://gist.github.com/olivierverdier/d939d4e0b4de8fbd9c26328a834721ab).

## Testing

Run:

```bash
pytest test_polynomial.py
```
