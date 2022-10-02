# modwtpy
modwt in pytorch implementation

The wavelet is imported from pywt:
http://www.pybytes.com/pywavelets/ref/wavelets.html


```
input data shape: (B, L, D) 

wavelet coefficients shape: (B, D, L, level + 1) 

wavelet coefficients shape: (B, D, L, level + 1) 

B: batch size
D: time series data dimension
L: time series data length
level: number of wavelets

```

## Quick start

```
trans = WaveletTransform(len, level, 'db10')
w = trans.modwt(x)
rec = trans.imodwt(w)
mra, rec = trans.modwtmra(w)
```