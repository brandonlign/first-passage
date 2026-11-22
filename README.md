# Sharp lower bounds on first-passage skewness and kurtosis in unicyclic networks

Brandon Li

Paper and verification code proving the lower-envelope conjectures of
Wampler and Barato ([J. Phys. A 55, 014002, 2022](https://doi.org/10.1088/1751-8121/ac3b0c)).

The entropy first-passage time of a unicyclic Markov network has skewness
and excess kurtosis, scaled by the threshold, denoted $S$ and $K$. For every
network with $n$ states and affinity $F>0$,

$$
S \ge \min_{1\le m\le n} f(F/m), \qquad K \ge \min_{1\le m\le n} g(F/m),
$$

where

$$
f(x)=2\sqrt{x}\frac{1+4e^x+e^{2x}}{\sqrt{(e^x-1)(1+e^x)^3}}, \qquad
g(x)=6x\frac{1+8e^x+e^{2x}}{e^{2x}-1}
$$

are the values for the homogeneous $m$-state cycle. Both bounds are sharp at
exactly $n$ states. The same result holds for every standardized cumulant of
order $k\ge3$, with the bound given by Legendre polynomials. Over all
networks, $S\ge3.909729\ldots$ and $K\ge25.09134\ldots$.

- [Paper (PDF)](paper/main.pdf)
- [LaTeX source and figure scripts](paper/)
- [Verification code](checks/)

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python checks/verify.py
```

`verify.py` checks the paper's formulas and inequalities numerically with
mpmath; its output is in [`checks/verify_output.txt`](checks/verify_output.txt).
The figures come from `paper/make_envelope.py`, `paper/make_random.py`,
`paper/make_bands.py` and `paper/ring.tex`. The PDF builds with Tectonic
(`cd paper && tectonic main.tex`) or `pdflatex`.

## License

Code: MIT. Paper and figures: CC BY 4.0.
