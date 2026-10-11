<p align="center">
  <img src="assets/EXO_logo.png" alt="EXO project logo" width="250">
</p>

<h1 align="center">EXO</h1>

<p align="center">
  Tools for the study of exoplanets interior composition
</p>

# Description

EXO is a series of tools for the study of exoplanets and their host star. It combines constraints from interferometry, transit light curves, spectral energy distributions, velocimetry and parallaxes. 

As of version 0.1, EXO can compute likelihood and probability density functions for a series of stellar and planetary parameters (mass, radius, effective temperature, luminosity). It can derive environmental properties of exoplanets (habitable zone extent, equilibrium temperatures).
It can also directly adjust interferometric data from different instruments and dates simultaneously to derive the angular diameter of the star, relying on the [OIModeler](https://oimodeler.readthedocs.io/en/latest/) suite.

Future developments will focus on the possibility to adjust other data products beyond interferometry.

If you use EXO in your research, please cite:
> [Nowacki et al. (2026)](https://scixplorer.org/XXXX/XXXX)
> Nowacki H., Mourard D. et al.; Interferometric Survey of Stellar Parameters: Reliable methodology for the direct measurement of the angular diameter of exoplanet host stars, A&A, 2026.


## Installation

Install EXO directly from GitHub:

```bash
python -m pip install git+https://github.com/nowacki-hugo/EXO.git
```

Alternatively, clone the repository:

```bash
git clone https://github.com/nowacki-hugo/EXO.git
cd EXO
python -m pip install .
```

For development, use an editable installation:

```bash
python -m pip install -e .
```

## Quick start and examples

Use EXO as a standard python package 
```python
import EXO
```
See the `examples/` directory for working examples.

## Requirements

- Python >= 3.9
- OIModeler >= 0.9.8

## Features

- objects.py : Derive stellar and planetary properties from the analysis of different techniques
- interf.py : Modeling of optical interferometric observations, particularly data from the CHARA Array
- utils.py : Several useful constants and conversion functions
- funcs.py : Mathematical tools

## License

EXO is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.
