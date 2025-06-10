# Isotropic Metal Electromagnetic Skin Effect Analysis

A Python toolkit that calculates the frequency-dependent surface impedance of a metal, using the free electron model, across a wide range of transport regimes.

## Overview

This project showcases a tool I put together as part of my work towards my experimental physics PhD, and I am uploading it as a project as part of my professional portfolio.

NOTE: I am very busy with my dissertation writing, but I wanted to showcase some of my work. It works well on my local machine, but I had to make some modificitations for upload: I can't upload some of the functions yet, as I am discussing with my collaborator how we wish to create and distribute a broader package). So at the moment it doesn't quite work, but I included some example figures and data to show what it produces. When I get a chance, I'll go and fix the remaining issues from the adjusted code. As well, I'm sure there are ways to improve it, so I am open to suggestions!

To interpret the experimental results I collected for my PhD dissertation, I needed to be able to calculate the surface impedance spectra for materials using Fermi surface meshes (for an introduction to the Fermi surface in metals and it's properties, see Chapter 2 of the textbook by Ashcroft and Mermin). I developed comprehensive tools to generate these meshes and calculate surface impedance spectra from them, but needed an analytic response against which to verify them and characterize their numerical errors. That is why I created this project.

The Jupyter notebook which allows a user to calculate the surface impedance $Z_s$ of an isotropic metal, that is, a metal with a spherical Fermi surface. Examples of metals where this approximation is valid are Sodium and the other alkali metals [7]. It is also useful in understanding the response of other metals, if the appropriate approximations and Fermi surface quantities are used.

The tool calculates the surface impedance in the following regimes:
- Classical skin effect regime
- Anomalous skin effect regime  
- Relaxation regime
- Anomalous reflection regime
Note: It does not include the response in the transmission regime that occurs above the plasma frequency.

Fully diffuse (random orientation) surface scattering is used as it is experimentally relevant.

While I created many tools and modules to put this together, I owe much to many other works:

Dr. Graham Baker's PhD Dissertation provides a great overview of all of the relevant skin effect regimes, calculations, and different skin effect regime limits [1]. Wooten's text is also very valuable for this [2]. The first rigorous theories for the anomalous skin effect in metals with spherical and ellipsoidal Fermi surfaces were produced by Reuter and Sondheimer [3,4]. A more recent paper, expressing the same theory in a way friendly to experimentalists, was produced by Hein et al. [5]. Pippard's textbook on conduction phenomena has served as a valuable reference [6]. Ashcroft and Mermin, the famous condensed matter physics textbook, tabulates many of the electronic transport properties for metals where the free electron model is valid [7]. Finally, the theory work by Valentinis, Baker et al. dives yet deeper into the skin effect response [8].

I did my best to carefully reference the equations in the individual functions where they appear, but in case I missed some, the reference list should contain everything.

AI Disclosure: I've been using AI tools heavily to develop my software development and data science skillset. I use Cursor (cursor.sh) as my code editor, and have used various ChatGPT and Claude models extensively for the programming and code documentation of this project. For the most part, I treat the AI as a colleague who I delegate programming tasks to, and try to learn all I can from the code it generates for me. I strive to write additional comments and documentation when needed. This document is written by myself with the help of AI for formatting and structure. I avoid using AI for writing directly (although I admit it did add some content when it made this template, so I went over and rewrote most of it), instead using it for discussion and advice. I've found writing to be a valuable tool for myself in learning and thinking.

## Features

- [x] Calculate the asymptotic limits of the surface impedance across the specified skin effect regimes
- [x] Calculate the analytic surface impedance for a free electron metal as a function of frequency
- [x] Compare the analytic surface impedance response with the asymptotic regime limits for verification
- [x] Allow options to compute the response for an arbitrary free electron metal, by accepting different Fermi wavevectors, Fermi velocities, and effective masses
- [x] Implement a default metal for testing
- [x] Implement the properties of Sodium (from Ashcroft [7])

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd isotropic-metal-electromagnetic-skin-effect-analysis

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

[Usage examples - to be filled]

## Documentation

See the `docs/` directory for detailed documentation:
- [Theory Background](docs/theory_background.md)
- [Methodology](docs/methodology.md)
- [API Reference](docs/api_reference.md)

## Project Structure

├── notebooks/              # Analysis notebook
├── src/                    # Core library modules
├── data/                   # Sample datasets
    ├── model-parameters/   # Isotropic model parameters to use for the calculation.
    └── results/            # Results
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── figures/                # Generated plots

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- Jupyter Notebook
- [Additional requirements - to be filled]

## References

1. Baker, G. (2022). Non-Local Electrical Conductivity in PdCoO$_{2}$ [PhD thesis]. University of British Columbia. http://hdl.handle.net/2429/82849

2. Wooten, F. (1972). Optical Properties of Solids. Academic Press.

3. Sondheimer, E. H. (1954). The theory of the anomalous skin effect in anisotropic metals. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences, 224(1157), 260-271. https://doi.org/10.1098/rspa.1954.0157

4. Reuter, G. E. H., & Sondheimer, E. H. (1948). The theory of the anomalous skin effect in metals. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences, 195(1042), 336-364. https://doi.org/10.1098/rspa.1948.0123

5. Hein, M. A., Ormeno, R. J., & Gough, C. E. (2001). High-frequency electrodynamic response of strongly anisotropic clean normal and superconducting metals. Physical Review B, 64(2), 024529. https://doi.org/10.1103/PhysRevB.64.024529

6. Pippard, A. B. (1965). The Dynamics of Conduction Electrons. Gordon and Breach.

7. Ashcroft, N. W., & Mermin, N. D. (1976). Solid State Physics. Saunders College Publishers.

8. Valentinis, D., Baker, G., Bonn, D. A., & Schmalian, J. (2023). Kinetic theory of the nonlocal electrodynamic response in anisotropic metals: Skin effect in 2D systems. Physical Review Research, 5(1), 013212. https://doi.org/10.1103/PhysRevResearch.5.013212

## License

[License - to be filled]
