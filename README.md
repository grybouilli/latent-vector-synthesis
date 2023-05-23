# Latent Vector Synthesis

Latent Vector Synthesis is a sound synthesis method combining latent audio spaces and vector synthesis techniques. 

This prototype of a Latent Vector Synthesizer incorporates  a Variational Autoencoder (VAE) model trained on short single cycle waveforms that enables interpolations and explorations of sonic textures. The generated waveforms are used as part of a vector/wavetable synthesis engine developed in Pure Data. 

This project was part of my Master's Thesis and builds on the work of Tatar et al. [1, 2]. 

## Installation

### Python

1 - Download and install Anaconda for your operating system: https://docs.anaconda.com/free/anaconda/install/index.html

2 - Open a terminal and create a new Python environment:

```
conda create --name lvs python=3.10
```

3 - Activate your environment:

```
conda activate lvs
```

4 - Install PyTorch using conda for your operating system: https://pytorch.org

5 - Install the following Python libraries: 

* Librosa

```
pip install librosa
```

* Python-osc

```
pip install python-osc
```

### Pure Data

Install Pure Data (Pd-vanilla): [https://puredata.info](https://puredata.info/)

## Run

1 - Open a terminal and navigate to the cloned repository.

2 - Run the Python script:

```
python main.py
```

Wait until the osc-infoformation appear (for sending/receiving osc-messages). 

3 - Run latent-vector-synth.pd in Pure Data.

* Make sure the right audio output device is selected (Go to Media —> Audio Settings…)
* Press RESET and then RANDOMIZE ALL. Make sure the DSP toggle is on.
* Make sure to toggle AUDIO OUT and raise the gain. 

4 - Happy droning!

## References

\[1\] Kıvanç Tatar, Daniel Bisig, and Philippe Pasquier. Latent Timbre Synthesis: Audio-based variational auto-encoders for music composition and sound design applications. *Neural Computing and Applications*, 33(1):67–84, 2021. URL: [https://link.springer.com/10.1007/s00521-020-05424-2](https://link.springer.com/10.1007/s00521-020-05424-2), doi:10.1007/s00521-020-05424-2

\[2\] Kıvanç Tatar, Kelsey Cotton, and Daniel Bisig. Sound design strategies for latent audio space explorations using deep learning architectures. In *Proceedings of Sound and Music Computing 2023*, 2023.

