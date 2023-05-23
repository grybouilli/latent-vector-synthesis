# Latent Vector Synthesis

### Installation

#### Python

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

#### Pure Data

Install Pure Data (Pd-vanilla): [https://puredata.info](https://puredata.info/)

### Run

1 - Open a terminal and run the Python script:

```
python main.py
```

Wait until the osc-info appear (for listening and sending). 

2 - Open Pure Data and run latent-vector-synth.pd

3 - Make sure the right audio output device is selected (Go to Media —> Audio Settings…)

4 - In the upper left corner of the GUI, press RESET and then RANDOMIZE ALL.

5 - In the lower right corner of the GUI, make sure to toggle AUDIO OUT and raise the gain . Also, make sure the DSP toggle is on.

6 - Play!