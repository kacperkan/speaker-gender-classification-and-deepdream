# Audio Deep Dream

## Table of Contents

1. [Description](#desc)
2. [Findings](#findings)
3. [Requirements](#requirements)
4. [Prerequisites](#prerequisites)
5. [Usage](#usage)
6. [Things that I tried](#itried)
7. [License](#license)



## Description <div id='desc' />
The following repository contains code and models trained on LibriSpeech dataset
([link](http://www.openslr.org/12) - only 'dev-clean' was used due to computational restrictions) for gender classification using deep neural networks. Trained models were
used to 'dream' on audio samples (instead of images), as has been shown by
Google team ([link](https://github.com/google/deepdream) to original repository).

How does it work?:
1. Audio sample is converted to a spectrogram (power spectrum).
2. The spectrogram is scaled down up to x8 and the dreaming procedure is started:
    1. The sample is randomly roll shifted in X axis.
    2. Gradient of the input with respect to a chosen layer output is calculated
    3. Calculated gradient is added to aggregated matrix o gradients
    4. The image is scaled up by x2 and procedure repeats until the image is the
    of the original one
3. Aggregated gradients are added to the original image and are combined
with the original phase
4. The audio is restored using inverse FFT

The goal of the step 2. is to increase the activation of a particular
convolutional layer. By aggregating gradients and adding it to the input, we
increase features that the layer exhibits.

The trained model achieved ~80% of accuracy at gender classification on the LibriSpeech dataset.

## Findings <div id="findings" />

1. I needed to add more randomization in roll shifting due to fast filtering of the image
by gradient accumulation. Decreasing the learning rate was subtle. More drastic decrease 
caused that no changes were noticeable.
2. Every time the model was learnt, the dreaming caused skewed, black lines to
appear in the image. The effect of those lines was that, once inverted, the audio
resembled falling airplane (or at least, that sound that is often presented in the TV 😊).
3. Once tuned, almost every, early layer calculating gradients at almost any layer produced
MIDI-like sounds.
4. Deeper layers created noise only, no particular effects were noticed.

## Requirements <div id="requirements" />

- Anaconda (for python environment)
- Git LFS (for models)

## Prerequisites <div id="prerequisites" />

Install the environment using:
```bash
conda env create -f env.yaml 
```

## Usage <div id="usage" />
The model works quite decently fast when short sample is used. 
Be aware of that for longer samples, the RAM can crash.

```bash
> conda activate audiodeepdream
> python deep_dream.py \
    -f <audio_file_path> \
    -l <layer_name=residual_1a> \
    -n <number_of_scales=10> \
    -i <num_iterations_per_scale=10> \
    -d <output_dir> \
    -o <output_file_base_name>
```

Example usage which will dream on one of samples
from LibriSpeech dataset:
```bash
> python deep_dream.py \
    -f audio/1089-134686-0000.flac \
    -l residual_1a \
    -n 10 \
    -i 10 \
    -d example_result \
    -o audio
```
Above command will create a file `example_result/audio_0.wav`.

Available layers names:
```bash
'residual_1a', 'residual_2a', 'residual_2b', 'residual_3a', 'residual_3b', 'residual_4a', 'residual_4b', 'residual_5a', 'residual_5b'
```
Use `python deep_dream.py --help` to get more info.

Additionally, you can use jupyter notebook `custom.ipynb` to play with parameters and see how the spectrogram
changes over time.
## Things that I tried <div id="itried" />

- **Original parameters and the inception network from the original work** - I do not know how it would be supposed to work -
the model was trained on the image dataset so it does not apply to spectrograms. Original parameters caused huge instability
in the optimisation process
- **SincNet** ([arXiv](https://arxiv.org/abs/1808.00158)) - the model works on raw audio data. However, the model, once
learnt, produced noise only



## Acknowledgments
The work was heavily influenced by repository [bapoczos/deep-dream-tensorflow](https://github.com/bapoczos/deep-dream-tensorflow).
## License <div id="license" />

Copyright (c) 2019 Kacper Kania

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.