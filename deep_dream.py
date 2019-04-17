import os
from functools import partial
from typing import Iterable, Tuple, Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
import tqdm
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from skorch.net import NeuralNet

import constants
from dataset import ExtractStft
from model import Classifier
from pytorch_extensions import roll


class DataLoader(TransformerMixin, BaseEstimator):
    def fit(self, x, y, **fit_params):
        return self

    @classmethod
    def transform(cls, x: Iterable[str]) -> Iterable[Tuple[np.ndarray, int]]:
        output = []
        for file_name in x:
            signal, sample_rate = librosa.load(file_name, sr=constants.LIBRISPEECH_SAMPLE_RATE)
            output.append((signal, sample_rate))
        return output


class DataPreprocessor(TransformerMixin, BaseEstimator):
    def fit(self, x, y, **fit_params):
        return self

    @classmethod
    def transform(cls, x: Iterable[Tuple[np.ndarray, int]]) -> Iterable[Tuple[np.ndarray, int]]:
        output = []
        for signal, sample_rate in x:
            stacked = ExtractStft.get_stft(signal)
            output.append((stacked, sample_rate))
        return output


class Model(TransformerMixin, BaseEstimator):
    def __init__(self,
                 model_path: str,
                 block_name: str,
                 number_of_iterations: int,
                 framing_window_step: int,
                 optimisation_step_size: float,
                 use_gpu: bool = False,
                 batch_size: int = 8,
                 jitter: int = 3,
                 verbose: bool = False,
                 seed: Optional[int] = None):
        self._model_path = model_path
        self._framing_window_step = framing_window_step
        self._use_gpu = use_gpu
        self._block_name = block_name
        self._verbose = verbose
        self._number_of_iterations = number_of_iterations
        self._np_rng = np.random.RandomState(seed)
        self._jitter = jitter
        self._batch_size = batch_size
        self._optimisation_step_size = optimisation_step_size

        self._available_layers = {}
        self._available_layers_names = []

        self._classifier = Classifier(constants.NUMBER_OF_CLASSES)
        self._net = NeuralNet(
            self._classifier, nn.CrossEntropyLoss
        )
        self._net.initialize()
        self._net.load_params(f_params=self._model_path)

        for layer_name, layer in self._classifier.layers_blocks.items():
            if "residual" in layer_name:
                current_register = partial(self._register_layer_output, layer_name=layer_name)
                layer.register_forward_hook(current_register)
                self._available_layers_names.append(layer_name)

        if self._verbose:
            print(f"Available layer names: \n{self._available_layers_names}")

    def _register_layer_output(self, module, input_, output, layer_name):
        self._available_layers[layer_name] = output

    def fit(self, x, y, **fit_params):
        return self

    def transform(self, x: Iterable[Tuple[np.ndarray, int]]) -> Iterable[Tuple[np.ndarray, int]]:
        output = []
        for stft, sample_rate in x:
            prediction = self._transform_single_normal_deep_dream(stft)
            output.append((prediction, sample_rate))
        return output

    def _transform_single_normal_deep_dream(self, stft: np.ndarray) -> np.ndarray:
        stft = torch.from_numpy(stft).contiguous().float()
        width = stft.size(1)
        casted_width = int(width / constants.STFT_CROP_WIDTH) * constants.STFT_CROP_WIDTH

        stft = stft[:, :casted_width]
        # stft = torch.zeros_like(stft)
        # stft.normal_(0, 0.01)
        stft = stft.permute(2, 0, 1)
        if self._use_gpu:
            stft = stft.cuda()
        octaves = [stft]

        detail = torch.zeros_like(octaves[-1]).contiguous().float()
        if self._use_gpu:
            detail = detail.cuda()

        output_data = stft
        for octave, octave_base in enumerate(octaves[::-1]):
            if self._verbose:
                print(f"Trying octave: {octave + 1} / {len(octaves)}")
            length = octave_base.size(-1)
            if octave > 0:
                rescaled_detail = nd.zoom(detail, length / len(detail), order=2)
                detail = torch.from_numpy(rescaled_detail).contiguous().float()
            output_data = octave_base + detail
            for _ in tqdm.tqdm(range(self._number_of_iterations), total=self._number_of_iterations,
                               disable=not self._verbose):
                output_data = self._make_step(output_data)
            detail = output_data - octave_base
        plt.imshow(output_data[0])
        plt.show()
        plt.imshow(detail[0])
        plt.show()
        output_data = output_data.permute(1, 2, 0)
        return output_data.numpy()

    def _make_step(self, stft: torch.Tensor) -> torch.Tensor:
        random_shift = self._np_rng.randint(-self._jitter, self._jitter + 1)
        stft = roll(stft, random_shift, 2)

        current_min, current_max = stft.min(), stft.max()
        frames = self._generate_frames(stft)

        grads = torch.zeros_like(frames)
        self._classifier.eval()

        for i in range(0, len(frames), self._batch_size):
            start_index = i
            end_index = min(i + self._batch_size, len(frames))
            batch = frames[start_index:end_index]
            batch.requires_grad = True
            if self._use_gpu:
                batch = batch.cuda()

            self._classifier(batch)

            layer_output = self._available_layers[self._block_name][:, 0:2]
            objective_output = self._objective(layer_output)
            objective_output.backward()

            batch_grad = batch.grad.detach().clone()
            batch.grad.zero_()
            grads[start_index:end_index] = batch_grad

        full_grad = self._restore_stft_shape(grads, stft)
        stft += self._optimisation_step_size / torch.mean(torch.abs(full_grad)) * full_grad
        stft = torch.clamp(stft, current_min, current_max)
        stft = roll(stft, -random_shift, 2)
        return stft

    def calc_grad_tiled(self, stft: torch.Tensor, tile_size: int = 512) -> torch.Tensor:
        h, w = stft.shape[1:]
        sx, sy = self._np_rng.randint(tile_size, size=2)
        stft_shift = roll(roll(stft, sx, axis=2), sy, axis=1)
        grads = torch.zeros_like(stft)
        for y in range(0, max(h - tile_size // 2, tile_size), tile_size):
            for x in range(0, max(w - tile_size // 2, tile_size), tile_size):
                frame = stft_shift[:, y:y + tile_size, x:x + tile_size]
                frame = frame.expand(1, -1)
                self._classifier(frame)

                layer_output = self._available_layers[self._block_name][:, 0:2]
                objective_output = self._objective(layer_output)
                objective_output.backward()

                grad = frame[0].grad.detach().clone()
                grads[:, y:y + tile_size, x:x + tile_size] = grad
        return roll(roll(grads, -sx, axis=2), -sy, axis=1)

    @classmethod
    def _objective(cls, data: torch.Tensor) -> torch.Tensor:
        return data.mean()

    def _generate_frames(self, stft: torch.Tensor) -> torch.Tensor:
        # split signals into chunks
        number_of_frames = int((stft.size(2) - constants.STFT_CROP_WIDTH) / self._framing_window_step + 1)
        frames = torch.zeros(
            (number_of_frames, 2, constants.LIBRISPEECH_COMPONENTS, constants.STFT_CROP_WIDTH)).float().contiguous()

        if self._use_gpu:
            frames = frames.cuda()

        beginning_of_sample = 0
        end_of_sample = constants.STFT_CROP_WIDTH
        frame_count = 0

        while end_of_sample < stft.size(2):
            frames[frame_count] = stft[:, :, beginning_of_sample:end_of_sample]
            beginning_of_sample = beginning_of_sample + self._framing_window_step
            end_of_sample = beginning_of_sample + constants.STFT_CROP_WIDTH
            frame_count = frame_count + 1

        return frames

    def _restore_stft_shape(self, frames: torch.Tensor, original_signal: torch.Tensor) -> torch.Tensor:
        restored_signal = torch.zeros((2, original_signal.size(1), len(frames) * constants.STFT_CROP_WIDTH),
                                      dtype=original_signal.dtype)
        if self._use_gpu:
            restored_signal = restored_signal.cuda()
        offset = 0
        previous_ending_offset = np.inf
        for frame in frames:
            input_slice = slice(offset, offset + constants.STFT_CROP_WIDTH)
            restored_signal[:, :, input_slice] = frame

            if previous_ending_offset < offset:
                overlap_slice = slice(offset, previous_ending_offset)
                restored_signal[:, :, overlap_slice] /= 2

            previous_ending_offset = input_slice.stop
            offset += self._framing_window_step
        return restored_signal


class Denormalize(TransformerMixin, BaseEstimator):
    def fit(self, x, y=None, **fit_params):
        return self

    @classmethod
    def transform(cls, x: Iterable[Tuple[np.ndarray, int]]) -> Iterable[Tuple[np.ndarray, int]]:
        output = []
        for stft, fs in x:
            mag, phase = np.split(stft, 2, axis=-1)
            mag, phase = mag[..., 0], phase[..., 0]

            mag = np.expm1(mag)
            phase *= np.pi
            real = mag * np.cos(phase)
            imag = mag * np.sin(phase)

            stft = real + 1j * imag
            unfouriered = librosa.istft(stft, win_length=constants.LIBRISPEECH_WINDOW_SIZE)
            output.append((unfouriered, fs))
        return output


class SaveResult(TransformerMixin, BaseEstimator):
    def __init__(self, output_dir: str, base_name: str):
        self.output_dir = output_dir
        self.base_name = base_name

    def fit(self, x, y=None, **fit_params):
        return self

    def transform(self, x: Iterable[Tuple[np.ndarray, int]]) -> Iterable[Tuple[np.ndarray, int]]:
        output = []
        for i, (signal, fs) in enumerate(x):
            path = os.path.join(self.output_dir, self.base_name + "_{}.wav".format(i))
            librosa.output.write_wav(path, signal, fs)
            output.append((signal, fs))
        return output


def get_processing_pipeline(model_path: str) -> Pipeline:
    return Pipeline([
        ("data load", DataLoader()),
        ("data processor", DataPreprocessor()),
        ("deep dream", Model(
            block_name="residual_4b",
            model_path=model_path,
            number_of_iterations=20,
            optimisation_step_size=0.5,
            framing_window_step=constants.STFT_CROP_WIDTH,
            jitter=8,
            verbose=True,
            use_gpu=False,
        )),
        ("denormalize", Denormalize()),
        ("data saver", SaveResult(".", "audio"))
    ])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model")
    parser.add_argument("file_path", help="Path to .wav file to process")
    args = parser.parse_args()

    pipe = get_processing_pipeline(args.model_path)
    print(pipe.transform([args.file_path]))


if __name__ == '__main__':
    main()
