from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, argparse, time, random
from pathlib import Path
import numpy as np
import librosa
import pdb
import configparser
from pythonosc import dispatcher
from pythonosc import osc_server
from typing import List, Any
from pythonosc.udp_client import SimpleUDPClient
import warnings
import sounddevice as sd
import json

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from rawvae.model import VAE, loss_function
from rawvae.dataset import AudioDataset, ToTensor

sampling_rate = 44100
sample_size = 600
f0 = sampling_rate / sample_size

audio_fold = Path(r'/Users/david/Documents/Datasets/Audio/AKWF/audio')
audio_files = [f for f in audio_fold.glob('*.wav')]

waveforms = np.zeros((4, sample_size)).astype('float32')

x = 0
y = 0

latent_space = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

segment_length = 600
hop_length = 600
n_units = 2048
latent_dim = 256

mu = torch.zeros(4, latent_dim)
logvar = torch.zeros(4, latent_dim)
predictions = np.zeros((4, sample_size)).astype('float32')

model = VAE(segment_length, n_units, latent_dim).to(device)
checkpoint_path = Path(r'/Users/david/Documents/Datasets/Audio/AKWF/nospectral_workstation/run-001/model/checkpoints/ckpt_00990')
state = torch.load(checkpoint_path, map_location=torch.device(device))
model.load_state_dict(state['state_dict'])
model.eval()

#################### WAVEFORM ####################
def get_wave_random():
    path = audio_files[random.randint(0, len(audio_files) - 1)]
    wave, _ = librosa.load(path, sr=None)

    return wave

def get_prediction(wave):
    with torch.no_grad():
        latent_mu, latent_logvar  = model.encode(torch.tensor(wave))
        latent_z = model.reparameterize(latent_mu, latent_logvar)
        pred = model.decode(latent_z)

    return pred, latent_mu, latent_logvar

def set_wave_random(index):
    global waveforms

    waveforms[index, :] = get_wave_random()

    if latent_space:
        set_prediction(index)

def set_waveforms_random():
    for i in range(4):
        set_wave_random(i)

def set_prediction(index):
    global predictions, mu, logvar

    pred, latent_mu, latent_logvar = get_prediction(waveforms[index, :])

    mu[index, :] = latent_mu
    logvar[index, :] = latent_logvar
    predictions[index, :] = pred.numpy()

def set_predictions():
    for i in range(4):
        set_prediction(i)

def interpolate2d(x, y):
    v = np.array([(1-x)*(1-y), x*(1-y), (1-x)*y, x*y]).astype('float32')

    if latent_space:
        v = torch.tensor(v)
        inter_mu = torch.matmul(torch.t(mu), v)
        inter_logvar = torch.matmul(torch.t(logvar), v)

        with torch.no_grad():
            latent_z = model.reparameterize(inter_mu, inter_logvar)
            interp = model.decode(latent_z).numpy()
    else:
        interp = np.matmul(np.transpose(waveforms), v)

    return interp

#################### OSC ####################
def reset(address: str, *osc_arguments: List[Any]) -> None:
    global waveforms

    waveforms = np.zeros((4, sample_size)).astype('float32')
    client.send_message("/reset", get_wave_random().tolist())
    send_waveforms()

def randomize_all(address: str, *osc_arguments: List[Any]) -> None:
    set_waveforms_random()
    send_waveforms()

def set_random(address: str, *osc_arguments: List[Any]) -> None:
    index = int(osc_arguments[0])
    set_wave_random(index)
    send_wave(index)

def morph(address: str, *osc_arguments: List[Any]) -> None:
    global x, y
    x = osc_arguments[0]
    y = osc_arguments[1]

    output = interpolate2d(x, y)
    send_output(output)

def send_wave(index):
    if latent_space:
        client.send_message("/waveform/" + chr(index+65), predictions[index, :].tolist())
    else:
        client.send_message("/waveform/" + chr(index+65), waveforms[index, :].tolist())

    output = interpolate2d(x, y)
    send_output(output)

def send_output(output):
    client.send_message("/waveform/output", output.tolist())

def send_waveforms():
    for i in range(4):
        send_wave(i)

def toggle_latent(address: str, *osc_arguments: List[Any]) -> None:
    global latent_space

    latent_space = bool(osc_arguments[0])

    if latent_space:
        set_predictions()

    send_waveforms()

if __name__ == "__main__":
    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiveIP", default="127.0.0.1",
        help="The ip of the OSC server")

    parser.add_argument("--receivePORT", type=int, default=5010,
        help="The port the OSC server is sending")

    parser.add_argument("--sendIP", default="127.0.0.1",
        help="The ip of the OSC client")

    parser.add_argument("--sendPORT", type=int, default=5005,
        help="The port the OSC client is sending")

    parser.add_argument("--verbose", default=True,
        help="Should I report what is happening?")
    args = parser.parse_args()

    verbose = args.verbose


    #import audio configs
    #n_bins = int(config['audio'].getint(num_octaves) * config['audio'].getint(bins_per_octave))


    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/reset", reset)
    dispatcher.map("/randomize", randomize_all)
    dispatcher.map("/waveform", set_random)
    dispatcher.map("/morphing", morph)
    dispatcher.map("/latent", toggle_latent)

    server = osc_server.ThreadingOSCUDPServer(
      (args.receiveIP, args.receivePORT), dispatcher)

    print("Listening to {}".format(server.server_address))
    client = SimpleUDPClient(args.sendIP, args.sendPORT)  # Create client
    print("Sending to {}".format((args.sendIP, args.sendPORT)))

    audio_device_list = sd.query_devices()
    #Following is to generate the umenu in Max GUI
    client.send_message("/audio/devicelist", "clear")
    for i in range(len(audio_device_list)):
        client.send_message("/audio/devicelist", str(i)+' '+audio_device_list[i]['name'])

    print(str(audio_device_list))

    print('Current audio device: {}'.format(sd.default.device))
    client.send_message("/audio/io", '{} {}'.format(sd.default.device[0], sd.default.device[1]) )

    server.serve_forever()
