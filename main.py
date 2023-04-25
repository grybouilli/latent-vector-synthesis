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

sampling_rate = 44100
sample_size = 600
f0 = sampling_rate / sample_size

audio_fold = Path(r'/Users/david/Documents/Datasets/Audio/AKWF/audio')
audio_files = [f for f in audio_fold.glob('*.wav')]

waveforms = np.zeros((4, sample_size))

x, y = 0

def get_random_wave(audio_files):
    path = audio_files[random.randint(0, len(audio_files) - 1)]
    wave, _ = librosa.load(path, sr=None)
    return wave

def morph2d(x, y):
    v = np.array([(1-x)*(1-y), x*(1-y), (1-x)*y, x*y])
    return np.matmul(np.transpose(waveforms), v)
    # waveforms[0,:]*v[0] + waveforms[1,:]*v[1] + waveforms[2,:]*v[2] + waveforms[3,:]*v[3]


def initialize(address: str, *osc_arguments: List[Any]) -> None:
    client.send_message("/init", get_random_wave(audio_files).tolist())
    randomize("/randomize")

def randomize(address: str, *osc_arguments: List[Any]) -> None:
    global waveforms

    for i in range(4):
        wave = get_random_wave(audio_files)
        waveforms[i, :] = wave
        client.send_message("/waveform/" + chr(i+65), wave.tolist())

def get_wave(address: str, *osc_arguments: List[Any]) -> None:
    global waveforms
    wave = get_random_wave(audio_files)
    index = int(osc_arguments[0])
    waveforms[index, :] = wave

    client.send_message("/waveform/" + chr(index+65), wave.tolist())

def morph(address: str, *osc_arguments: List[Any]) -> None:
    global x, y
    x = osc_arguments[0]
    y = osc_arguments[1]

    wave = morph2d(x, y)

    client.send_message("/waveform/output", wave.tolist())

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
    dispatcher.map("/initialize", initialize)
    dispatcher.map("/randomize", randomize)
    dispatcher.map("/waveform", get_wave)
    dispatcher.map("/morphing", morph)

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
