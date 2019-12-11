import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence


def plot_data(data, figsize=(16, 4), output_file='waves.png'):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                interpolation='none')
    plt.savefig('./logdir/' + output_file)

def save_mel(c_mel_postnet):
    mel_filename = './logdir/001.npy'
    np.save(mel_filename, c_mel_postnet, allow_pickle=False)

### MELS FROM wav files
mel_files = [
        './ljs_dataset_folder/mels/LJ001-0068.npy', 
        './ljs_dataset_folder/mels/LJ002-0219.npy',
        './ljs_dataset_folder/mels/LJ003-0166.npy'
]
np_mels = [np.load(mel_file) for mel_file in mel_files] 
for np_mel in np_mels:
    print('min {}, max {}'.format( np.amin(np_mel), np.amax(np_mel) ))
plot_data(np_mels, output_file='old_waves.png')

### MELS generated from text

hparams = create_hparams()

#checkpoint_path = "./outdir/checkpoint_12400"
checkpoint_path = './blizzard_outdir/checkpoint_28000'
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

text = "Mike Krieger is really awesomem"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

torch.save(mel_outputs_postnet, './logdir/001.pt')

c_mel_outputs = mel_outputs.float().data.cpu().numpy()[0]
c_mel_postnet = mel_outputs_postnet.float().data.cpu().numpy()[0]
c_alignments = alignments.float().data.cpu().numpy()[0].T

print('mel outputs {}'.format(c_mel_outputs.shape))
print('min {}, max {}'.format( np.amin(c_mel_outputs), np.amax(c_mel_outputs) ))

print('mel postnet {}'.format(c_mel_postnet.shape))
print('min {}, max {}'.format( np.amin(c_mel_postnet), np.amax(c_mel_postnet) ))


plot_data([c_mel_outputs, c_mel_postnet, c_alignments])

save_mel(c_mel_postnet)

#import sys
#sys.path.append('waveglow/')
#from denoiser import Denoiser
#waveglow_path = './waveglow/waveglow_256channels_ljs_v3.pt'
#waveglow = torch.load(waveglow_path)['model']
#waveglow.cuda().eval().half()
#for k in waveglow.convinv:
#    k.float()
#denoiser = Denoiser(waveglow)

#with torch.no_grad():
#    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
