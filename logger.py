import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

import wandb
class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)
            wandb.log({'training.loss': reduced_loss, 
                        'grad.norm': grad_norm,
                        'learning.rate': learning_rate,
                        'duration': duration
                    }, step=iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        wandb.log({'validation.loss': reduced_loss}, step=iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)

        alignment_arr = plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T)
        self.add_image(
            "alignment",
            alignment_arr,
            iteration)
        wandb.log({"alignment": [wandb.Image(alignment_arr, caption="Alignment")]}, step=iteration)

        mel_target = plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy())
        self.add_image(
            "mel_target",
            mel_target,
            iteration)
        wandb.log({"mel_target": [wandb.Image(mel_target, caption="Mel target")]}, step=iteration)
        
        mel_predicted = plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy())
        self.add_image(
            "mel_predicted",
            mel_predicted,
            iteration)
        wandb.log({"mel_predicted": [wandb.Image(mel_predicted, caption="Mel predicted")]}, step=iteration)

        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration)
