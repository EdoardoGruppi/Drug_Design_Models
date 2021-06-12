import rdkit
import torch
import torch.nn.utils as tnnu
import tqdm
import os
import re
import time
import models.dataset as reinvent_dataset
import models.vocabulary as reinvent_vocabulary
import utils.smiles as chem_smiles
from models.model import Model
from running_modes.configurations.transfer_learning.transfer_learning_configuration import TransferLearningConfiguration
from running_modes.transfer_learning.adaptive_learning_rate import AdaptiveLearningRate

rdkit.rdBase.DisableLog("rdApp.error")


class TransferLearningRunner:
    """Trains a given model."""

    def __init__(self, model: Model, config: TransferLearningConfiguration, adaptive_learning_rate: AdaptiveLearningRate):
        self._model = model
        self._adaptive_learning_rate = adaptive_learning_rate
        self._config = config
        path = self._config.output_model_path.replace('/' + self._config.output_model_path.split('/')[-1], '')
        self._config.starting_epoch, self.file_out = self.find_starting_epoch(path)

    def run(self):
        last_epoch = self._config.starting_epoch + self._config.num_epochs - 1

        print('{:<10} | {:<20} | {:<20} | {:<10} | {:<10} | {:<20}'.format('Epochs', 'Time elapsed', 'Loss', 'LR', 'Batch size', 'Dataset') + '\n' + '--' * 50)

        for epoch in range(self._config.starting_epoch, last_epoch + 1):
            if not self._adaptive_learning_rate.learning_rate_is_valid():
                break
            self._train_epoch(epoch, self._config.input_smiles_path)
        
        self.file_out.close()

        if self._config.save_every_n_epochs == 0 or (
                self._config.save_every_n_epochs != 1 and last_epoch % self._config.save_every_n_epochs > 0):
            self._save_model(last_epoch)
            self._adaptive_learning_rate.log_out_inputs()

    def _train_epoch(self, epoch, training_set_path):
        data_loader = self._initialize_dataloader(training_set_path)
        for _, batch in enumerate(self._progress_bar(data_loader, total=len(data_loader))):
            input_vectors = batch.long()
            loss = self._calculate_loss(input_vectors)

            self._adaptive_learning_rate.clear_gradient()
            loss.backward()
            if self._config.clip_gradient_norm > 0:
                tnnu.clip_grad_norm_(self._model.network.parameters(), self._config.clip_gradient_norm)
            self._adaptive_learning_rate.optimizer_step()

        hours, minutes, seconds = self.time_elapsed(self.t0)
        time_str = f"{hours:3d} h {minutes:2d} m {seconds:2d} s"
        print('{:<10} | {:20} | {:<20.8f} | {:<10.7f} | {:<10d} | {:<20}'.format(epoch, time_str, loss.item(), self._adaptive_learning_rate.get_lr(), self._config.batch_size, training_set_path.split("/")[-1]))
        self.file_out.write('{:<10} | {:20} | {:<20.8f} | {:<10.7f} | {:<10d} | {:<20}\n'.format(epoch, time_str, loss.item(), self._adaptive_learning_rate.get_lr(), self._config.batch_size, training_set_path.split("/")[-1]))
        self.file_out.flush()

        if self._config.save_every_n_epochs > 0 and epoch % self._config.save_every_n_epochs == 0:
            model_path = self._save_model(epoch)
            self._calculate_stats_and_update_learning_rate(epoch, model_path)

    def _progress_bar(self, iterable, total, **kwargs):
        return tqdm.tqdm(iterable=iterable, total=total, ascii=True, **kwargs)

    def _initialize_dataloader(self, path):
        training_set = chem_smiles.read_smiles_file(path, standardize=self._config.standardize, randomize=self._config.randomize)
        dataset = reinvent_dataset.Dataset(smiles_list=training_set, vocabulary=self._model.vocabulary,
                                           tokenizer=reinvent_vocabulary.SMILESTokenizer())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self._config.batch_size,
                                                 shuffle=self._config.shuffle_each_epoch,
                                                 collate_fn=reinvent_dataset.Dataset.collate_fn)
        return dataloader

    def _calculate_loss(self, input_vectors):
        log_p = self._model.likelihood(input_vectors)
        return log_p.mean()

    def _save_model(self, epoch):
        self._model.save(self._model_path(epoch))
        return self._model_path(epoch)

    def _model_path(self, epoch):
        # path = f"{self._config.output_model_path}.{epoch}" if epoch != self._config.num_epochs else f"{self._config.output_model_path}"
        path = f"{self._config.output_model_path}.{epoch}"
        return path

    def _calculate_stats_and_update_learning_rate(self, epoch, model_path):
        if self._config.collect_stats_frequency > 0 and epoch % self._config.collect_stats_frequency == 0:
            self._adaptive_learning_rate.collect_stats(epoch, model_path, self._config.input_smiles_path,
                                                       validation_set_path=self._config.validation_smiles_path)
        self._adaptive_learning_rate.update_lr_scheduler(epoch)

    @staticmethod
    def time_elapsed(start_time):
        elapsed = time.time() - start_time
        hours = int(elapsed / 3600)
        minutes = int(int(elapsed / 60) % 60)
        seconds = int(elapsed % 60)
        return hours, minutes, seconds

    def find_starting_epoch(self, path):
        model_name = self._config.input_model_path
        try:
            # load the last record values saved in the log.out file
            with open(os.path.join(path, 'log.out')) as f:
                records = f.readlines()
            # The records[-1] is the string: 'Finished Training!' unless an error occurred.
            final_record = records[-1]
            # Each line in the log.out file contains: the step number, the time spent (min), the loss and the lr values
            # Retrieve the last step number and time recorded.
            t_final = final_record.split('|')[1]
            t_final = [int(item) for item in re.findall('(.*?)h(.*?)m(.*?)s', t_final)[0]]
            t_final = t_final[0] * 3600 + t_final[1] * 60 + t_final[2]
            self.t0 = time.time() - t_final
            print(os.path.join(path, 'log.out'))
            out_file = open(os.path.join(path, 'log.out'), 'a')
            return int(model_name.split('.')[-1]) + 1, out_file
        except:
            self.t0 = time.time()
            out_file = open(os.path.join(path, 'log.out'), 'w')
            out_file.write('{:<10} | {:<20} | {:<20} | {:<10} | {:<10} | {:<20}'.format('Epochs', 'Time elapsed', 'Loss', 'LR', 'Batch size', 'Dataset') + '\n' + '--' * 50 + '\n')
            return 1, out_file

