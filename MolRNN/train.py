# Import packages
import mxnet as mx
from mxnet import gluon, autograd, nd
import gc
import time
import os
import re
import numpy as np
import json
from mx_mg import models, data
from mx_mg.data import get_mol_spec


# Code for unconditional models
def engine(file_name='datasets/ChEMBL.txt', checkpoint_dir='checkpoint/mol_rnn', is_full=False, num_folds=5,
           fold_id=0, batch_size=50, batch_size_test=100, num_workers=0, k=5, p=0.8, embedding_size=16,
           hidden_sizes=(32, 64, 128, 128, 256, 256), skip_layer_size=256, dense_layer_sizes=(512,),
           policy_layers_size=128, activation='relu', rnn_layers=3, gpu_ids=(0, 1), lr=1e-3, decay=0.01,
           decay_step=100, clip_grad=3.0, iterations=30000, summary_step=200, steps_per_epoch=1000):
    """
    Prepares the data from the given file, train the described model and saves it into the chosen folder. The model
    created by this function is the unconditional molRnn architecture.

    :param file_name: location of the training dataset, default to datasets/ChEMBL.txt
    :param checkpoint_dir: location where the training results and model will be stored.
    :param is_full: train using the full dataset.
    :param num_folds: Specify the number of folds used in cross validation, default to 5.
    :param fold_id: Specify which fold is used as test set, default to 0.
    :param batch_size: mini-batch size for training, default to 50.
    :param batch_size_test: mini-batch size for evaluation during training, default to 100.
    :param num_workers: number of worker for data processing, default to 0 otherwise an error occurs.
    :param k: Number of decoding route used, default to 5
    :param p: Parameter controlling the randomness of importance sampling. Alpha in the literature.
    :param embedding_size: embedding size, i.e. size of the initial atom embedding, default to 16. In the source code
        F_e.
    :param hidden_sizes: hidden size for graph convolution layers, should be provided as a list. In the source code F_h.
    :param skip_layer_size: size of skip connection layer, default ot 256. In the source code F_s.
    :param dense_layer_sizes: the hidden sizes of fully connected layers after graph convolution, should be provided
        as a list. In the source code F_c.
    :param policy_layers_size:  hidden size for policy layer, default to 0.8. In the source code policy_layers_size.
    :param activation: the type of activation function used, default to relu. choices=['relu', 'tanh']
    :param rnn_layers:  number of layers used in GRUs, default to 3
    :param gpu_ids: GPUs used in the training, default to the first GPU [0, ].
    :param lr: the initial learning rate of Adam optimizer, default to 1e-3.
    :param decay: the rate of learning rate decay, default to 0.01.
    :param decay_step:  perform learning rate decay in every decay_step steps, default to 100
    :param clip_grad: parameter to clip the gradient values during their updates. default to 3.0
    :param iterations: number of iterations to perform during the training, default to 30,000 iterations
    :param summary_step: output performance metrics and model checkpoints for every summary-step steps, default to
        200 steps. Such a parameter could increase notably the execution time since it defines when the model is saved.
    :return:
    """
    # If a checkpoint is already saved put is_continuous as True
    is_continuous = all([os.path.isfile(os.path.join(checkpoint_dir, _n)) for _n in ['log.out', 'ckpt.params',
                                                                                     'trainer.status']])
    # Read every line of the given file and remove all the characters: '\n' and '\r'
    with open(file_name) as f:
        dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))

    if is_full:
        db_train = dataset
        # Get Sampler and Loader to prepare the dataset
        sampler_train = data.BalancedSampler(cost=[len(item) for item in db_train], batch_size=batch_size)
        # Loads data from a dataset and returns mini-batches of data using the sampler passed.
        loader_train = data.MolRNNLoader(db_train, batch_sampler=sampler_train, num_workers=num_workers, k=k, p=p)
        # Iterable object to get the batches of samples
        it_train = iter(loader_train)
        # If is_full is None there is no test dataset
        loader_test, it_test = None, None
    else:
        # Perform k-fold split of a given dataset
        db_train = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=True)
        # Select the fold_id fold for the testing phase
        db_test = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=False)
        # Get Sampler and Loader to prepare the training dataset
        sampler_train = data.BalancedSampler(cost=[len(item) for item in db_train], batch_size=batch_size)
        loader_train = data.MolRNNLoader(db_train, batch_sampler=sampler_train, num_workers=num_workers, k=k, p=p)
        # Get Sampler and Loader to prepare the test dataset
        sampler_test = data.BalancedSampler(cost=[len(item) for item in db_test], batch_size=batch_size_test)
        loader_test = data.MolRNNLoader(db_test, batch_sampler=sampler_test, num_workers=num_workers, k=k, p=p)
        # Iterable objects to get the batches of training and testing samples
        it_train, it_test = iter(loader_train), iter(loader_test)

    if not is_continuous:
        # If the model does not still exist, create a json file saving all the model architecture parameters
        configs = {'F_e': embedding_size, 'F_h': hidden_sizes, 'F_skip': skip_layer_size, 'F_c': dense_layer_sizes,
                   'Fh_policy': policy_layers_size, 'activation': activation, 'N_rnn': rnn_layers}
        # Open a json file and print the content. Create it if does not exist.
        with open(os.path.join(checkpoint_dir, 'configs.json'), 'w') as f:
            json.dump(configs, f)
    else:
        # If the model already exists, load the architecture parameters
        with open(os.path.join(checkpoint_dir, 'configs.json')) as f:
            configs = json.load(f)

    # Build the model
    model = models.VanillaMolGen_RNN(get_mol_spec().num_atom_types, get_mol_spec().num_bond_types, D=2, **configs)
    # List of all the available GPUs
    ctx = [mx.gpu(i) for i in gpu_ids]
    # Initialize or reinitialize the parameters depending on whether the model has been already created.
    if not is_continuous:
        model.collect_params().initialize(mx.init.Xavier(), force_reinit=True, ctx=ctx)
    else:
        model.load_parameters(os.path.join(checkpoint_dir, 'ckpt.params'), ctx=ctx)
    # Load optimizer
    opt = mx.optimizer.Adam(learning_rate=lr, clip_gradient=clip_grad)
    # Applies an Optimizer on a set of Parameters. Trainer should be used together with autograd.
    trainer = gluon.Trainer(params=model.collect_params(), optimizer=opt)
    # If the model has been already trained load the trainer states (e.g. optimizer, momentum) from a file.
    if is_continuous:
        trainer.load_states(os.path.join(checkpoint_dir, 'trainer.status'))

    # If the model is new start the counter from zero otherwise...
    if not is_continuous:
        t0 = time.time()
        global_counter = 0
        epoch = 0
    else:
        # load the last record values saved in the log.out file
        with open(os.path.join(checkpoint_dir, 'log.out')) as f:
            records = f.readlines()
            # The records[-1] is the string: 'Finished Training!' unless an error occurred.
            final_record = records[-1]
        # Each line in the log.out file contains: the step number, the time spent (min), the loss and the lr values
        # Retrieve the last step number and time recorded.
        count, t_final = int(final_record.split('|')[0]), str(final_record.split('|')[1])
        t_final = [int(item) for item in re.findall('(.*?)h(.*?)m(.*?)s', t_final)[0]]
        t_final = t_final[0] * 3600 + t_final[1] * 60 + t_final[2]
        # Code line to stat the time from the last saved value. t0 is expressed in seconds.
        t0 = time.time() - t_final
        # Code line to stat the step count from the last saved value.
        global_counter = count
        epoch = global_counter // steps_per_epoch

    # Open the log.out file in different modes according if it has been already created
    with open(os.path.join(checkpoint_dir, 'log.out'), mode='w' if not is_continuous else 'a') as f:
        if not is_continuous:
            # If not initialise the table with the name of the columns
            f.write('{:<10} | {:<20} | {:<20} | {:<10} | {:<10} | {:<20}'.format('Step', 'Time elapsed', 'Loss', 'LR', 'Batch size', 'Dataset') + '\n' + '--' * 50 + '\n')
        # Perform a training step until required
        print('{:<10} | {:<20} | {:<20} | {:<10} | {:<10} | {:<20}'.format('Step', 'Time elapsed', 'Loss', 'LR', 'Batch size', 'Dataset') + '\n' + '--' * 50)
        while global_counter < iterations:
            global_counter += 1
            # Load a batch of input data for each GPUs available
            try:
                inputs = [next(it_train) for _ in range(len(gpu_ids))]
            except StopIteration:
                it_train = iter(loader_train)
                inputs = [next(it_train) for _ in range(len(gpu_ids))]
            # Transform the data into tensors and pass them to the available GPUs
            inputs = [data.MolRNNLoader.from_numpy_to_tensor(input_i, j) for j, input_i in zip(gpu_ids, inputs)]
            # Returns an autograd recording scope context to be used in ???with??? statement and captures code that needs
            # gradients to be calculated.
            with autograd.record():
                # Compute the average loss computed for the n batch simultaneously processed.
                # The as_in_context returns an array on the target device with the same value as this array.
                loss = [(model(*input_i)).as_in_context(mx.gpu(gpu_ids[0])) for input_i in inputs]
                # Average loss
                loss = sum(loss) / len(gpu_ids)
                # Backward computation
                loss.backward()
            # Use nd.waitall to ensure that the async operations are completely executed in the time guard.
            nd.waitall()
            # Execute the garbage collector
            gc.collect()

            # Execution of a training step
            trainer.step(batch_size=1)
            # Decay the learning rate after n training steps
            if global_counter % decay_step == 0:
                trainer.set_learning_rate(trainer.learning_rate * (1.0 - decay))
            # Record an observation in the table only every m steps carried out
            if global_counter % summary_step == 0:
                # If the model is trained in the entire dataset the loss is that already computed
                if is_full:
                    loss = np.asscalar((sum(loss) / len(gpu_ids)).asnumpy())
                # In the opposite case compute the loss on a test batch
                else:
                    # Delete useless variables and collect the garbage
                    del loss, inputs
                    gc.collect()
                    # Load a batch of input test data for each GPUs available
                    try:
                        inputs = [next(it_test) for _ in range(len(gpu_ids))]
                    except StopIteration:
                        it_test = iter(loader_test)
                        inputs = [next(it_test) for _ in range(len(gpu_ids))]
                    # Returns a scope context to be used in ???with??? statement in which forward pass behavior is set to
                    # inference mode, without changing the recording states.
                    with autograd.predict_mode():
                        # Move the data converted into tensors into the GPUs
                        inputs = [data.MolRNNLoader.from_numpy_to_tensor(input_i, j) for j, input_i in
                                  zip(gpu_ids, inputs)]
                        # Compute the average loss on the batches simultaneously processed
                        loss = [(model(*input_i)).as_in_context(mx.gpu(gpu_ids[0])) for input_i in inputs]
                        loss = np.asscalar((sum(loss) / len(gpu_ids)).asnumpy())
                # Save the parameters and the states of the model
                model.save_parameters(os.path.join(checkpoint_dir, 'ckpt.params'))
                trainer.save_states(os.path.join(checkpoint_dir, 'trainer.status'))

                hours, minutes, seconds = time_elapsed(t0)
                time_str = f"{hours:3d} h {minutes:2d} m {seconds:2d} s"

                # Save a new observation in the log.out file
                f.write('{:<10} | {:20} | {:<20.8f} | {:<10.7f} | {:<10d} | {:<20}\n'.format(global_counter, time_str, loss, trainer.learning_rate, batch_size, file_name.split("/")[-1]))
                print('{:<10} | {:20} | {:<20.8f} | {:<10.7f} | {:<10d} | {:<20}'.format(global_counter, time_str, loss, trainer.learning_rate, batch_size, file_name.split("/")[-1]))
                # The flush() method clears the internal buffer of the file.
                f.flush()
                # Delete useless variables and collect the garbage
                del loss, inputs
                gc.collect()

            if global_counter % steps_per_epoch == 0:
                epoch += 1
                model.save_parameters(os.path.join(checkpoint_dir, f'ckpt_{epoch}epochs.params'))
                trainer.save_states(os.path.join(checkpoint_dir, f'trainer_{epoch}epochs.status'))

        # Save model parameters and state before exiting
        model.save_parameters(os.path.join(checkpoint_dir, 'ckpt.params'))
        trainer.save_states(os.path.join(checkpoint_dir, 'trainer.status'))


# Code for conditional models
def engine_cond(cond_type='scaffold', file_name='datasets/ChEMBL_scaffold.txt', num_scaffolds=734, is_full=False,
                checkpoint_dir='ckpt/scaffold', num_folds=5, fold_id=0, batch_size=50, batch_size_test=100,
                num_workers=0, k=5, p=0.8, embedding_size=16, hidden_sizes=(32, 64, 128, 128, 256, 256),
                skip_layer_size=256, dense_layer_sizes=(512,), policy_layers_size=128, activation='relu', rnn_layers=3,
                gpu_ids=(0, 1, 2, 3), lr=1e-3, decay=0.015, decay_step=100, clip_grad=3.0, iterations=30000,
                summary_step=200, steps_per_epoch=1000, cond_code_length=2):
    """
    Prepares the data from the given file, train the described model and saves it into the chosen folder. The model
    created by this function is the conditional molRnn architecture.

    :param cond_type: Train 'scaffold' or 'prop' or 'kinase' based conditional generator. default='scaffold'.
    :param file_name: Location of the training dataset, default to datasets/ChEMBL_scaffold.txt. With cond_type equal
        to prop or kinase select datasets/ChEMBL_prop.txt and datasets/ChEMBL_k.txt respectively.
    :param num_scaffolds: default_to 734.
    :param is_full: train using the full dataset.
    :param checkpoint_dir: location where the training results and model will be stored.
    :param num_folds: Specify the number of folds used in cross validation, default to 5.
    :param fold_id: Specify which fold is used as test set, default to 0.
    :param batch_size: mini-batch size for training, default to 50.
    :param batch_size_test: mini-batch size for evaluation during training, default to 100.
    :param num_workers: number of worker for data processing, default to 0 otherwise an error occurs.
    :param k: Number of decoding route used, default to 5
    :param p: Parameter controlling the randomness of importance sampling. Alpha in the literature.
    :param embedding_size: embedding size, i.e. size of the initial atom embedding, default to 16. In the source code
        F_e.
    :param hidden_sizes: hidden size for graph convolution layers, should be provided as a list. In the source code F_h.
    :param skip_layer_size: size of skip connection layer, default ot 256. In the source code F_s.
    :param dense_layer_sizes: the hidden sizes of fully connected layers after graph convolution, should be provided
        as a list. In the source code F_c.
    :param policy_layers_size:  hidden size for policy layer, default to 0.8. In the source code policy_layers_size.
    :param activation: the type of activation function used, default to relu. choices=['relu', 'tanh']
    :param rnn_layers:  number of layers used in GRUs, default to 3
    :param gpu_ids: GPUs used in the training, default to the first GPU [0, ].
    :param lr: the initial learning rate of Adam optimizer, default to 1e-3.
    :param decay: the rate of learning rate decay, default to 0.01.
    :param decay_step:  perform learning rate decay in every decay_step steps, default to 100
    :param clip_grad: parameter to clip the gradient values during their updates. default to 3.0
    :param iterations: number of iterations to perform during the training, default to 30,000 iterations
    :param summary_step: output performance metrics and model checkpoints for every summary-step steps, default to
        200 steps. Such a parameter could increase notably the execution time since it defines when the model is saved.
    :return:
    """
    # If a checkpoint is already saved put is_continuous as True
    is_continuous = all([os.path.isfile(os.path.join(checkpoint_dir, _n)) for _n in ['log.out', 'ckpt.params',
                                                                                     'trainer.status']])

    if is_full:
        if cond_type != 'kinase':
            if cond_type == 'scaffold':
                # Create the scaffold conditional code as a list of booleans. True is encountered when the scaffold
                # is present in the related molecule
                cond = data.SparseFP(num_scaffolds)
                N_C = num_scaffolds
            elif cond_type == 'prop':
                # Create the property conditional code as a list of property values
                cond = data.Delimited()
                N_C = cond_code_length
            else:
                raise ValueError
            # Read every line of the given file and remove all the characters: '\n' and '\r'
            with open(file_name) as f:
                dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))
            # get sampler and loader for training set
            sampler_train = data.BalancedSampler(cost=[len(l.split('\t')[0]) for l in dataset], batch_size=batch_size)
            loader_train = data.CMolRNNLoader(dataset, batch_sampler=sampler_train, num_workers=num_workers,
                                              k=k, p=p, conditional=cond)
            loader_test = []
        else:
            # Create the kinase conditional code as a list of float values.
            cond = data.Delimited()
            N_C = cond_code_length
            # Read every line of the given file and remove all the characters: '\n' and '\r'
            with open(file_name) as f:
                dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))

            # Filter a line setting the last element as the value passed.
            def _filter(_line, _i):
                return int(_line.split('\t')[-1]) == _i

            db_train = data.Lambda(data.Filter(dataset, fn=lambda _x: not _filter(_x, fold_id)), fn=lambda _x: _x[:-2])
            db_test = data.Lambda(data.Filter(dataset, fn=lambda _x: _filter(_x, fold_id)), fn=lambda _x: _x[:-2])

            # Get Sampler and Loader to prepare the training dataset
            loader_train = data.CMolRNNLoader(db_train, shuffle=True, num_workers=num_workers, k=k, p=p,
                                              conditional=cond, batch_size=batch_size)
            # Get Sampler and Loader to prepare the test dataset
            loader_test = data.CMolRNNLoader(db_test, shuffle=True, num_workers=num_workers, k=k, p=p, conditional=cond,
                                             batch_size=batch_size_test)
        # Iterable objects to get the batches of samples
        it_train, it_test = iter(loader_train), iter(loader_test)
    else:
        if cond_type != 'kinase':
            if cond_type == 'scaffold':
                # Create the scaffold conditional code as a list of booleans. True is encountered when the scaffold
                # is present in the related molecule
                cond = data.SparseFP(num_scaffolds)
                N_C = num_scaffolds
            elif cond_type == 'prop':
                # Create the property conditional code as a list of property values
                cond = data.Delimited()
                N_C = cond_code_length
            else:
                raise ValueError
            # Read every line of the given file and remove all the characters: '\n' and '\r'
            with open(file_name) as f:
                dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))
            # Perform k-fold split of a given dataset
            db_train = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=True)
            # Select the fold_id fold for the testing phase
            db_test = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=False)
            # Get Sampler and Loader to prepare the training dataset
            sampler_train = data.BalancedSampler(cost=[len(l.split('\t')[0]) for l in db_train], batch_size=batch_size)
            loader_train = data.CMolRNNLoader(db_train, batch_sampler=sampler_train, num_workers=num_workers,
                                              k=k, p=p, conditional=cond)
            # Get Sampler and Loader to prepare the test dataset
            sampler_test = data.BalancedSampler(cost=[len(l.split('\t'[0])) for l in db_test],
                                                batch_size=batch_size_test)
            loader_test = data.CMolRNNLoader(db_test, batch_sampler=sampler_test, num_workers=num_workers,
                                             k=k, p=p, conditional=cond)
        else:
            # Create the kinase conditional code as a list of float values.
            cond = data.Delimited()
            N_C = cond_code_length
            # Read every line of the given file and remove all the characters: '\n' and '\r'
            with open(file_name) as f:
                dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))

            # Filter each element observing if the last element corresond to the fold_id
            def _filter(_line, _i):
                return int(_line.split('\t')[-1]) == _i

            dataset = data.Filter(dataset, fn=lambda _x: _filter(_x, 1))
            print(dataset[0:20])
            db_train = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=True)
            db_test = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=False)
#            db_train = data.Lambda(data.Filter(dataset, fn=lambda _x: not _filter(_x, fold_id)), fn=lambda _x: _x[:-2])
#            db_test = data.Lambda(data.Filter(dataset, fn=lambda _x: _filter(_x, fold_id)), fn=lambda _x: _x[:-2])
            
            # Get Sampler and Loader to prepare the training dataset
            loader_train = data.CMolRNNLoader(db_train, shuffle=True, num_workers=num_workers, k=k, p=p,
                                              conditional=cond, batch_size=batch_size)
            # Get Sampler and Loader to prepare the test dataset
            loader_test = data.CMolRNNLoader(db_test, shuffle=True, num_workers=num_workers, k=k, p=p, conditional=cond,
                                             batch_size=batch_size_test)
        # Iterable objects to get the batches of training and testing samples
        it_train, it_test = iter(loader_train), iter(loader_test)

    if not is_continuous:
        # If the model does not still exist, create a json file saving all the model architecture parameters
        configs = {'N_C': N_C, 'F_e': embedding_size, 'F_h': hidden_sizes, 'F_skip': skip_layer_size,
                   'F_c': dense_layer_sizes, 'Fh_policy': policy_layers_size, 'activation': activation, 'rename': True,
                   'N_rnn': rnn_layers}
        # Open a json file and print the content. Create it if does not exist.
        with open(os.path.join(checkpoint_dir, 'configs.json'), 'w') as f:
            json.dump(configs, f)
    else:
        # If the model already exists, load the architecture parameters
        with open(os.path.join(checkpoint_dir, 'configs.json')) as f:
            configs = json.load(f)

    # Build the model
    model = models.CVanillaMolGen_RNN(get_mol_spec().num_atom_types, get_mol_spec().num_bond_types, D=2, **configs)
    # List of all the available GPUs
    ctx = [mx.gpu(i) for i in gpu_ids]
    # Initialize or reinitialize the parameters depending on whether the model has been already created.
    model.collect_params().initialize(mx.init.Xavier(), force_reinit=True, ctx=ctx)
    if is_continuous:
        model.load_parameters(os.path.join(checkpoint_dir, 'ckpt.params'), ctx=ctx)
    # Load optimizer
    opt = mx.optimizer.Adam(learning_rate=lr, clip_gradient=clip_grad)
    # Applies an Optimizer on a set of Parameters. Trainer should be used together with autograd.
    trainer = gluon.Trainer(model.collect_params(), opt)
    # If the model has been already trained load the trainer states (e.g. optimizer, momentum) from a file.
    if is_continuous:
        trainer.load_states(os.path.join(checkpoint_dir, 'trainer.status'))

    # If the model is new start the counter from zero otherwise...
    if not is_continuous:
        t0 = time.time()
        global_counter = 0
        epoch = 0
    else:
        # load the last record values saved in the log.out file
        with open(os.path.join(checkpoint_dir, 'log.out')) as f:
            records = f.readlines()
            # The records[-1] is the string: 'Finished Training!' unless an error occurred.
            final_record = records[-1]
        # Each line in the log.out file contains: the step number, the time spent (min), the loss and the lr values
        # Retrieve the last step number and time recorded.
        count, t_final = int(final_record.split('|')[0]), str(final_record.split('|')[1])
        t_final = [int(item) for item in re.findall('(.*?)h(.*?)m(.*?)s', t_final)[0]]
        t_final = t_final[0] * 3600 + t_final[1] * 60 + t_final[2]
        t0 = time.time() - t_final
        # Code line to stat the step count from the last saved value.
        global_counter = count
        epoch = global_counter // steps_per_epoch

    # Open the log.out file in different modes according if it has been already created
    with open(os.path.join(checkpoint_dir, 'log.out'), mode='w' if not is_continuous else 'a') as f:
        if not is_continuous:
            # If not initialise the table with the name of the columns
            f.write('{:<10} | {:<20} | {:<20} | {:<10} | {:<10} | {:<20}'.format('Step', 'Time elapsed', 'Loss', 'LR', 'Batch size', 'Dataset') + '\n' + '--' * 50 + '\n')
        # Perform a training step until required
        print('{:<10} | {:<20} | {:<20} | {:<10} | {:<10} | {:<20}'.format('Step', 'Time elapsed', 'Loss', 'LR', 'Batch size', 'Dataset') + '\n' + '--' * 50)
        
        while global_counter < iterations:
            global_counter += 1
            # Load a batch of input data for each GPUs available
            try:
                inputs = [next(it_train) for _ in range(len(gpu_ids))]
            except StopIteration:
                it_train = iter(loader_train)
                inputs = [next(it_train) for _ in range(len(gpu_ids))]
            # Transform the data into tensors and pass them to the available GPUs
            inputs = [data.CMolRNNLoader.from_numpy_to_tensor(input_i, j) for j, input_i in zip(gpu_ids, inputs)]
            # Returns an autograd recording scope context to be used in ???with??? statement and captures code that needs
            # gradients to be calculated.
            with autograd.record():
                # Compute the average loss computed for the n batch simultaneously processed.
                # The as_in_context returns an array on the target device with the same value as this array.
                loss = [(model(*input_i)).as_in_context(mx.gpu(gpu_ids[0])) for input_i in inputs]
                # Average loss
                loss = sum(loss) / len(gpu_ids)
                # Backward computation
                loss.backward()
            # Use nd.waitall to ensure that the async operations are completely executed in the time guard.
            nd.waitall()
            # Execute the garbage collector
            gc.collect()

            # Execution of a training step
            trainer.step(batch_size=1)
            # Decay the learning rate after n training steps
            if global_counter % decay_step == 0:
                trainer.set_learning_rate(trainer.learning_rate * (1.0 - decay))
            # Record an observation in the table only every m steps carried out
            if global_counter % summary_step == 0:
                # If the model is trained in the entire dataset the loss is that already computed
                if is_full:
                    loss = np.asscalar((sum(loss) / len(gpu_ids)).asnumpy())
                # In the opposite case compute the loss on a test batch
                else:
                    # Delete useless variables and collect the garbage
                    del loss, inputs
                    gc.collect()
                    # Load a batch of input test data for each GPUs available
                    try:
                        inputs = [next(it_test) for _ in range(len(gpu_ids))]
                    except StopIteration:
                        it_test = iter(loader_test)
                        inputs = [next(it_test) for _ in range(len(gpu_ids))]
                    # Returns a scope context to be used in ???with??? statement in which forward pass behavior is set to
                    # inference mode, without changing the recording states.
                    with autograd.predict_mode():
                        # Move the data converted into tensors into the GPUs
                        inputs = [data.CMolRNNLoader.from_numpy_to_tensor(input_i, j) for j, input_i in
                                  zip(gpu_ids, inputs)]
                        # Compute the average loss on the batches simultaneously processed
                        loss = [(model(*input_i)).as_in_context(mx.gpu(gpu_ids[0])) for input_i in inputs]
                        loss = np.asscalar((sum(loss) / len(gpu_ids)).asnumpy())
                # Save the parameters and the states of the model
                model.save_parameters(os.path.join(checkpoint_dir, 'ckpt.params'))
                trainer.save_states(os.path.join(checkpoint_dir, 'trainer.status'))
                
                hours, minutes, seconds = time_elapsed(t0)
                time_str = f"{hours:3d} h {minutes:2d} m {seconds:2d} s"

                # Save a new observation in the log.out file
                f.write('{:<10} | {:20} | {:<20.8f} | {:<10.7f} | {:<10d} | {:<20}\n'.format(global_counter, time_str, loss, trainer.learning_rate, batch_size, file_name.split("/")[-1]))
                print('{:<10} | {:20} | {:<20.8f} | {:<10.7f} | {:<10d} | {:<20}'.format(global_counter, time_str, loss, trainer.learning_rate, batch_size, file_name.split("/")[-1]))
                # The flush() method clears the internal buffer of the file.
                f.flush()
                # Delete useless variables and collect the garbage
                del loss, inputs
                gc.collect()

            if global_counter % steps_per_epoch == 0:
                epoch += 1
                model.save_parameters(os.path.join(checkpoint_dir, f'ckpt_{epoch}epochs.params'))
                trainer.save_states(os.path.join(checkpoint_dir, f'trainer_{epoch}epochs.status'))

        # Save model parameters and state before exiting
        model.save_parameters(os.path.join(checkpoint_dir, 'ckpt.params'))
        trainer.save_states(os.path.join(checkpoint_dir, 'trainer.status'))


def time_elapsed(start_time):
  elapsed = time.time() - start_time
  hours = int(elapsed / 3600)
  minutes = int(int(elapsed / 60) % 60)
  seconds = int(elapsed % 60)
  return hours, minutes, seconds
