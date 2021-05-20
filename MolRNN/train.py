import mxnet as mx
from mxnet import gluon, autograd, nd
import gc
import time
import os
import numpy as np
import json
from mx_mg import models, data
from mx_mg.data import get_mol_spec


# Code for unconditional models
def engine(file_name='datasets/ChEMBL.txt', checkpoint_dir='checkpoint/mol_rnn', is_full=False, num_folds=5,
           fold_id=0, batch_size=50, batch_size_test=100, num_workers=0, k=5, p=0.8, embedding_size=16,
           hidden_sizes=(32, 64, 128, 128, 256, 256), skip_layer_size=256, dense_layer_sizes=(512,),
           policy_layers_size=128, activation='relu', rnn_layers=3, gpu_ids=(0, 1), lr=1e-3, decay=0.01,
           decay_step=100, clip_grad=3.0, iterations=30000, summary_step=200):
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
    :param clip_grad: default to 3.0
    :param iterations: number of iterations to perform during the training, default to 30,000 iterations
    :param summary_step: output performance metrics and model checkpoints for every summary-step steps, default to
        200 steps
    :return:
    """
    # If a checkpoint is already saved put is_continuous as True
    if all([os.path.isfile(os.path.join(checkpoint_dir, _n)) for _n in ['log.out', 'ckpt.params', 'trainer.status']]):
        is_continuous = True
    else:
        is_continuous = False
    # Read every line of the given file and remove all the characters: '\n' and '\r'
    with open(file_name) as f:
        dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))

    if is_full:
        db_train = dataset
        # get sampler and loader for training set
        sampler_train = data.BalancedSampler(cost=[len(l) for l in db_train], batch_size=batch_size)
        loader_train = data.MolRNNLoader(db_train, batch_sampler=sampler_train, num_workers=num_workers, k=k, p=p)
        it_train = iter(loader_train)
        loader_test, it_test = None, None
    else:
        # get dataset
        db_train = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=True)
        db_test = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=False)
        # get sampler and loader for training set
        sampler_train = data.BalancedSampler(cost=[len(l) for l in db_train], batch_size=batch_size)
        loader_train = data.MolRNNLoader(db_train, batch_sampler=sampler_train, num_workers=num_workers, k=k, p=p)
        # get sampler and loader for test set
        sampler_test = data.BalancedSampler(cost=[len(l) for l in db_test], batch_size=batch_size_test)
        loader_test = data.MolRNNLoader(db_test, batch_sampler=sampler_test, num_workers=num_workers, k=k, p=p)
        # get iterator
        it_train, it_test = iter(loader_train), iter(loader_test)
    if not is_continuous:
        configs = {'embedding_size': embedding_size, 'hidden_sizes': hidden_sizes, 'skip_layer_size': skip_layer_size,
                   'dense_layer_sizes': dense_layer_sizes, 'policy_layers_size': policy_layers_size,
                   'activation': activation, 'rnn_layers': rnn_layers}
        with open(os.path.join(checkpoint_dir, 'configs.json'), 'w') as f:
            json.dump(configs, f)
    else:
        with open(os.path.join(checkpoint_dir, 'configs.json')) as f:
            configs = json.load(f)
    model = models.VanillaMolGen_RNN(get_mol_spec().num_atom_types, get_mol_spec().num_bond_types, D=2, **configs)

    ctx = [mx.gpu(i) for i in gpu_ids]
    if not is_continuous:
        model.collect_params().initialize(mx.init.Xavier(), force_reinit=True, ctx=ctx)
    else:
        model.load_parameters(os.path.join(checkpoint_dir, 'ckpt.params'), ctx=ctx)

    # construct optimizer
    opt = mx.optimizer.Adam(learning_rate=lr, clip_gradient=clip_grad)
    trainer = gluon.Trainer(model.collect_params(), opt)
    if is_continuous:
        trainer.load_states(os.path.join(checkpoint_dir, 'trainer.status'))

    if not is_continuous:
        t0 = time.time()
        global_counter = 0
    else:
        with open(os.path.join(checkpoint_dir, 'log.out')) as f:
            records = f.readlines()
            if records[-1] != 'Training finished\n':
                final_record = records[-1]
            else:
                final_record = records[-2]
        count, t_final = int(final_record.split('\t')[0]), float(final_record.split('\t')[1])
        t0 = time.time() - t_final * 60
        global_counter = count

    with open(os.path.join(checkpoint_dir, 'log.out'), mode='w' if not is_continuous else 'a') as f:
        if not is_continuous:
            f.write('step\ttime(h)\tloss\tlr\n')
        while True:
            global_counter += 1

            try:
                inputs = [next(it_train) for _ in range(len(gpu_ids))]
            except StopIteration:
                it_train = iter(loader_train)
                inputs = [next(it_train) for _ in range(len(gpu_ids))]

            # move to gpu
            inputs = [data.MolRNNLoader.from_numpy_to_tensor(input_i, j) for j, input_i in zip(gpu_ids, inputs)]

            with autograd.record():
                loss = [(model(*input_i)).as_in_context(mx.gpu(gpu_ids[0])) for input_i in inputs]
                loss = sum(loss) / len(gpu_ids)
                loss.backward()

            nd.waitall()
            gc.collect()

            trainer.step(batch_size=1)
            if global_counter % decay_step == 0:
                trainer.set_learning_rate(trainer.learning_rate * (1.0 - decay))

            if global_counter % summary_step == 0:
                if is_full:
                    loss = np.asscalar((sum(loss) / len(gpu_ids)).asnumpy())
                else:
                    del loss, inputs
                    gc.collect()

                    try:
                        inputs = [next(it_test) for _ in range(len(gpu_ids))]
                    except StopIteration:
                        it_test = iter(loader_test)
                        inputs = [next(it_test) for _ in range(len(gpu_ids))]

                    with autograd.predict_mode():
                        # move to gpu
                        inputs = [data.MolRNNLoader.from_numpy_to_tensor(input_i, j) for j, input_i in
                                  zip(gpu_ids, inputs)]
                        loss = [(model(*input_i)).as_in_context(mx.gpu(gpu_ids[0])) for input_i in inputs]
                        loss = np.asscalar((sum(loss) / len(gpu_ids)).asnumpy())

                model.save_parameters(os.path.join(checkpoint_dir, 'ckpt.params'))
                trainer.save_states(os.path.join(checkpoint_dir, 'trainer.status'))

                f.write('{}\t{}\t{}\t{}\n'.format(global_counter, float(time.time() - t0) / 60, loss,
                                                  trainer.learning_rate))
                print('{}\t{}\t{}\t{}\n'.format(global_counter, float(time.time() - t0) / 60, loss,
                                                trainer.learning_rate))
                f.flush()
                del loss, inputs
                gc.collect()

            if global_counter >= iterations:
                break

        # save before exit
        model.save_params(os.path.join(checkpoint_dir, 'ckpt.params'))
        trainer.save_states(os.path.join(checkpoint_dir, 'trainer.status'))
        f.write('Training finished\n')


# Code for conditional models
def engine_cond(cond_type='scaffold', file_name='datasets/ChEMBL_scaffold.txt', num_scaffolds=734, is_full=False,
                checkpoint_dir='ckpt/scaffold', num_folds=5, fold_id=0, batch_size=50, batch_size_test=100,
                num_workers=0, k=5, p=0.8, embedding_size=16, hidden_sizes=(32, 64, 128, 128, 256, 256),
                skip_layer_size=256, dense_layer_sizes=(512,), policy_layers_size=128, activation='relu', rnn_layers=3,
                gpu_ids=(0, 1, 2, 3), lr=1e-3, decay=0.015, decay_step=100, clip_grad=3.0, iterations=30000,
                summary_step=200):
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
    :param clip_grad: default to 3.0
    :param iterations: number of iterations to perform during the training, default to 30,000 iterations
    :param summary_step: output performance metrics and model checkpoints for every summary-step steps, default to
        200 steps
    :return:
    """
    if all([os.path.isfile(os.path.join(checkpoint_dir, _n)) for _n in ['log.out', 'ckpt.params', 'trainer.status']]):
        is_continuous = True
    else:
        is_continuous = False

    if is_full:
        if cond_type != 'kinase':
            if cond_type == 'scaffold':
                cond = data.SparseFP(num_scaffolds)
                N_C = num_scaffolds
            elif cond_type == 'prop':
                cond = data.Delimited()
                N_C = 2
            else:
                raise ValueError

            with open(file_name) as f:
                dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))

            # get sampler and loader for training set
            sampler_train = data.BalancedSampler(cost=[len(l.split('\t')[0]) for l in dataset], batch_size=batch_size)
            loader_train = data.CMolRNNLoader(dataset, batch_sampler=sampler_train, num_workers=num_workers,
                                              k=k, p=p, conditional=cond)

            loader_test = []
        else:
            cond = data.Delimited()
            N_C = 2

            if all([os.path.isfile(os.path.join(checkpoint_dir, _n)) for _n in
                    ['log.out', 'ckpt.params', 'trainer.status']]):
                is_continuous = True
            else:
                is_continuous = False

            with open(file_name) as f:
                dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))

            # get dataset
            def _filter(_line, _i):
                return int(_line.split('\t')[-1]) == _i

            db_train = data.Lambda(data.Filter(dataset,
                                               fn=lambda _x: not _filter(_x, fold_id)),
                                   fn=lambda _x: _x[:-2])
            db_test = data.Lambda(data.Filter(dataset,
                                              fn=lambda _x: _filter(_x, fold_id)),
                                  fn=lambda _x: _x[:-2])

            # get sampler and loader for test set
            loader_test = data.CMolRNNLoader(db_test, shuffle=True, num_workers=num_workers,
                                             k=k, p=p, conditional=cond, batch_size=batch_size_test)

            # get sampler and loader for training set
            loader_train = data.CMolRNNLoader(db_train, shuffle=True, num_workers=num_workers,
                                              k=k, p=p, conditional=cond, batch_size=batch_size)

        # get iterator
        it_train, it_test = iter(loader_train), iter(loader_test)
    else:
        if cond_type != 'kinase':
            if cond_type == 'scaffold':
                cond = data.SparseFP(num_scaffolds)
                N_C = num_scaffolds
            elif cond_type == 'prop':
                cond = data.Delimited()
                N_C = 2
            else:
                raise ValueError

            if all([os.path.isfile(os.path.join(checkpoint_dir, _n)) for _n in
                    ['log.out', 'ckpt.params', 'trainer.status']]):
                is_continuous = True
            else:
                is_continuous = False

            with open(file_name) as f:
                dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))

            # get dataset
            db_train = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=True)
            db_test = data.KFold(dataset, k=num_folds, fold_id=fold_id, is_train=False)

            # get sampler and loader for training set
            sampler_train = data.BalancedSampler(cost=[len(l.split('\t')[0]) for l in db_train], batch_size=batch_size)
            loader_train = data.CMolRNNLoader(db_train, batch_sampler=sampler_train, num_workers=num_workers,
                                              k=k, p=p, conditional=cond)

            # get sampler and loader for test set
            sampler_test = data.BalancedSampler(cost=[len(l.split('\t'[0])) for l in db_test],
                                                batch_size=batch_size_test)
            loader_test = data.CMolRNNLoader(db_test, batch_sampler=sampler_test, num_workers=num_workers,
                                             k=k, p=p, conditional=cond)

        else:
            cond = data.Delimited()
            N_C = 2

            if all([os.path.isfile(os.path.join(checkpoint_dir, _n)) for _n in
                    ['log.out', 'ckpt.params', 'trainer.status']]):
                is_continuous = True
            else:
                is_continuous = False

            with open(file_name) as f:
                dataset = data.Lambda(f.readlines(), lambda _x: _x.strip('\n').strip('\r'))

            # get dataset
            def _filter(_line, _i):
                return int(_line.split('\t')[-1]) == _i

            db_train = data.Lambda(data.Filter(dataset, fn=lambda _x: not _filter(_x, fold_id)), fn=lambda _x: _x[:-2])
            db_test = data.Lambda(data.Filter(dataset, fn=lambda _x: _filter(_x, fold_id)), fn=lambda _x: _x[:-2])
            # get sampler and loader for training set
            loader_train = data.CMolRNNLoader(db_train, shuffle=True, num_workers=num_workers,
                                              k=k, p=p, conditional=cond, batch_size=batch_size)
            # get sampler and loader for test set
            loader_test = data.CMolRNNLoader(db_test, shuffle=True, num_workers=num_workers,
                                             k=k, p=p, conditional=cond, batch_size=batch_size_test)
        # get iterator
        it_train, it_test = iter(loader_train), iter(loader_test)

    # build model
    if not is_continuous:
        configs = {'N_C': N_C, 'embedding_size': embedding_size, 'hidden_sizes': hidden_sizes,
                   'skip_layer_size': skip_layer_size, 'dense_layer_sizes': dense_layer_sizes,
                   'policy_layers_size': policy_layers_size,
                   'activation': activation, 'rename': True, 'rnn_layers': rnn_layers}
        with open(os.path.join(checkpoint_dir, 'configs.json'), 'w') as f:
            json.dump(configs, f)
    else:
        with open(os.path.join(checkpoint_dir, 'configs.json')) as f:
            configs = json.load(f)

    model = models.CVanillaMolGen_RNN(get_mol_spec().num_atom_types, get_mol_spec().num_bond_types, D=2, **configs)

    ctx = [mx.gpu(i) for i in gpu_ids]
    model.collect_params().initialize(mx.init.Xavier(), force_reinit=True, ctx=ctx)
    if not is_continuous:
        if cond_type == 'kinase':
            # todo model.load_params(os.path.join(checkpoint_dir, 'ckpt.params.bk'), ctx=ctx, allow_missing=True)
            argsd = 5
    else:
        model.load_params(os.path.join(checkpoint_dir, 'ckpt.params'), ctx=ctx)

    # construct optimizer
    opt = mx.optimizer.Adam(learning_rate=lr, clip_gradient=clip_grad)
    trainer = gluon.Trainer(model.collect_params(), opt)
    if is_continuous:
        trainer.load_states(os.path.join(checkpoint_dir, 'trainer.status'))

    if not is_continuous:
        t0 = time.time()
        global_counter = 0
    else:
        with open(os.path.join(checkpoint_dir, 'log.out')) as f:
            records = f.readlines()
            if records[-1] != 'Training finished\n':
                final_record = records[-1]
            else:
                final_record = records[-2]
        count, t_final = int(final_record.split('\t')[0]), float(final_record.split('\t')[1])
        t0 = time.time() - t_final * 60
        global_counter = count

    with open(os.path.join(checkpoint_dir, 'log.out'),
              mode='w' if not is_continuous else 'a') as f:
        if not is_continuous:
            f.write('step\ttime(h)\tloss\tlr\n')
        while True:
            global_counter += 1

            try:
                inputs = [next(it_train) for _ in range(len(gpu_ids))]
            except StopIteration:
                it_train = iter(loader_train)
                inputs = [next(it_train) for _ in range(len(gpu_ids))]

            # move to gpu
            inputs = [data.CMolRNNLoader.from_numpy_to_tensor(input_i, j)
                      for j, input_i in zip(gpu_ids, inputs)]

            with autograd.record():
                loss = [(model(*input_i)).as_in_context(mx.gpu(gpu_ids[0])) for input_i in inputs]
                loss = sum(loss) / len(gpu_ids)
                loss.backward()

            nd.waitall()
            gc.collect()

            trainer.step(batch_size=1)

            if global_counter % decay_step == 0:
                trainer.set_learning_rate(trainer.learning_rate * (1.0 - decay))

            if global_counter % summary_step == 0:
                if is_full:
                    loss = np.asscalar((sum(loss) / len(gpu_ids)).asnumpy())
                else:
                    del loss, inputs
                    gc.collect()

                    try:
                        inputs = [next(it_test) for _ in range(len(gpu_ids))]
                    except StopIteration:
                        it_test = iter(loader_test)
                        inputs = [next(it_test) for _ in range(len(gpu_ids))]

                    with autograd.predict_mode():
                        # move to gpu
                        inputs = [data.CMolRNNLoader.from_numpy_to_tensor(input_i, j)
                                  for j, input_i in zip(gpu_ids, inputs)]
                        loss = [(model(*input_i)).as_in_context(mx.gpu(gpu_ids[0])) for input_i in inputs]
                        loss = np.asscalar((sum(loss) / len(gpu_ids)).asnumpy())

                model.save_params(os.path.join(checkpoint_dir, 'ckpt.params'))
                trainer.save_states(os.path.join(checkpoint_dir, 'trainer.status'))

                f.write('{}\t{}\t{}\t{}\n'.format(global_counter, float(time.time() - t0) / 60, loss,
                                                  trainer.learning_rate))
                print('{}\t{}\t{}\t{}\n'.format(global_counter, float(time.time() - t0) / 60, loss,
                                                trainer.learning_rate))
                f.flush()

                del loss, inputs
                gc.collect()

            if global_counter >= iterations:
                break

        # save before exit
        model.save_parameters(os.path.join(checkpoint_dir, 'ckpt.params'))
        trainer.save_states(os.path.join(checkpoint_dir, 'trainer.status'))

        f.write('Training finished\n')
