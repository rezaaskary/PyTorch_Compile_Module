import torch as pt
from tqdm import tqdm
from torch import nn
import torchmetrics as tm
import functools
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class CompatibilityCompiler:
    def __init__(self,
                 train_split: pt.utils.data.DataLoader = None,
                 valid_split: pt.utils.data.DataLoader = None,
                 verbose: bool = False,
                 print_every: int = None,
                 n_class: int = None,
                 loss: str = 'MSELoss',
                 optimizer: str = 'Adam',
                 metrics: str = None,
                 learning_rate: float = 0.01,
                 device: str = 'cpu',
                 random_seed: int = 42,
                 model: pt.Callable = None,
                 batch_sizes: int = None,
                 n_batches: int = None,
                 epochs: int = None
                 ):
        """
        A class used for checking the compatibility if the input variable
        :param train_split: The training split in the format f either 'pt.utils.data.DataLoader' or a tuple of (x,y)
        :param valid_split: The validation split in the format f either 'pt.utils.data.DataLoader' or a tuple of (x,y)
        :param verbose: A boolean variable used to activate/deactivate the progress bar during training
        :param print_every: Printing the metrics of the model during several iterations
        :param n_class: The number of classes (only for the classification problem)
        :param loss: The loss function used for the optimization. The supported loss functions are:[L1Loss, MSELoss,
                     CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, GaussianNLLLoss, KLDivLoss, BCELoss,
                     BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, HuberLoss,
                     SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss,
                     TripletMarginLoss, TripletMarginWithDistanceLoss]
        :param optimizer: Different optimizer from torch.optim with their default hyperparameters were implemented here.
                    The optimizers are [Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, LBFGS, NAdam, RAdam,
                    RMSprop, Rprop,SGD]
        :param metrics: Several metrics from torchmetrics implemented for classification/regression problems. The
                    available metrics are [pairwise_cosine_similarity, pairwise_euclidean_distance,
                     pairwise_linear_similarity, pairwise_manhattan_distance, ConcordanceCorrCoef, CosineSimilarity,
                     Accuracy, BinaryAccuracy, AUROC, BinaryAUROC]
        :param learning_rate: A float value representative of the learning rate of the optimizer
        :param device: The hardware device used for computation. available choices: [cpu, cuda]
        :param random_seed: An integer used to fix random number generator. The default value is 42.
        :param model: A python class specified with the model.
        :param batch_sizes: The size of each batch of the data
        :param n_batches: The numbrt of batches in the training split.
        :param epochs: An integer value used as the max iterations.
        """

        if isinstance(random_seed, int):
            pt.manual_seed(random_seed)
            pt.cuda.manual_seed(random_seed)
            self.random_seed = random_seed
        elif not random_seed:
            pass
        else:
            raise Exception(f'{random_seed} is not a correct value of random seed')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(batch_sizes, int):
            self.batch_sizes = batch_sizes
        elif not batch_sizes:
            self.batch_sizes = None
        else:
            raise ValueError('The batch sizes is not specified correctly!')
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if isinstance(n_batches, int):
            self.n_batches = n_batches
        elif not n_batches:
            self.n_batches = None
        else:
            raise ValueError('The number of batches is not specified correctly!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if device in ['cpu', 'cuda']:
            self.device = device
        else:
            raise ValueError(f'{device} is not a correct device type.')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if isinstance(model, pt.Callable):
            self.model = model.to(device=self.device)
        else:
            raise ValueError('The pytorch model function is not specified correctly!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if isinstance(learning_rate, float):
            self.learning_rate = learning_rate
        elif not learning_rate:
            self.learning_rate = 1e-2
        else:
            raise ValueError('The learning rate is not specified correctly!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if isinstance(loss, str):
            if loss == 'L1Loss':
                # https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
                self.loss = nn.L1Loss(reduction='mean')

            elif loss == 'MSELoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
                self.optimizer = nn.MSELoss(reduction='mean')

            elif loss == 'CrossEntropyLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
                # logits are the input of the loss function
                self.loss = nn.CrossEntropyLoss(weight=None,
                                                reduction='mean',
                                                ignore_index=-100,
                                                label_smoothing=0)
            elif loss == 'CTCLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss
                self.loss = nn.CTCLoss(blank=0,
                                       reduction='mean',
                                       zero_infinity=False)

            elif loss == 'NLLLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
                self.optimizer = nn.NLLLoss(weight=None,
                                            ignore_index=-100,
                                            reduction='mean'
                                            )
            elif loss == 'PoissonNLLLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html#torch.nn.PoissonNLLLoss
                self.optimizer = nn.PoissonNLLLoss(log_input=True,
                                                   full=False,
                                                   reduction='mean'
                                                   )
            elif loss == 'GaussianNLLLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html#torch.nn.GaussianNLLLoss
                self.optimizer = nn.GaussianNLLLoss(full=False,
                                                    reduction='mean',
                                                    eps=1e-6
                                                    )
            elif loss == 'KLDivLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss
                self.loss = nn.KLDivLoss(reduction='mean',
                                         log_target=False)

            elif loss == 'BCELoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
                self.loss = nn.BCELoss(weight=None,
                                       reduction='mean')
            elif loss == 'BCEWithLogitsLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
                self.loss = nn.BCEWithLogitsLoss(weight=None,
                                                 reduction='mean',
                                                 pos_weight=None
                                                 )
            elif loss == 'MarginRankingLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html#torch.nn.MarginRankingLoss
                self.loss = nn.MarginRankingLoss(margin=0,
                                                 reduction='mean'
                                                 )
            elif loss == 'HingeEmbeddingLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html#torch.nn.HingeEmbeddingLoss
                self.loss = nn.HingeEmbeddingLoss(margin=1.0,
                                                  reduction='mean'
                                                  )
            elif loss == 'MultiLabelMarginLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelMarginLoss.html#torch.nn.MultiLabelMarginLoss
                self.loss = nn.MultiLabelMarginLoss(reduction='mean')

            elif loss == 'HuberLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html#torch.nn.HuberLoss
                self.loss = nn.HuberLoss(reduction='mean',
                                         delta=1.0
                                         )
            elif loss == 'SmoothL1Loss':
                # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
                self.loss = nn.SmoothL1Loss(reduction='mean',
                                            beta=1.0
                                            )
            elif loss == 'SoftMarginLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.SoftMarginLoss.html#torch.nn.SoftMarginLoss
                self.optimizer = nn.SoftMarginLoss(reduction='mean')

            elif loss == 'MultiLabelSoftMarginLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html#torch.nn.MultiLabelSoftMarginLoss
                self.optimizer = nn.MultiLabelSoftMarginLoss(weight=None,
                                                             reduction='mean'
                                                             )
            elif loss == 'CosineEmbeddingLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html#torch.nn.CosineEmbeddingLoss
                self.optimizer = nn.CosineEmbeddingLoss(reduction='mean',
                                                        margin=0
                                                        )
            elif loss == 'MultiMarginLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.MultiMarginLoss.html#torch.nn.MultiMarginLoss
                self.optimizer = nn.MultiMarginLoss(p=1,
                                                    margin=1.0,
                                                    weight=None,
                                                    reduction='mean'
                                                    )
            elif loss == 'TripletMarginLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss
                self.optimizer = nn.TripletMarginLoss(margin=1.0,
                                                      p=2,
                                                      eps=1e-6,
                                                      reduction='mean')

            elif loss == 'TripletMarginWithDistanceLoss':
                # https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss
                self.optimizer = nn.TripletMarginWithDistanceLoss(distance_function=None,
                                                                  margin=1.0,
                                                                  swap=False,
                                                                  reduction='mean')

            else:
                raise ValueError('The specified loss function is not implemented!')
        else:
            raise Exception('Please n')

        if isinstance(optimizer, str):
            if optimizer == 'Adadelta':
                # https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta
                self.optimizer = pt.optim.Adadelta(params=self.model.parameters(),
                                                   lr=self.learning_rate,
                                                   rho=0.9,
                                                   eps=1e-6,
                                                   weight_decay=0
                                                   )
            elif optimizer == 'Adagrad':
                # https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad
                self.optimizer = pt.optim.Adagrad(params=self.model.parameters(),
                                                  lr=self.learning_rate,
                                                  lr_decay=0,
                                                  weight_decay=0,
                                                  eps=1e-10
                                                  )
            elif optimizer == 'Adam':
                # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
                self.optimizer = pt.optim.Adam(params=self.model.parameters(),
                                               lr=self.learning_rate,
                                               betas=(0.9, 0.99),
                                               eps=1e-9,
                                               weight_decay=0,
                                               amsgrad=False
                                               )
            elif optimizer == 'AdamW':
                # https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW
                self.optimizer = pt.optim.AdamW(params=self.model.parameters(),
                                                lr=self.learning_rate,
                                                betas=(0.9, 0.99),
                                                eps=1e-9,
                                                weight_decay=0,
                                                amsgrad=False
                                                )
            elif optimizer == 'SparseAdam':
                # https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam
                self.optimizer = pt.optim.SparseAdam(params=self.model.parameters(),
                                                     lr=self.learning_rate,
                                                     betas=(0.9, 0.99),
                                                     eps=1e-9
                                                     )
            elif optimizer == 'Adamax':
                # https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax
                self.optimizer = pt.optim.Adamax(params=self.model.parameters(),
                                                 lr=self.learning_rate,
                                                 betas=(0.9, 0.99),
                                                 eps=1e-9
                                                 )
            elif optimizer == 'ASGD':
                # https://pytorch.org/docs/stable/generated/torch.optim.ASGD.html#torch.optim.ASGD
                self.optimizer = pt.optim.ASGD(params=self.model.parameters(),
                                               lr=self.learning_rate,
                                               lambd=1e-4,
                                               alpha=0.75,
                                               t0=1e6,
                                               weight_decay=0
                                               )
            elif optimizer == 'LBFGS':
                # https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS
                self.optimizer = pt.optim.LBFGS(params=self.model.parameters(),
                                                lr=self.learning_rate,
                                                max_iter=20,
                                                max_eval=None,
                                                tolerance_grad=1e-7,
                                                tolerance_change=1e-9,
                                                history_size=100,
                                                line_search_fn=None
                                                )
            elif optimizer == 'NAdam':
                # https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam
                self.optimizer = pt.optim.NAdam(params=self.model.parameters(),
                                                lr=self.learning_rate,
                                                betas=(0.9, 0.99),
                                                eps=1e-9,
                                                weight_decay=0,
                                                momentum_decay=0.004
                                                )
            elif optimizer == 'RAdam':
                # https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam
                self.optimizer = pt.optim.RAdam(params=self.model.parameters(),
                                                lr=self.learning_rate,
                                                betas=(0.9, 0.99),
                                                eps=1e-9,
                                                weight_decay=0,
                                                )
            elif optimizer == 'RMSprop':
                # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
                self.optimizer = pt.optim.RMSprop(params=self.model.parameters(),
                                                  lr=self.learning_rate,
                                                  alpha=0.99,
                                                  eps=1e-8,
                                                  weight_decay=0,
                                                  momentum=0,
                                                  centered=False
                                                  )
            elif optimizer == 'Rprop':
                # https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html#torch.optim.Rprop
                self.optimizer = pt.optim.Rprop(params=self.model.parameters(),
                                                lr=self.learning_rate,
                                                etas=(0.5, 1.2),
                                                step_sizes=(1e-6, 50),
                                                )
            elif optimizer == 'SGD':
                # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
                self.optimizer = pt.optim.SGD(params=self.model.parameters(),
                                              lr=self.learning_rate,
                                              momentum=0,
                                              dampening=0,
                                              weight_decay=0,
                                              nesterov=False
                                              )
            else:
                raise ValueError('The specified optimizer is nor implemented')
        else:
            raise Exception('The format of given optimizer variable is not valid!')

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if isinstance(train_split, pt.utils.data.DataLoader):
            self.train_split = train_split
            self.train_n_batches = len(self.train_split)

        elif isinstance(train_split, tuple):
            tensordataset_train = TensorDataset(pt.tensor(train_split[0],
                                                          device=self.device),
                                                pt.tensor(train_split[1],
                                                          device=self.device)
                                                )
            self.train_split = DataLoader(dataset=tensordataset_train,
                                          shuffle=True,
                                          batch_size=self.batch_sizes
                                          )
            self.train_n_batches = len(self.train_split)
        else:
            raise ValueError('Only tuple and DataLoader variables are supported!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(valid_split, pt.utils.data.DataLoader):
            self.valid_split = valid_split
            self.valid_n_batches = len(self.valid_split)

        elif isinstance(valid_split, tuple):
            tensordataset_valid = TensorDataset(pt.tensor(valid_split[0],
                                                          device=self.device),
                                                pt.tensor(valid_split[1],
                                                          device=self.device)
                                                )

            self.valid_split = DataLoader(dataset=tensordataset_valid,
                                          shuffle=True,
                                          batch_size=self.batch_sizes
                                          )
            self.valid_n_batches = len(self.valid_split)

        elif not valid_split:
            self.valid_split = None
            self.valid_n_batches = 0
        else:
            raise ValueError('Only tuple and DataLoader variables are supported!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(epochs, int):
            self.epochs = epochs
        elif not epochs:
            self.epochs = 1000
        else:
            raise ValueError('The number of epochs is not specified correctly!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(n_class, int):
            self.n_class = n_class
        elif not n_class:
            self.n_class = None
        else:
            raise ValueError(f'{n_class} is not the correct value for the number of classes.')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(metrics, list):
            self.metrics = []
            for sub_fun in metrics:
                if isinstance(sub_fun, str):
                    if sub_fun is 'pairwise_cosine_similarity':
                        # https://torchmetrics.readthedocs.io/en/stable/pairwise/cosine_similarity.html
                        self.metrics.append(tm.functional.pairwise_cosine_similarity)
                    elif sub_fun is 'pairwise_euclidean_distance':
                        # https://torchmetrics.readthedocs.io/en/stable/pairwise/euclidean_distance.html
                        self.metrics.append(tm.functional.pairwise_euclidean_distance)
                    elif sub_fun is 'pairwise_linear_similarity':
                        # https://torchmetrics.readthedocs.io/en/stable/pairwise/linear_similarity.html
                        self.metrics.append(tm.functional.pairwise_linear_similarity)
                    elif sub_fun is 'pairwise_manhattan_distance':
                        # https://torchmetrics.readthedocs.io/en/stable/pairwise/manhattan_distance.html
                        self.metrics.append(tm.functional.pairwise_manhattan_distance)
                    elif sub_fun is 'ConcordanceCorrCoef':
                        # https://torchmetrics.readthedocs.io/en/stable/regression/concordance_corr_coef.html
                        self.metrics.append(tm.ConcordanceCorrCoef)
                    elif sub_fun is 'CosineSimilarity':
                        # https://torchmetrics.readthedocs.io/en/stable/regression/cosine_similarity.html
                        self.metrics.append(tm.CosineSimilarity)
                    ###############################################
                    elif sub_fun is 'Accuracy':
                        # https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html
                        self.metrics.append(tm.Accuracy(task='multiclass',
                                                        num_classes=self.n_class))
                    elif sub_fun is 'BinaryAccuracy':
                        # https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html
                        self.metrics.append(tm.classification.BinaryAccuracy)
                    elif sub_fun is 'AUROC':
                        # https://torchmetrics.readthedocs.io/en/stable/classification/auroc.html
                        self.metrics.append(tm.AUROC(task='multiclass', num_classes=self.n_class))
                    elif sub_fun is 'BinaryAUROC':
                        # https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html
                        self.metrics.append(tm.classification.BinaryAUROC)
                else:
                    raise ValueError('Please enter the list of metric functions correctly')
        self.num_metrics = len(self.metrics)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(verbose, bool):
            self.verbose = verbose
        elif not verbose:
            self.verbose = False
        else:
            raise ValueError('Please correctly specify a boolean variable to activate/deactivae verbosity!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(print_every, int):
            self.print_every = max([print_every, self.batch_sizes])
        elif not print_every:
            self.print_every = self.batch_sizes
        else:
            raise Exception('Please correctly specify an integer value to visualize the results.')


class TrainPytorchNN(CompatibilityCompiler):
    def __init__(self, train_split: pt.utils.data.DataLoader = None, valid_split: pt.utils.data.DataLoader = None,
                 n_class: int = None, model: pt.Callable = None, loss: str = 'MSELoss', metrics: list = None,
                 optimizer: str = 'Adam', epochs: int = 1000, learning_rate: float = 1e-2, batch_sizes: int = None,
                 n_batches: int = None, verbose: bool = True, device: str = 'cpu', random_seed: int = 42,
                 print_every: int = None) -> None:
        """
        A class used for checking the compatibility if the input variable
        :param train_split: The training split in the format f either 'pt.utils.data.DataLoader' or a tuple of (x,y)
        :param valid_split: The validation split in the format f either 'pt.utils.data.DataLoader' or a tuple of (x,y)
        :param verbose: A boolean variable used to activate/deactivate the progress bar during training
        :param print_every: Printing the metrics of the model during several iterations
        :param n_class: The number of classes (only for the classification problem)
        :param loss: The loss function used for the optimization. The supported loss functions are:[L1Loss, MSELoss,
                     CrossEntropyLoss, CTCLoss, NLLLoss, PoissonNLLLoss, GaussianNLLLoss, KLDivLoss, BCELoss,
                     BCEWithLogitsLoss, MarginRankingLoss, HingeEmbeddingLoss, MultiLabelMarginLoss, HuberLoss,
                     SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, CosineEmbeddingLoss, MultiMarginLoss,
                     TripletMarginLoss, TripletMarginWithDistanceLoss]
        :param optimizer: Different optimizer from torch.optim with their default hyperparameters were implemented here.
                    The optimizers are [Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, LBFGS, NAdam, RAdam,
                    RMSprop, Rprop,SGD]
        :param metrics: Several metrics from torchmetrics implemented for classification/regression problems. The
                    available metrics are [pairwise_cosine_similarity, pairwise_euclidean_distance,
                     pairwise_linear_similarity, pairwise_manhattan_distance, ConcordanceCorrCoef, CosineSimilarity,
                     Accuracy, BinaryAccuracy, AUROC, BinaryAUROC]
        :param learning_rate: A float value representative of the learning rate of the optimizer
        :param device: The hardware device used for computation. available choices: [cpu, cuda]
        :param random_seed: An integer used to fix random number generator. The default value is 42.
        :param model: A python class specified with the model.
        :param batch_sizes: The size of each batch of the data
        :param n_batches: The numbrt of batches in the training split.
        :param epochs: An integer value used as the max iterations.
        """
        super(TrainPytorchNN, self).__init__(optimizer=optimizer,
                                             train_split=train_split,
                                             valid_split=valid_split,
                                             model=model,
                                             loss=loss,
                                             verbose=verbose,
                                             n_class=n_class,
                                             epochs=epochs,
                                             batch_sizes=batch_sizes,
                                             metrics=metrics,
                                             n_batches=n_batches,
                                             print_every=print_every,
                                             learning_rate=learning_rate,
                                             device=device,
                                             random_seed=random_seed)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        def _metric_calculator(predicted_variable: pt.tensor, true_variables: pt.tensor,
                               index: int, previous_scores: list) -> list:
            """

            :param true_variables:
            :param predicted_variable:
            :param index:
            :param previous_scores:
            :return:
            """
            return [met(predicted_variable, true_variables) / (index + 1) + previous_scores[metric_ind] * \
                    (index / (index + 1)) for metric_ind, met in enumerate(self.metrics)]

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        def _class_calculator(predicted_probablities: pt.tensor) -> pt.tensor:
            return pt.argmax(predicted_probablities, dim=1)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        def _class_calculator_binary(predicted_probablities: pt.tensor) -> pt.tensor:
            return pt.round(predicted_probablities)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.metric_calculator = _metric_calculator
        if self.n_class == 2:
            self.prob2label = _class_calculator_binary
        elif self.n_class > 2:
            self.prob2label = _class_calculator
        else:
            raise ValueError('The number of classes must be a greater than two.')

        def _model_run_():
            for epoch in tqdm(range(self.epochs), desc="Training...", disable=not self.verbose):
                train_loss, train_metrics = 0, [0] * self.num_metrics
                for batch_index, (x_mini_train, y_mini_train) in enumerate(self.train_split):
                    self.model.train()
                    y_train_pred_prob = self.model(x_mini_train)
                    train_loss_value = self.loss(y_train_pred_prob, y_mini_train)
                    y_train_pred_label = self.prob2label(y_train_pred_prob)
                    train_metrics = self.metric_calculator(y_train_pred_prob, y_mini_train,
                                                           batch_index, train_metrics)
                    train_loss += train_loss_value
                    self.optimizer.zero_grad()
                    train_loss_value.backward()
                    self.optimizer.step()
                    # if batch_index %

                valid_loss, valid_metrics = 0, [0] * self.num_metrics
                self.model.eval()
                with pt.inference_mode():
                    for batch_index_valid, (x_mini_valid, y_mini_valid) in enumerate(self.valid_split):
                        y_valid_pred_prob = self.model(x_mini_valid)
                        valid_loss_value = self.loss(y_valid_pred_prob, y_mini_valid)
                        # %%                    y_train_pred_label = self.prob2label(y_valid_pred_prob)
                        valid_loss += valid_loss_value
                        valid_metrics = self.metric_calculator(y_valid_pred_prob, y_mini_valid,
                                                               batch_index_valid, valid_metrics)
