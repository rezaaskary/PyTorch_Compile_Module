import torch as pt
from tqdm import tqdm
from torch import nn
import functools
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

[nn.L1Loss, nn.MSELoss, nn.CrossEntropyLoss, nn.CTCLoss, nn.NLLLoss,
 nn.PoissonNLLLoss, nn.GaussianNLLLoss, nn.KLDivLoss, nn.BCELoss,
 nn.BCEWithLogitsLoss, nn.MarginRankingLoss, nn.HingeEmbeddingLoss,
 nn.MultiLabelMarginLoss, nn.HuberLoss, nn.SmoothL1Loss, nn.SoftMarginLoss,
 nn.MultiLabelMarginLoss, nn.CosineEmbeddingLoss, nn.MultiMarginLoss,
 nn.TripletMarginLoss, nn.TripletMarginWithDistanceLoss]


class CompatibilityCompiler:
    def __init__(self,
                 optimizer: pt.Callable,
                 ):
        if optimizer is 'MAE':
            self.optimizer = nn.L1Loss(reduction='mean')
            self.regr_problem = True
        elif optimizer is 'MSE':
            self.optimizer = nn.MSELoss(reduction='sum')
            self.regr_problem = True
        elif optimizer is 'cross_entropy':
            self.optimizer = nn.CrossEntropyLoss(weight=None,
                                                 reduction='mean',
                                                 ignore_index=-100,
                                                 label_smoothing=0,

                                                 )

        pass


class TrainPytorchNN(CompatibilityCompiler):
    def __init__(self,
                 train_split: pt.utils.data.DataLoader = None,
                 valid_split: pt.utils.data.DataLoader = None,
                 n_class: int = None,
                 model: pt.Callable = None,
                 loss_fcn: pt.Callable = None,
                 metrics: list = None,
                 optimizer: pt.Callable = None,
                 epochs: int = 1000,
                 learning_rate: float = 1e-2,
                 batch_sizes: int = None,
                 n_batches: int = None,
                 verbose: bool = True,
                 device: str = 'cpu',
                 random_seed: int = 42,
                 print_every: int = None,
                 dtype=None
                 ) -> None:

        super(self, TrainPytorchNN).__init__(optimizer=optimizer)

        if dtype in [pt.float, pt.float16, pt.float32, pt.float64, pt.int32, pt.short, pt.long]:
            self.dtype = dtype
        elif not dtype:
            self.dtype = pt.float32
        else:
            raise ValueError(f' {dtype} is not the correct type of variables in pytorch!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if device in ['cpu', 'cuda']:
            self.device = device
        else:
            raise ValueError(f'{device} is not a correct device type.')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(batch_sizes, int):
            self.batch_sizes = batch_sizes
        elif not batch_sizes:
            self.batch_sizes = None
        else:
            raise ValueError('The batch sizes is not specified correctly!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(random_seed, int):
            pt.manual_seed(random_seed)
            pt.cuda.manual_seed(random_seed)
            self.random_seed = random_seed
        elif not random_seed:
            pass
        else:
            raise Exception(f'{random_seed} is not a correct value of random seed')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(n_batches, int):
            self.n_batches = n_batches
        elif not n_batches:
            self.n_batches = None
        else:
            raise ValueError('The number of batches is not specified correctly!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if isinstance(train_split, pt.utils.data.DataLoader):
            self.train_split = train_split
            self.train_n_batches = len(self.train_split)

        elif isinstance(train_split, tuple):
            tensordataset_train = TensorDataset(pt.tensor(train_split[0],
                                                          dtype=self.dtype,
                                                          device=self.device),
                                                pt.tensor(train_split[1],
                                                          dtype=self.dtype,
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

        if isinstance(n_class, int):
            self.n_class = n_class
        elif not n_class:
            self.n_class = 2
        else:
            raise ValueError(f'{n_class} is not the correct value for the number of classes.')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if isinstance(valid_split, pt.utils.data.DataLoader):
            self.valid_split = valid_split
            self.valid_n_batches = len(self.valid_split)

        elif isinstance(valid_split, tuple):
            tensordataset_valid = TensorDataset(pt.tensor(valid_split[0],
                                                          dtype=self.dtype,
                                                          device=self.device),
                                                pt.tensor(valid_split[1],
                                                          dtype=self.dtype,
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

        if isinstance(model, pt.Callable):
            self.model = model.to(device=self.device)
        else:
            raise ValueError('The pytorch model function is not specified correctly!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if isinstance(loss_fcn, pt.Callable):
            self.loss_fcn = loss_fcn
        else:
            raise ValueError('The pytorch loss function is not specified correctly!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(optimizer, pt.Callable):
            self.optimizer = optimizer
        else:
            raise ValueError('The pytorch optimizer function is not specified correctly!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(epochs, int):
            self.epochs = epochs
        elif not epochs:
            self.epochs = 1000
        else:
            raise ValueError('The number of epochs is not specified correctly!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if isinstance(learning_rate, float):
            self.learning_rate = learning_rate
        elif not learning_rate:
            self.learning_rate = 1e-2
        else:
            raise ValueError('The learning rate is not specified correctly!')
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.optimizer = self.optimizer(lr=self.learning_rate,
                                        params=self.model.parameters()
                                        )

        if isinstance(verbose, bool):
            self.verbose = verbose
        elif not verbose:
            self.verbose = False
        else:
            raise ValueError('Please correctly specify a boolean variable to activate/deactivae verbosity!')

        if isinstance(metrics, list):
            for sub_fun in metrics:
                if isinstance(sub_fun, pt.Callable):
                    pass
                else:
                    raise ValueError('Please enter the list of metric functions correctly')
            self.metrics = metrics

        elif isinstance(metrics, pt.Callable):
            self.metrics = [metrics]
        else:
            raise ValueError('Please enter the list of metric functions correctly')

        self.num_metrics = len(self.metrics)

        def _metric_calculator(true_variables: pt.tensor, predicted_variable: pt.tensor,
                               index: int, previous_scores: list) -> list:
            return [met(true_variables, predicted_variable) / (index + 1) + previous_scores[metric_ind] * \
                    (index / (index + 1)) for metric_ind, met in enumerate(self.metrics)]

        def _class_calculator(predicted_probablities: pt.tensor) -> pt.tensor:
            return pt.argmax(predicted_probablities, dim=1)

        def _class_calculator_binary(predicted_probablities: pt.tensor) -> pt.tensor:
            return pt.round(predicted_probablities)

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
                    train_loss_value = self.loss_fcn(y_train_pred_prob, y_mini_train)
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
                        valid_loss_value = self.loss_fcn(y_valid_pred_prob, y_mini_valid)
                        # %%                    y_train_pred_label = self.prob2label(y_valid_pred_prob)
                        valid_loss += valid_loss_value
                        valid_metrics = self.metric_calculator(y_valid_pred_prob, y_mini_valid,
                                                               batch_index_valid, valid_metrics)
