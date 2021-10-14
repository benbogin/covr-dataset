import logging
from collections import deque
from typing import List, Iterator, Optional, Union

import torch
from allennlp.common.tqdm import Tqdm
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict, allennlp_collate
from allennlp.data.dataset_readers import DatasetReader, DatasetReaderInput
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
import allennlp.nn.util as nn_util

from ..dataset_readers.vcqa import VCQADatasetReader

logger = logging.getLogger(__name__)


class TorchDataset(Dataset):
    def __init__(self, instances, data_path, dataset_reader: VCQADatasetReader):
        self.instances = instances
        self.dataset_reader = dataset_reader
        self.data_path = data_path

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        instance = Instance(dict(self.instances[index].fields))
        instance.fields.update(self.dataset_reader.get_extra_fields(self.data_path, index))
        return instance


@DataLoader.register("vcqa_data_loader")
class MultiProcessDataLoader(DataLoader):
    """
    This is mostly similar to AllenNLP's `MultiProcessDataLoader`, however, with two main changes:
    (1) It uses PyTorch dataset loader, which we found to work a bit faster (~10%)
    (2) It will only load the features of the images (i.e. method `get_extra_fields` of the dataset reader)
        during the actual loading of the dataset, and not during the first pass of the dataset, which is only used to
        extract the vocabulary. This saves up some loading time.
    """
    def __init__(
            self,
            reader: DatasetReader,
            data_path: DatasetReaderInput,
            batch_size: int = None,
            drop_last: bool = False,
            shuffle: bool = False,
            num_workers: int = 0,
            cuda_device: Optional[Union[int, str, torch.device]] = None,
    ) -> None:
        # Do some parameter validation.
        if num_workers is not None and num_workers < 0:
            raise ValueError("num_workers cannot be a negative number")

        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if batch_size is None:
            raise ValueError("batch_size is required ")

        self.reader = reader
        self.data_path = data_path
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = allennlp_collate
        self.cuda_device: Optional[torch.device] = None
        if cuda_device is not None:
            if not isinstance(cuda_device, torch.device):
                self.cuda_device = torch.device(cuda_device)
            else:
                self.cuda_device = cuda_device

        # If max_instances_in_memory is not given, we'll keep a cache of all instances in this list.
        self._instances: Optional[List[Instance]] = None

        # For indexing instances.
        self._vocab: Optional[Vocabulary] = None

        self._torch_data_loader = None
        self._loaded_extra_features = False

        deque(self.iter_instances(load_extra_features=False), maxlen=0)


    @overrides
    def index_with(self, vocab: Vocabulary) -> None:
        self._vocab = vocab
        if self._instances:
            for instance in self._instances:
                instance.index_fields(vocab)

    @overrides
    def __len__(self) -> int:
        num_instances = self.reader.get_length(self.data_path)
        # We know batch_size won't be None here since `batch_sampler` is None.
        batch_size: int = self.batch_size  # type: ignore
        if self.drop_last or num_instances % batch_size == 0:
            return num_instances // batch_size
        else:
            return 1 + num_instances // batch_size

    @overrides
    def __iter__(self) -> Iterator[TensorDict]:
        if self._vocab is None:
            raise ValueError(
                "This DataLoader has not been indexed with a Vocabulary yet. "
                "Did you forget to call DataLoader.index_with(vocab)?"
            )

        if not self._torch_data_loader:
            if not self._loaded_extra_features:
                self._instances = []
            deque(self.iter_instances(load_extra_features=True), maxlen=0)

            def collate(instances):
                return allennlp_collate(instances)

            self._torch_data_loader = TorchDataLoader(dataset=TorchDataset(self._instances, self.data_path, self.reader),
                                                      batch_size=self.batch_size,
                                                      shuffle=self.shuffle, num_workers=self.num_workers,
                                                      drop_last=self.drop_last, collate_fn=collate, pin_memory=True)
            self._loaded_extra_features = True

        for batch in self._torch_data_loader.__iter__():
            yield nn_util.move_to_device(batch, self.cuda_device)

    @overrides
    def iter_instances(self, load_extra_features: bool = False) -> Iterator[Instance]:
        if self._instances:
            yield from self._instances
        else:
            self._instances = []

            # inform dataset reader if loading for vocab (first time) or not. If for vocab, loading could be faster
            # (skip loading of visual features)

            # Just read all instances in main process.
            for instance in Tqdm.tqdm(
                    self.reader.read(self.data_path), desc="loading instances"
            ):
                self.reader.apply_token_indexers(instance)
                if self._vocab is not None:
                    instance.index_fields(self._vocab)
                if load_extra_features:
                    self._instances.append(instance)
                yield instance

    @overrides
    def set_target_device(self, device: torch.device) -> None:
        self.cuda_device = device
