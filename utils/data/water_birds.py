import os
import time

import torch
import numpy as np
import pandas as pd
from PIL import Image

from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class WILDSDataset:
    """
    Shared dataset class for all WILDS datasets.
    Each data point in the dataset is an (x, y, metadata) tuple, where:
    - x is the input features
    - y is the target
    - metadata is a vector of relevant information, e.g., domain.
      For convenience, metadata also contains y.
    """
    DEFAULT_SPLITS = {'train': 0, 'val': 1, 'test': 2}
    DEFAULT_SPLIT_NAMES = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
    DEFAULT_SOURCE_DOMAIN_SPLITS = [0]

    def __init__(self, root_dir, download, split_scheme):
        if len(self._metadata_array.shape) == 1:
            self._metadata_array = self._metadata_array.unsqueeze(1)
        self._add_coarse_domain_metadata()
        self.check_init()

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y = self.y_array[idx]
        metadata = self.metadata_array[idx]
        return x, y

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        raise NotImplementedError

    def eval(self, y_pred, y_true):
        """
        Args:
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
            - metadata (Tensor): Metadata
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        raise NotImplementedError

    def get_subset(self, split, frac=1.0, transform=None):
        """
        Args:
            - split (str): Split identifier, e.g., 'train', 'val', 'test'.
                           Must be in self.split_dict.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")

        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        if frac < 1.0:
            # Randomly sample a fraction of the split
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

        return WILDSSubset(self, split_idx, transform)

    def _add_coarse_domain_metadata(self):
        """
        Update metadata fields, map and values with coarse-grained domain information.
        """
        if hasattr(self, '_metadata_map'):
            self._metadata_map['from_source_domain'] = [False, True]
        self._metadata_fields.append('from_source_domain')
        from_source_domain = torch.as_tensor(
            [1 if split in self.source_domain_splits else 0 for split in self.split_array],
            dtype=torch.int64
        ).unsqueeze(dim=1)
        self._metadata_array = torch.cat(
            [self._metadata_array, from_source_domain],
            dim=1
        )

    def check_init(self):
        """
        Convenience function to check that the WILDSDataset is properly configured.
        """
        required_attrs = ['_dataset_name', '_data_dir',
                          '_split_scheme', '_split_array',
                          '_y_array', '_y_size',
                          '_metadata_fields', '_metadata_array']
        for attr_name in required_attrs:
            assert hasattr(self, attr_name), f'WILDSDataset is missing {attr_name}.'

        # Check that data directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Check splits
        assert self.split_dict.keys()==self.split_names.keys()
        assert 'train' in self.split_dict
        assert 'val' in self.split_dict

        # Check the form of the required arrays
        assert (isinstance(self.y_array, torch.Tensor) or isinstance(self.y_array, list))
        assert isinstance(self.metadata_array, torch.Tensor), 'metadata_array must be a torch.Tensor'

        # Check that dimensions match
        assert len(self.y_array) == len(self.metadata_array)
        assert len(self.split_array) == len(self.metadata_array)

        # Check metadata
        assert len(self.metadata_array.shape) == 2
        assert len(self.metadata_fields) == self.metadata_array.shape[1]

        # Check that it is not both classification and detection
        assert not (self.is_classification and self.is_detection)

        # For convenience, include y in metadata_fields if y_size == 1
        if self.y_size == 1:
            assert 'y' in self.metadata_fields

    @property
    def latest_version(cls):
        def is_later(u, v):
            """Returns true if u is a later version than v."""
            u_major, u_minor = tuple(map(int, u.split('.')))
            v_major, v_minor = tuple(map(int, v.split('.')))
            if (u_major > v_major) or (
                (u_major == v_major) and (u_minor > v_minor)):
                return True
            else:
                return False

        latest_version = '0.0'
        for key in cls.versions_dict.keys():
            if is_later(key, latest_version):
                latest_version = key
        return latest_version

    @property
    def dataset_name(self):
        """
        A string that identifies the dataset, e.g., 'amazon', 'camelyon17'.
        """
        return self._dataset_name

    @property
    def version(self):
        """
        A string that identifies the dataset version, e.g., '1.0'.
        """
        if self._version is None:
            return self.latest_version
        else:
            return self._version

    @property
    def versions_dict(self):
        """
        A dictionary where each key is a version string (e.g., '1.0')
        and each value is a dictionary containing the 'download_url' and
        'compressed_size' keys.

        'download_url' is the URL for downloading the dataset archive.
        If None, the dataset cannot be downloaded automatically
        (e.g., because it first requires accepting a usage agreement).

        'compressed_size' is the approximate size of the compressed dataset in bytes.
        """
        return self._versions_dict

    @property
    def data_dir(self):
        """
        The full path to the folder in which the dataset is stored.
        """
        return self._data_dir

    @property
    def collate(self):
        """
        Torch function to collate items in a batch.
        By default returns None -> uses default torch collate.
        """
        return getattr(self, '_collate', None)

    @property
    def split_scheme(self):
        """
        A string identifier of how the split is constructed,
        e.g., 'standard', 'mixed-to-test', 'user', etc.
        """
        return self._split_scheme

    @property
    def split_dict(self):
        """
        A dictionary mapping splits to integer identifiers (used in split_array),
        e.g., {'train': 0, 'val': 1, 'test': 2}.
        Keys should match up with split_names.
        """
        return getattr(self, '_split_dict', WILDSDataset.DEFAULT_SPLITS)

    @property
    def split_names(self):
        """
        A dictionary mapping splits to their pretty names,
        e.g., {'train': 'Train', 'val': 'Validation', 'test': 'Test'}.
        Keys should match up with split_dict.
        """
        return getattr(self, '_split_names', WILDSDataset.DEFAULT_SPLIT_NAMES)

    @property
    def source_domain_splits(self):
        """
        List of split IDs that are from the source domain.
        """
        return getattr(self, '_source_domain_splits', WILDSDataset.DEFAULT_SOURCE_DOMAIN_SPLITS)

    @property
    def split_array(self):
        """
        An array of integers, with split_array[i] representing what split the i-th data point
        belongs to.
        """
        return self._split_array

    @property
    def y_array(self):
        """
        A Tensor of targets (e.g., labels for classification tasks),
        with y_array[i] representing the target of the i-th data point.
        y_array[i] can contain multiple elements.
        """
        return self._y_array

    @property
    def y_size(self):
        """
        The number of dimensions/elements in the target, i.e., len(y_array[i]).
        For standard classification/regression tasks, y_size = 1.
        For multi-task or structured prediction settings, y_size > 1.
        Used for logging and to configure models to produce appropriately-sized output.
        """
        return self._y_size

    @property
    def n_classes(self):
        """
        Number of classes for single-task classification datasets.
        Used for logging and to configure models to produce appropriately-sized output.
        None by default.
        Leave as None if not applicable (e.g., regression or multi-task classification).
        """
        return getattr(self, '_n_classes', None)

    @property
    def is_classification(self):
        """
        Boolean. True if the task is classification, and false otherwise.
        """
        return getattr(self, '_is_classification', (self.n_classes is not None))

    @property
    def is_detection(self):
        """
        Boolean. True if the task is detection, and false otherwise.
        """
        return getattr(self, '_is_detection', False)

    @property
    def metadata_fields(self):
        """
        A list of strings naming each column of the metadata table, e.g., ['hospital', 'y'].
        Must include 'y'.
        """
        return self._metadata_fields

    @property
    def metadata_array(self):
        """
        A Tensor of metadata, with the i-th row representing the metadata associated with
        the i-th data point. The columns correspond to the metadata_fields defined above.
        """
        return self._metadata_array

    @property
    def metadata_map(self):
        """
        An optional dictionary that, for each metadata field, contains a list that maps from
        integers (in metadata_array) to a string representing what that integer means.
        This is only used for logging, so that we print out more intelligible metadata values.
        Each key must be in metadata_fields.
        For example, if we have
            metadata_fields = ['hospital', 'y']
            metadata_map = {'hospital': ['East', 'West']}
        then if metadata_array[i, 0] == 0, the i-th data point belongs to the 'East' hospital
        while if metadata_array[i, 0] == 1, it belongs to the 'West' hospital.
        """
        return getattr(self, '_metadata_map', None)

    @property
    def original_resolution(self):
        """
        Original image resolution for image datasets.
        """
        return getattr(self, '_original_resolution', None)

    def initialize_data_dir(self, root_dir, download):
        """
        Helper function for downloading/updating the dataset if required.
        Note that we only do a version check for datasets where the download_url is set.
        Currently, this includes all datasets except Yelp.
        Datasets for which we don't control the download, like Yelp,
        might not handle versions similarly.
        """
        self.check_version()

        os.makedirs(root_dir, exist_ok=True)
        data_dir = os.path.join(root_dir, f'{self.dataset_name}_v{self.version}')
        version_file = os.path.join(data_dir, f'RELEASE_v{self.version}.txt')

        # If the dataset exists at root_dir, then don't download.
        if not self.dataset_exists_locally(data_dir, version_file):
            self.download_dataset(data_dir, download)
        return data_dir

    def dataset_exists_locally(self, data_dir, version_file):
        download_url = self.versions_dict[self.version]['download_url']
        # There are two ways to download a dataset:
        # 1. Automatically through the WILDS package
        # 2. From a third party (e.g. OGB-MolPCBA is downloaded through the OGB package)
        # Datasets downloaded from a third party need not have a download_url and RELEASE text file.
        return (
            os.path.exists(data_dir) and (
                os.path.exists(version_file) or
                (len(os.listdir(data_dir)) > 0 and download_url is None)
            )
        )

    def download_dataset(self, data_dir, download_flag):
        version_dict = self.versions_dict[self.version]
        download_url = version_dict['download_url']
        compressed_size = version_dict['compressed_size']

        # Check that download_url exists.
        if download_url is None:
            raise ValueError(f'{self.dataset_name} cannot be automatically downloaded. Please download it manually.')

        # Check that the download_flag is set to true.
        if not download_flag:
            raise FileNotFoundError(
                f'The {self.dataset_name} dataset could not be found in {data_dir}. Initialize the dataset with '
                f'download=True to download the dataset. If you are using the example script, run with --download. '
                f'This might take some time for large datasets.'
            )

        from wilds.datasets.download_utils import download_and_extract_archive
        print(f'Downloading dataset to {data_dir}...')
        print(f'You can also download the dataset manually at https://wilds.stanford.edu/downloads.')

        try:
            start_time = time.time()
            download_and_extract_archive(
                url=download_url,
                download_root=data_dir,
                filename='archive.tar.gz',
                remove_finished=True,
                size=compressed_size)
            download_time_in_minutes = (time.time() - start_time) / 60
            print(f"\nIt took {round(download_time_in_minutes, 2)} minutes to download and uncompress the dataset.\n")
        except Exception as e:
            print(f"\n{os.path.join(data_dir, 'archive.tar.gz')} may be corrupted. Please try deleting it and rerunning this command.\n")
            print(f"Exception: ", e)

    def check_version(self):
        # Check that the version is valid.
        if self.version not in self.versions_dict:
            raise ValueError(f'Version {self.version} not supported. Must be in {self.versions_dict.keys()}.')

        # Check that the specified version is the latest version. Otherwise, warn.
        current_major_version, current_minor_version = tuple(map(int, self.version.split('.')))
        latest_major_version, latest_minor_version = tuple(map(int, self.latest_version.split('.')))
        if latest_major_version > current_major_version:
            print(
                f'*****************************\n'
                f'{self.dataset_name} has been updated to version {self.latest_version}.\n'
                f'You are currently using version {self.version}.\n'
                f'We highly recommend updating the dataset by not specifying the older version in the '
                f'command-line argument or dataset constructor.\n'
                f'See https://wilds.stanford.edu/changelog for changes.\n'
                f'*****************************\n')
        elif latest_minor_version > current_minor_version:
            print(
                f'*****************************\n'
                f'{self.dataset_name} has been updated to version {self.latest_version}.\n'
                f'You are currently using version {self.version}.\n'
                f'Please consider updating the dataset.\n'
                f'See https://wilds.stanford.edu/changelog for changes.\n'
                f'*****************************\n')

    @staticmethod
    def standard_eval(metric, y_pred, y_true):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results = {
            **metric.compute(y_pred, y_true),
        }
        results_str = (
            f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        )
        return results, results_str

    @staticmethod
    def standard_group_eval(metric, grouper, y_pred, y_true, aggregate=True):
        """
        Args:
            - metric (Metric): Metric to use for eval
            - grouper (CombinatorialGrouper): Grouper object that converts metadata into groups
            - y_pred (Tensor): Predicted targets
            - y_true (Tensor): True targets
            - metadata (Tensor): Metadata
        Output:
            - results (dict): Dictionary of results
            - results_str (str): Pretty print version of the results
        """
        results, results_str = {}, ''
        if aggregate:
            results.update(metric.compute(y_pred, y_true))
            results_str += f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
        g = grouper.metadata_to_group(metadata)
        group_results = metric.compute_group_wise(y_pred, y_true, g, grouper.n_groups)
        for group_idx in range(grouper.n_groups):
            group_str = grouper.group_field_str(group_idx)
            group_metric = group_results[metric.group_metric_field(group_idx)]
            group_counts = group_results[metric.group_count_field(group_idx)]
            results[f'{metric.name}_{group_str}'] = group_metric
            results[f'count_{group_str}'] = group_counts
            if group_results[metric.group_count_field(group_idx)] == 0:
                continue
            results_str += (
                f'  {grouper.group_str(group_idx)}  '
                f"[n = {group_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
                f"{metric.name} = {group_results[metric.group_metric_field(group_idx)]:5.3f}\n")
        results[f'{metric.worst_group_metric_field}'] = group_results[f'{metric.worst_group_metric_field}']
        results_str += f"Worst-group {metric.name}: {group_results[metric.worst_group_metric_field]:.3f}\n"
        return results, results_str


class WILDSSubset(WILDSDataset):
    def __init__(self, dataset, indices, transform, do_transform_y=False):
        """
        This acts like `torch.utils.data.Subset`, but on `WILDSDatasets`.
        We pass in `transform` (which is used for data augmentation) explicitly
        because it can potentially vary on the training vs. test subsets.

        `do_transform_y` (bool): When this is false (the default),
                                 `self.transform ` acts only on  `x`.
                                 Set this to true if `self.transform` should
                                 operate on `(x,y)` instead of just `x`.
        """
        self.dataset = dataset
        self.indices = indices
        inherited_attrs = ['_dataset_name', '_data_dir', '_collate',
                           '_split_scheme', '_split_dict', '_split_names',
                           '_y_size', '_n_classes',
                           '_metadata_fields', '_metadata_map']
        for attr_name in inherited_attrs:
            if hasattr(dataset, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))
        self.transform = transform
        self.do_transform_y = do_transform_y

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform is not None:
            if self.do_transform_y:
                x, y = self.transform(x, y)
            else:
                x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)

    @property
    def split_array(self):
        return self.dataset._split_array[self.indices]

    @property
    def y_array(self):
        return self.dataset._y_array[self.indices]

    @property
    def metadata_array(self):
        return self.dataset.metadata_array[self.indices]

    def eval(self, y_pred, y_true):
        return self.dataset.eval(y_pred, y_true)

class WaterbirdsDataset(WILDSDataset):
    """
    The Waterbirds dataset.
    This dataset is not part of the official WILDS benchmark.
    We provide it for convenience and to facilitate comparisons to previous work.

    Supported `split_scheme`:
        'official'

    Input (x):
        Images of birds against various backgrounds that have already been cropped and centered.

    Label (y):
        y is binary. It is 1 if the bird is a waterbird (e.g., duck), and 0 if it is a landbird.

    Metadata:
        Each image is annotated with whether the background is a land or water background.

    Original publication:
        @inproceedings{sagawa2019distributionally,
          title = {Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
          author = {Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
          booktitle = {International Conference on Learning Representations},
          year = {2019}
        }

    The dataset was constructed from the CUB-200-2011 dataset and the Places dataset:
        @techreport{WahCUB_200_2011,
        	Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
        	Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
        	Year = {2011}
        	Institution = {California Institute of Technology},
        	Number = {CNS-TR-2011-001}
        }
        @article{zhou2017places,
          title = {Places: A 10 million Image Database for Scene Recognition},
          author = {Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
          journal ={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          year = {2017},
          publisher = {IEEE}
        }

    License:
        The use of this dataset is restricted to non-commercial research and educational purposes.
    """

    '''
    Change so that eval function may be computed in models -> reweights and averages
    '''

    _dataset_name = 'waterbirds'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/',
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        # Note: metadata_df is one-indexed.
        metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the y values
        self._y_array = torch.LongTensor(metadata_df['y'].values)
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack(
            (torch.LongTensor(metadata_df['place'].values), self._y_array),
            dim=1
        )
        self._metadata_fields = ['background', 'y']
        self._metadata_map = {
            'background': [' land', 'water'], # Padding for str formatting
            'y': [' landbird', 'waterbird']
        }

        # Extract filenames
        self._input_array = metadata_df['img_filename'].values
        self._original_resolution = (224, 224)

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        self._split_array = metadata_df['split'].values

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['background', 'y']))

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
       """
       Returns x for a given idx.
       """
       img_filename = os.path.join(
           self.data_dir,
           self._input_array[idx])
       x = Image.open(img_filename).convert('RGB')
       return x

def eval(self, y_pred, y_true, metadata, prediction_fn=None):
    """
    Computes all evaluation metrics.
    Args:
        - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                            But they can also be other model outputs such that prediction_fn(y_pred)
                            are predicted labels.
        - y_true (LongTensor): Ground-truth labels
        - metadata (Tensor): Metadata
        - prediction_fn (function): A function that turns y_pred into predicted labels
    Output:
        - results (dictionary): Dictionary of evaluation metrics
        - results_str (str): String summarizing the evaluation metrics
    """
    metric = Accuracy(prediction_fn=prediction_fn)

    results, results_str = self.standard_group_eval(
        metric,
        self._eval_grouper,
        y_pred, y_true, metadata)

    # For Waterbirds, the validation and test sets are constructed to be more balanced
    # compared to the training set.
    # To compute the actual average accuracy over the empirical (training) distribution,
    # we therefore weight each groups according to their frequency in the training set.

    results['adj_acc_avg'] = (
        (results['acc_y:landbird_background:land'] * 3498
        + results['acc_y:landbird_background:water'] * 184
        + results['acc_y:waterbird_background:land'] * 56
        + results['acc_y:waterbird_background:water'] * 1057) /
        (3498 + 184 + 56 + 1057))

    del results['acc_avg']
    results_str = f"Adjusted average acc: {results['adj_acc_avg']:.3f}\n" + '\n'.join(results_str.split('\n')[1:])

    return results, results_str
