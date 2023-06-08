import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import coalesce

import os
import os.path as osp
import shutil

import scipy.sparse as sp
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class AttributedGraphDataset(InMemoryDataset):
    r"""A variety of attributed graph datasets from the
    `"Scaling Attributed Network Embedding to Massive Graphs"
    <https://arxiv.org/abs/2009.00826>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Wiki"`, :obj:`"Cora"`
            :obj:`"CiteSeer"`, :obj:`"PubMed"`, :obj:`"BlogCatalog"`,
            :obj:`"PPI"`, :obj:`"Flickr"`, :obj:`"Facebook"`, :obj:`"Twitter"`,
            :obj:`"TWeibo"`, :obj:`"MAG"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    url = 'https://docs.google.com/uc?export=download&id={}&confirm=t'

    datasets = {
        'wiki': '1EPhlbziZTQv19OsTrKrAJwsElbVPEbiV',
        'cora': '1FyVnpdsTT-lhkVPotUW8OVeuCi1vi3Ey',
        'citeseer': '1d3uQIpHiemWJPgLgTafi70RFYye7hoCp',
        'pubmed': '1DOK3FfslyJoGXUSCSrK5lzdyLfIwOz6k',
        'blogcatalog': '178PqGqh67RUYMMP6-SoRHDoIBh8ku5FS',
        'ppi': '1dvwRpPT4gGtOcNP_Q-G1TKl9NezYhtez',
        'flickr': '1tZp3EB20fAC27SYWwa-x66_8uGsuU62X',
        'facebook': '12aJWAGCM4IvdGI2fiydDNyWzViEOLZH8',
        'twitter': '1fUYggzZlDrt9JsLsSdRUHiEzQRW1kSA4',
        'tweibo': '1-2xHDPFCsuBuFdQN_7GLleWa8R_t50qU',
        'mag': '1ggraUMrQgdUyA3DjSRzzqMv0jFkU65V5',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['attrs.npz', 'edgelist.txt', 'labels.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.url.format(self.datasets[self.name])
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        path = osp.join(self.raw_dir, f'{self.name}.attr')
        if self.name == 'mag':
            path = osp.join(self.raw_dir, self.name)
        for name in self.raw_file_names:
            os.rename(osp.join(path, name), osp.join(self.raw_dir, name))
        shutil.rmtree(path)

    def process(self):
        import pandas as pd
        from torch_sparse import SparseTensor

        x = sp.load_npz(self.raw_paths[0])
        if x.shape[-1] > 10000 or self.name == 'mag':
            x = SparseTensor.from_scipy(x).to(torch.float)
        else:
            x = torch.from_numpy(x.todense()).to(torch.float)

        df = pd.read_csv(self.raw_paths[1], header=None, sep=None,
                         engine='python')
        edge_index = torch.from_numpy(df.values).t().contiguous()

        with open(self.raw_paths[2], 'r') as f:
            ys = f.read().split('\n')[:-1]
            ys = [[int(y) - 1 for y in row.split()[1:]] for row in ys]
            multilabel = max([len(y) for y in ys]) > 1

        if not multilabel:
            y = torch.tensor(ys).view(-1)
        else:
            num_classes = max([y for row in ys for y in row]) + 1
            y = torch.zeros((len(ys), num_classes), dtype=torch.float)
            for i, row in enumerate(ys):
                for j in row:
                    y[i, j] = 1.

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'
        
class Airports(InMemoryDataset):
    r"""The Airports dataset from the `"struc2vec: Learning Node
    Representations from Structural Identity"
    <https://arxiv.org/abs/1704.03165>`_ paper, where nodes denote airports
    and labels correspond to activity levels.
    Features are given by one-hot encoded node identifiers, as described in the
    `"GraLSP: Graph Neural Networks with Local Structural Patterns"
    ` <https://arxiv.org/abs/1911.07675>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"USA"`, :obj:`"Brazil"`,
            :obj:`"Europe"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    edge_url = ('https://github.com/leoribeiro/struc2vec/'
                'raw/master/graph/{}-airports.edgelist')
    label_url = ('https://github.com/leoribeiro/struc2vec/'
                 'raw/master/graph/labels-{}-airports.txt')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['usa', 'brazil', 'europe']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [
            f'{self.name}-airports.edgelist',
            f'labels-{self.name}-airports.txt',
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.edge_url.format(self.name), self.raw_dir)
        download_url(self.label_url.format(self.name), self.raw_dir)

    def process(self):
        index_map, ys = {}, []
        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            for i, row in enumerate(data):
                idx, y = row.split()
                index_map[int(idx)] = i
                ys.append(int(y))
        y = torch.tensor(ys, dtype=torch.long)
        x = torch.eye(y.size(0))

        edge_indices = []
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[:-1]
            for row in data:
                src, dst = row.split()
                edge_indices.append([index_map[int(src)], index_map[int(dst)]])
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_index = coalesce(edge_index, num_nodes=y.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Airports()'