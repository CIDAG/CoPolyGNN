import os
import os.path as osp
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data import Data, download_url, extract_zip, InMemoryDataset
from torch_geometric.utils import one_hot, remove_self_loops

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Atom types and their corresponding indices
ATOM_TYPES = {
    'H': 0, 'C': 1, 'N': 2, 'O': 3,
    'F': 4, 'Si': 5, 'P': 6, 'S': 7,
    'Cl': 8, 'Br': 9, 'I': 10,
    'Na': 11, 'Ge': 12
}

# Monomer sets
NMR_MONOMERS = ['%PEGA', '%HEA', '%TFEA', '%MSEA', '%HexaFOEA', '%NonaFOEA']
ACIDS_MONOMERS = [
    '%SA', '%ADP', '%DDDA', '%AZL', '%1,4-CHDA', '%1,3-CHDA',
    '%1,2-CHDA', '%HHPA', '%PA', '%IPA', '%TPA', '%TMA'
]
GLYCOLS_MONOMERS = [
    '%EG', '%DEG', '%1,3-PROP', '%1,4-BUT', '%HDO', '%NPG',
    '%1,4-CHDM', '%1,3-CHDM', '%TMCD', '%TMP', '%MPD', '%TCDDM'
]

class Polymers(InMemoryDataset):
    raw_url = 'https://www.dropbox.com/scl/fi/ykbeg4u7fzlobua4m9tju/datasets.zip?rlkey=dsnmkmq6pdt4tvkrp6wktj9cj&st=6es2c8ji&dl=1'

    def __init__(self, root: str, task_id: int, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
        """
        Initializes the Polymers dataset.

        :param root: Root directory where the dataset should be saved.
        :param task_id: Task ID for the specific dataset.
        :param transform: A function/transform that takes in a Data object and returns a transformed version. The data object will be transformed before every access. Default is None.
        :param pre_transform: A function/transform that takes in a Data object and returns a transformed version. The data object will be transformed before being saved to disk. Default is None.
        :param pre_filter: A function that takes in a Data object and returns a boolean value, indicating whether the data object should be included in the final dataset. Default is None.
        """
        self.task_id = task_id
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """
        Returns the raw file names.

        :return: List of raw file names
        """
        return ['polymers.csv']

    @property
    def processed_file_names(self) -> str:
        """
        Returns the processed file name.

        :return: Processed file name
        """
        return 'data.pt'

    def download(self):
        """
        Downloads and extracts the dataset.
        """
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        for file in os.listdir(self.raw_dir):
            if file != f'dataset_{self.task_id}.csv':
                os.remove(osp.join(self.raw_dir, file))

        os.rename(osp.join(self.raw_dir, f'dataset_{self.task_id}.csv'),
                  osp.join(self.raw_dir, 'polymers.csv'))

    def get_features(self, mol: Chem.Mol, target: float, features: Optional[float] = None) -> Data:
        """
        Extracts features from an RDKit molecule.

        :param mol: RDKit molecule
        :param target: Target value for the molecule
        :param features: Additional features. Default is None.
        :return: A PyTorch Geometric Data object containing the molecular features
        """
        N = mol.GetNumAtoms()
        type_idx, atomic_number, aromatic, sp, sp2, sp3, num_hs, pp, ratio, monid = self.get_atom_features(mol)

        z = torch.tensor(atomic_number, dtype=torch.long)
        edge_index = self.get_edge_index(mol, N)
        edge_index, _ = remove_self_loops(edge_index)

        x1 = one_hot(torch.tensor(type_idx), num_classes=len(ATOM_TYPES))
        x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs, pp], dtype=torch.float).t().contiguous()
        x = torch.cat([x1, x2], dim=-1)

        data_kwargs = {
            'x': x, 'edge_index': edge_index, 'y': torch.tensor(target, dtype=torch.float32),
            'monomer_id': torch.tensor(monid), 'ratio': torch.tensor(ratio)
        }
        if features is not None:
            data_kwargs['feats'] = torch.tensor([[features]], dtype=torch.float32)
        return Data(**data_kwargs)

    def get_atom_features(self, mol: Chem.Mol) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[float], List[int]]:
        """
        Extracts atomic features from an RDKit molecule.

        :param mol: An RDKit molecule object representing the molecule from which features are to be extracted.
        
        :return: A tuple of lists containing the following atomic features:
            - type_idx (List[int]): Indices representing the type of each atom, based on predefined atom types.
            - atomic_number (List[int]): Atomic numbers for each atom in the molecule.
            - aromatic (List[int]): Flags indicating whether each atom is aromatic (1) or not (0).
            - sp (List[int]): Flags indicating whether each atom is in the sp hybridization state (1) or not (0).
            - sp2 (List[int]): Flags indicating whether each atom is in the sp2 hybridization state (1) or not (0).
            - sp3 (List[int]): Flags indicating whether each atom is in the sp3 hybridization state (1) or not (0).
            - num_hs (List[int]): Number of explicit hydrogen atoms attached to each atom.
            - pp (List[int]): Polymerization point indicators for each atom, showing whether the atom is involved in polymerization.
            - ratio (List[float]): Ratios associated with each atom, typically used in polymer chemistry.
            - monid (List[int]): Monomer IDs corresponding to each atom, indicating which monomer the atom belongs to.
        """
        type_idx, atomic_number, aromatic = [], [], []
        sp, sp2, sp3, num_hs, pp, ratio, monid = [], [], [], [], [], [], []
    
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_TYPES[atom.GetSymbol()])
            atomic_number.append(atom.GetAtomicNum())
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == HybridizationType.SP else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            num_hs.append(atom.GetNumExplicitHs())
            pp.append(atom.GetIntProp('polymerization_point'))
            ratio.append(atom.GetDoubleProp('ratio'))
            monid.append(atom.GetIntProp('monomer_id'))
    
        return type_idx, atomic_number, aromatic, sp, sp2, sp3, num_hs, pp, ratio, monid

    def get_edge_index(self, mol: Chem.Mol, N: int) -> torch.Tensor:
        """
        Constructs the edge index for the molecular graph.

        :param mol: RDKit molecule
        :param N: Number of atoms in the molecule
        :return: Edge index tensor
        """
        row, col = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
        edge_index = torch.tensor([row, col], dtype=torch.long)
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        return edge_index[:, perm]

    def build_monomer(self, smi: str, ratio: float, monomer_id: int) -> tuple:
        """
        Builds a monomer from a SMILES string.

        :param smi: SMILES string of the monomer
        :param ratio: Ratio of the monomer in the polymer
        :param monomer_id: Monomer ID
        :return: RDKit molecule with hydrogen atoms added and the updated monomer ID
        """
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.rdmolops.AddHs(mol)

        for atom in mol.GetAtoms():
            atom.SetIntProp(
                'polymerization_point',
                int('*' in [a_n.GetSymbol() for a_n in atom.GetNeighbors()])
            )
            atom.SetDoubleProp('ratio', ratio)
            atom.SetIntProp('monomer_id', monomer_id)
        return Chem.DeleteSubstructs(mol, Chem.MolFromSmiles('[*]')), monomer_id + 1

    def process(self):
        """
        Processes the raw dataset and saves it in a processed format.
        """
        df = pd.read_csv(self.raw_paths[0])
        monomer_id = 0
        data_list = []

        for _, row in df.iterrows():
            mol, monomer_id, features = self.build_molecule(row, monomer_id)
            data = self.get_features(mol, row[row.index[-1]], features)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def build_molecule(self, row: pd.Series, monomer_id: int) -> Tuple[Chem.Mol, int, Optional[float]]:
        """
        Builds a molecule based on the task ID and the input row.

        :param row: Pandas series containing row data.
        :param monomer_id: Current monomer ID.
        :return: Tuple of the RDKit molecule and optional features.
        """
        if self.task_id < 2:
            mol1, monomer_id = self.build_monomer(row['monoA'], row['%monoA'], monomer_id)
            mol2, monomer_id = self.build_monomer(row['monoB'], 1 - row['%monoA'], monomer_id)
            mol = Chem.CombineMols(mol1, mol2)
            features = None
        elif self.task_id in [35, 36, 37]:
            mol, monomer_id = self.build_polymer_from_monomers(row, monomer_id, [NMR_MONOMERS] if self.task_id == 37 else [ACIDS_MONOMERS, GLYCOLS_MONOMERS])
            features = self._get_task_specific_features(row)
        else:
            mol, monomer_id = self.build_monomer(row['SMILES'], 1., monomer_id)
            features = self._get_task_specific_features(row)

        return mol, monomer_id, features

    def _get_task_specific_features(self, row: pd.Series) -> Optional[float]:
        """
        Retrieves task-specific features from the input row.

        :param row: Pandas series containing row data.
        :return: Optional float representing task-specific features.
        """
        feature_map = {
            11: 'Slope',
            31: 'length',
            35: 'Mw',
            36: 'Mw',
            37: 'Weight % Fluorine'
        }
        return row.get(feature_map.get(self.task_id))

    def build_polymer_from_monomers(self, row: pd.Series, monomer_id: int, monomer_sets: List[List[str]]) -> Tuple[Chem.Mol, int]:
        """
        Builds a polymer from monomers.

        :param row: Pandas series containing row data.
        :param monomer_id: Current monomer ID.
        :param monomer_sets: List of monomer sets.
        :return: Tuple of the combined RDKit molecule and the updated monomer ID.
        """
        mols = []
        for monomer_set in monomer_sets:
            for monomer in monomer_set:
                if row[monomer] > 0:
                    aux, monomer_id = self.build_monomer(row[monomer[1:]], row[monomer], monomer_id)
                    mols.append(aux)
        return self._combine_molecules(mols), monomer_id
    
    def _combine_molecules(self, mols: List[Chem.Mol]) -> Chem.Mol:
        """
        Combines multiple molecules into a single molecule.

        :param mols: List of RDKit molecules.
        :return: Combined RDKit molecule.
        """
        mol = mols[0]
        for aux in mols[1:]:
            mol = Chem.CombineMols(mol, aux)
        return mol
