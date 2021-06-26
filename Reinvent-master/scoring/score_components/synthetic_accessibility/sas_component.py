import pickle
from typing import List, Tuple
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem, Descriptors
from scoring.component_parameters import ComponentParameters
from scoring.score_components.base_score_component import BaseScoreComponent
from scoring.score_components.synthetic_accessibility.sascorer import calculateScore
from scoring.score_summary import ComponentSummary


class SASComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = self._calculate_sa(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def _calculate_sa(self, query_mols) -> np.array:
      sa_scores = []
      for mol in query_mols:
          try:
              sa_score = 1 / calculateScore(mol)
          except ValueError:
              sa_score = 0.0
          sa_scores.append(sa_score)
      return np.array(sa_scores, dtype=np.float32)
