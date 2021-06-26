import numpy as np
from typing import List
from cmp_model import ChEMBL27
from rdkit.Chem import MolToSmiles

from scoring.component_parameters import ComponentParameters
from scoring.score_components.base_score_component import BaseScoreComponent
from scoring.score_summary import ComponentSummary


class ActivityScore(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.model = ChEMBL27()

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = self._calculate_activity(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def _calculate_activity(self, query_mols) -> np.array:
        activity_scores = []
        for mol in query_mols:
            smiles = MolToSmiles(mol)
            try:
                activity_score = self.model.predict_single_molecule(smiles)
            except:
                activity_score = 0.0
            activity_scores.append(activity_score)
        return np.array(activity_scores, dtype=np.float32)

