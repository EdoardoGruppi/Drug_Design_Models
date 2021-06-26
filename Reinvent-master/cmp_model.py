# Import packages
from rdkit.Chem import Descriptors, AllChem as Chem, DataStructs
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Lipinski import NumRotatableBonds, NumHDonors, NumHAcceptors
from rdkit.Chem.MolSurf import TPSA
import numpy as np
import os
import joblib
import pickle
from data.selected_targets import o_dict, pn_dict, th_dict
from pandas import DataFrame


# IMPORTANT: the model used is called ChEMBL_27 and is published by the ChEMBL group in one of their official
# repository called "of_conformal". The link to access the repository is: https://github.com/chembl/of_conformal. The
# paper describing the model is: https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-018-0325-4.pdf.

# The authors suggested to work with the official code using docker to build the environment. In this case instead the
# environment, whose packages can be found in the requirements.txt file, has been directly created with conda.
# The code is also modified to work directly in local without the need to send https requests and consequently to
# handle the responses. Following this strategy, errors due to timeout are avoided as well. Finally, comments and
# definitions are also added to facilitate the comprehension of the users.

class ChEMBL27:
    def __init__(self, n_bits=1024, model_folder="chembl_mcp_models"):
        # Number of bits of the Morgan fingerprints used
        self.n_bits = n_bits
        # Name of the directory hosting the model files
        self.input_dir = model_folder

        self.models = {}
        self.scalers = {}
        # Load all the files within the folder downloaded
        for target_id in pn_dict.keys():
            # Load models
            model_path = f"{self.input_dir}/models/{target_id}/{target_id}_conformal_prediction_model"
            self.models[target_id] = joblib.load(model_path)
            # Load scalers
            scaler_path = f"{self.input_dir}/scalers/{target_id}_scaler.pkl"
            self.scalers[target_id] = pickle.load(open(scaler_path, "rb"))

    @staticmethod
    def pred_category(p0, p1, significance):
        """
        Assign the molecule activity behaviour to a specific class according to the p-values obtained. Molecules can be
        assigned to the following classes: 'active', 'inactive', 'both' or 'empty'.

        :param p0: first p-value obtained from the model.
        :param p1: second p-value obtained from the model.
        :param significance: significance level to overcome to be assigned to a specific class. It is associated with
        the level of confidence of the prediction.
        :return: the predicted class.
        """
        if (p0 >= significance) & (p1 >= significance):
            return "both"
        if (p0 >= significance) & (p1 < significance):
            return "inactive"
        if (p0 < significance) & (p1 >= significance):
            return "active"
        else:
            return "empty"

    def predict(self, descriptors, targets=None):
        """
        Predict the activity behaviour of a molecule towards a set of different targets.

        :param descriptors: molecule information and descriptors.
        :param targets: if None the prediction is computed for all the 500 available targets. Otherwise,
            it corresponds to a list of target_ids on which to evaluate the given molecule.
        :return: the predicted activity of the molecule towards each target.
        """
        # Empty list where to save the results
        predictions = []
        # Select the targets to work with
        target_ids = self.models
        if targets is not None:
            target_ids = {target: self.models[target] for target in targets}
        # For every target
        for target in target_ids:
            # Load the corresponding scalers
            scaler = self.scalers[target]
            # Transform the input as required by the model
            X = np.column_stack((scaler.transform(np.array(descriptors[:6]).reshape(1, -1)),
                                 descriptors[-1].reshape(1, -1),))
            # Predict the activity values
            pred = self.models[target].predict(X)
            # Get the P values
            p0 = float(pred[:, 0])
            p1 = float(pred[:, 1])
            # Format output for a single prediction
            res = {"Target_chembl_id": target,
                   "Organism": o_dict[target],
                   "Pref_name": pn_dict[target],
                   "70%": self.pred_category(p0, p1, 0.3),
                   "80%": self.pred_category(p0, p1, 0.2),
                   "90%": self.pred_category(p0, p1, 0.1),
                   "Threshold": th_dict[target]}
            # Append the results to the dedicated list
            predictions.append(res)
        return predictions

    def calc_descriptors(self, rd_mol):
        """
        Compute the Morgan fingerprints and the molecule information required as input by the model.

        :param rd_mol: rd mol object.
        :return: a list of all the descriptors information.
        """
        # Get Morgan fingerprint
        fp = Chem.GetMorganFingerprintAsBitVect(rd_mol, radius=2, nBits=self.n_bits, useFeatures=False)
        np_fp = np.zeros(self.n_bits)
        # Convert the fingerprint into a numpy array
        DataStructs.ConvertToNumpyArray(fp, np_fp)
        # Get the octanol-water partition coefficient
        log_p = MolLogP(rd_mol)
        #  Get molecular weight
        mwt = Descriptors.MolWt(rd_mol)
        # Get number of rotatable bonds
        rtb = NumRotatableBonds(rd_mol)
        # Get number of hydrogen bond donors
        hbd = NumHDonors(rd_mol)
        # Get number of hydrogen bond acceptors
        hba = NumHAcceptors(rd_mol)
        # Get the topological polar surface area
        tpsa = TPSA(rd_mol)
        return [log_p, mwt, rtb, hbd, hba, tpsa, np_fp]

    def predict_activity(self, smiles, targets=None):
        """
        Computes the activity of a given molecule.

        :param smiles: smiles string defining the molecule to evaluate.
        :param targets: if None the prediction is computed for all the 500 available targets. Otherwise,
            it corresponds to a list of target_ids on which to evaluate the given molecule.
        :return: a pandas dataframe with activity values.
        """
        predictions = []
        # Load molecule from smiles and calculate fp
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Compute descriptors
            descriptors = self.calc_descriptors(mol)
            # Predict the activity values
            predictions = self.predict(descriptors, targets)
        # Return the results as a pandas dataframe
        predictions = DataFrame(predictions)
        return predictions

    def predict_single_molecule(self, smiles, targets=None, confidence=90):
        """
        Computes the activity of a given molecule.

        :param smiles: smiles string defining the molecule to evaluate.
        :param targets: if None the prediction is computed for all the 500 available targets. Otherwise,
            it corresponds to a list of target_ids on which to evaluate the given molecule.
        :return: a pandas dataframe with activity values.
        """
        predictions = []
        # Load molecule from smiles and calculate fp
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Compute descriptors
            descriptors = self.calc_descriptors(mol)
            # Predict the activity values
            predictions = self.predict(descriptors, targets)
        # Return the results as a pandas dataframe
        predictions = DataFrame(predictions)
        if predictions.iloc[0][f'{confidence}%'] in ['active']:
            return 1.0
        else:
            return 0.0
        