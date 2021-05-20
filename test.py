from mx_mg import builders, data
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import random
import matplotlib.pyplot as plt
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import QED

# # loading models
# mol_rnn = builders.Vanilla_RNN_Builder('checkpoint/mol_rnn/', gpu_id=0)
#
# # sampling
# samples_mol_rnn = [m for m in mol_rnn.sample(1000) if m is not None]
#
# random.shuffle(samples_mol_rnn)
# img = Draw.MolsToGridImage(samples_mol_rnn[:12], molsPerRow=3)
# img.save('ajeje2.png')
# plt.imshow(Draw.MolsToGridImage(samples_mol_rnn[:12], molsPerRow=3))
# plt.axis('off')
# plt.tight_layout()
# plt.show()


# # CONDITIONAL
#
# # defining scaffolds ------------------------------------------------------------------------------------------
# scaffolds = ['N1(CC2=CC=C(C3=CC=CC=C3C4=NNN=N4)C=C2)C=NC=C1', # -> losartan
#              'C12=CCCCC1C3C(C(CCC4)C4CC3)CC2', # -> hydrocortisone
#              'C1(C2C=CNC=C2)=CC=CC=C1', # -> nifendipine
#              'C12=CC=CC=C1N=C(N3CCNCC3)C4=C(C=CC=C4)N2'] # -> clozapine
# names = ['scaffold_{}'.format(ids) for ids in ['a', 'b', 'c', 'd']]
# scaffolds_mol = [Chem.MolFromSmiles(s) for s in scaffolds]
# Draw.MolsToGridImage(scaffolds_mol, molsPerRow=2, legends=names)
#
# # helper for building fingperprints
# fp_builder = data.conditionals.ScaffoldFP('datasets/scaffolds.smi')
# # build fingerprint for the four scaffolds
# c = np.zeros([4, len(fp_builder.scaffolds)], dtype=np.float32)
# for i, s in enumerate(scaffolds):
#     on_bits = fp_builder.get_on_bits(s)
#     c[i, on_bits] = 1.0
#
# # load model
# mdl_scaffold = builders.CVanilla_RNN_Builder('ckpt/scaffold_0/', gpu_id=0)
#
# # sample results
# scaffold_outputs = []
# for i in range(4):
#     samples_scaffold_i = [m for m in mdl_scaffold.sample(100, c=c[i, :], output_type='mol')
#                           if m is not None]
#     samples_scaffold_i_match = []
#     for m in samples_scaffold_i:
#         c_i = fp_builder(m)[1].astype(np.float)
#         if np.all(c_i==c[i, :]):
#             samples_scaffold_i_match.append(m)
#     _smiles_list = [Chem.MolToSmiles(m) for m in samples_scaffold_i_match]
#     _smiles_list = list(set(_smiles_list))
#     samples_scaffold_i_match = [Chem.MolFromSmiles(s) for s in _smiles_list]
#     random.shuffle(samples_scaffold_i_match)
#     scaffold_outputs.append(samples_scaffold_i_match)
#
# random.shuffle(scaffold_outputs[0])
# Draw.MolsToGridImage(scaffold_outputs[0][:12], molsPerRow=4)
#
#
# random.shuffle(scaffold_outputs[1])
# Draw.MolsToGridImage(scaffold_outputs[1][:12], molsPerRow=4)
#
# random.shuffle(scaffold_outputs[2])
# Draw.MolsToGridImage(scaffold_outputs[2][:12], molsPerRow=4)
#
# random.shuffle(scaffold_outputs[3])
# Draw.MolsToGridImage(scaffold_outputs[3][:12], molsPerRow=4)
#
# Property based generation ---------------------------------------------------------------------------------------
# conditional codes:
c = [[0.84, 1.9],
     [0.27, 2.5],
     [0.84, 3.8],
     [0.27, 4.8]]
c = np.array(c, dtype=np.float32)
# load model
mdl_prop = builders.CVanilla_RNN_Builder('checkpoint/cp_mol_rnn', gpu_id=0)
# sample results
prop_outputs = []
for i in range(4):
    samples_prop_i = [m for m in mdl_prop.sample(2000, c=c[i, :], output_type='mol') if m is not None]
    _smiles_list = [Chem.MolToSmiles(m) for m in samples_prop_i]
    _smiles_list = list(set(_smiles_list))
    samples_prop_i = [Chem.MolFromSmiles(s) for s in _smiles_list]
    random.shuffle(samples_prop_i)
    prop_outputs.append(samples_prop_i)

random.shuffle(prop_outputs[0])
prop_outputs[0] = [mol for mol in prop_outputs[0] if mol is not None]
legends = ['QED={:.2f},\n SAscore={:.2f}'.format(QED.qed(m), sascorer.calculateScore(m))
           for m in prop_outputs[0][:12]]
img = Draw.MolsToGridImage(prop_outputs[0][:12], molsPerRow=4, legends=legends)
img.save('ajeje.png')
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()

random.shuffle(prop_outputs[1])
prop_outputs[1] = [mol for mol in prop_outputs[1] if mol is not None]
legends = ['QED={:.2f},\n SAscore={:.2f}'.format(QED.qed(m), sascorer.calculateScore(m))
           for m in prop_outputs[1][:12]]
img = Draw.MolsToGridImage(prop_outputs[1][:12], molsPerRow=4, legends=legends)
img.save('ajeje1.png')
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()

random.shuffle(prop_outputs[2])
prop_outputs[2] = [mol for mol in prop_outputs[2] if mol is not None]
legends = ['QED={:.2f},\n SAscore={:.2f}'.format(QED.qed(m), sascorer.calculateScore(m))
           for m in prop_outputs[2][:12]]
img = Draw.MolsToGridImage(prop_outputs[2][:12], molsPerRow=4, legends=legends)
img.save('ajeje2.png')
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()

random.shuffle(prop_outputs[3])
prop_outputs[3] = [mol for mol in prop_outputs[3] if mol is not None]
legends = ['QED={:.2f},\n SAscore={:.2f}'.format(QED.qed(m), sascorer.calculateScore(m))
           for m in prop_outputs[3][:12]]
img = Draw.MolsToGridImage(prop_outputs[3][:12], molsPerRow=4, legends=legends)
img.save('ajeje3.png')
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()

# # GSK-3$\beta$ and JNK3 ------------------------------------------------------------------------
# # conditional codes:
# c = [[1, 1],  # GSK-3b(+) & JNK3(+)
#      [1, 0],  # GSK-3b(+) & JNK3(-)
#      [0, 1]]  # GSK-3b(-) & JNK3(+)
# c = np.array(c, dtype=np.float32)
# # load model
# mdl_kinase = builders.CVanilla_RNN_Builder('checkpoint/ck_mol_rnn', gpu_id=0)
# # sample results
# kinase_outputs = []
# for i in range(3):
#     samples_kinase_i = [m for m in mdl_kinase.sample(10000, c=c[i, :], output_type='mol') if m is not None]
#     _smiles_list = [Chem.MolToSmiles(m) for m in samples_kinase_i]
#     _smiles_list = list(set(_smiles_list))
#     samples_kinase_i = [Chem.MolFromSmiles(s) for s in _smiles_list]
#     random.shuffle(samples_kinase_i)
#     kinase_outputs.append(samples_kinase_i)
#
# random.shuffle(kinase_outputs[0])
# img = Draw.MolsToGridImage(kinase_outputs[0][:12], molsPerRow=4)
# img.save('ajeje1.png')
# plt.imshow(img)
# plt.axis('off')
# plt.tight_layout()
# plt.show()
#
# random.shuffle(kinase_outputs[1])
# img = Draw.MolsToGridImage(kinase_outputs[1][:12], molsPerRow=4)
# img.save('ajeje2.png')
# plt.imshow(img)
# plt.axis('off')
# plt.tight_layout()
# plt.show()
#
# random.shuffle(kinase_outputs[2])
# img = Draw.MolsToGridImage(kinase_outputs[2][:12], molsPerRow=4)
# img.save('ajeje3.png')
# plt.imshow(img)
# plt.axis('off')
# plt.tight_layout()
# plt.show()
