from ROOT import *
import root_numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print 1
import numpy
import sys

file=TFile('/gpfs/ddn/cms/user/lgiannini/DeepNtupleRegression/Train_wSV.root')#/uscms_data/d3/lgiannin/tree_Reg.root')
tree=file.Get("tree")
tree2=root_numpy.tree2array(tree, ["nPVs", "jetpt", "jeteta", "genjetpt", "genjetpt_wNu"], selection="")

print tree2.shape
print tree2[0].shape
for i in range(len(tree2[0])):
    print i, tree2[0][i]

print tree2

tree2=root_numpy.rec2array(tree2)

print tree2.shape

t2=root_numpy.tree2array(tree, ["ptfrac", "ptrel", "deltaR","deltaEta","deltaPhi","charge", "id"], selection="")
ll=len(t2)
t2=root_numpy.rec2array(t2)

print t2.shape


t3=root_numpy.tree2array(tree, ["sv_pt", "sv_mass", "sv_energyRatio", "sv_deltaR", "sv_deltaEta","sv_deltaPhi","sv_Ntracks", "sv_3dIp", "sv_3dIpErr", "sv_chi2"], selection="")
ll=len(t3)
t3=root_numpy.rec2array(t3)

print t3.shape

numpy.savez_compressed("Tree_training2",tree2, t2, t3)

