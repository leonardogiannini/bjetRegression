from ROOT import *
import root_numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print 1
import numpy
import sys

file=TFile('/eos/uscms/store/user/lgiannin/Processed_TTbar_new.root')#/uscms_data/d3/lgiannin/tree_Reg.root')
tree=file.Get("tree")
tree=root_numpy.tree2array(tree, selection=" Jet_mcPt>20  && abs(Jet_mcFlavour)==5 &&   Jet_pt>15 ")

print tree.shape
print tree[0].shape
for i in range(len(tree[0])):
    print i, tree[0][i]


tree2=root_numpy.rec2array(tree)

print tree2.shape

numpy.savez_compressed("NumpyTreeRegression_TTbar",tree2,tree)

[('nPVs', '<f4'), 		#0
('Jet_vtxNtrk', '<i4'), 	#1
('Jet_vtxMass', '<f4'), 	#2
('Jet_vtx3dL', '<f4'), 		#3
('Jet_vtx3deL', '<f4'), 	#4
('Jet_vtxPt', '<f4'), 		#5
('dR', '<f4'), 			#6
('Jet_puId', '<f4'), 		#7
('Jet_btagCSV', '<f4'), 	#8
('Jet_rawPt', '<f4'), 		#9
('Jet_corr', '<f4'), 		#10
('Jet_mcPt', '<f4'), 		#11
('Jet_mcPtq', '<f4'), 		#12
('Jet_mcFlavour', '<f4'), 	#13
('Jet_pt', '<f4'), 		#14
('Jet_ptd', '<f4'),             #15
('Jet_mt', '<f4'), 		#16
('Jet_eta', '<f4'), 		#17
('Jet_phi', '<f4'), 		#18
('Jet_mass', '<f4'), 		#19
('Jet_chHEF', '<f4'), 		#20
('Jet_neHEF', '<f4'), 		#21
('Jet_chEmEF', '<f4'), 		#22
('Jet_neEmEF', '<f4'), 		#23
('Jet_chMult', '<f4'), 		#24	
('Jet_leadTrackPt', '<f4'), 	#25
('Jet_mcEta', '<f4'), 		#26
('Jet_mcPhi', '<f4'), 		#27
('Jet_mcM', '<f4'), 		#28
('Jet_mcE', '<f4'), 		#29
('Jet_leptonPt', '<f4'), 	#30
('Jet_leptonPtRel', '<f4'), 	#31
('Jet_leptonPtRelInv', '<f4'),  #32
('Jet_leptonDeltaR', '<f4'), 	#33
('Jet_leptonDeltaEta', '<f4'),  #34
('Jet_leptonDeltaPhi', '<f4'),  #35
('Jet_leptonIsPFMuon', '<f4'),  #36
('Jet_leptonIsTrackerMuon', '<f4'), #37 
('Jet_leptonIsGlobalMuon', '<f4'),  #38
('rho', '<f4'), 		#39
('met_pt', '<f4'), 		#40
('met_phi', '<f4'), 		#41
('Jet_met_dPhi', '<f4'), 	#42
('Jet_met_proj', '<f4')]	#43






