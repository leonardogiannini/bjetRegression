from ROOT import *
import root_numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print 1
import numpy
import sys,glob

filelist=['deepNtupleTTbar_v1_23.root','deepNtupleTTbar_v1_22.root']
filelist=glob.glob("deepNtupleTTbar_v1_*.root")
print len(filelist)
print filelist
bsize=5120
total=0
for filename in filelist:
    file=TFile(filename, 'r')#/uscms_data/d3/lgiannin/tree_Reg.root')
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

    try:
	print tree2.shape,t3.shape,t2.shape
	tree2=numpy.concatenate((change1,tree2))
	t2=numpy.concatenate((change2,t2))
	t3=numpy.concatenate((change3,t3))
	print "------> attached change"
	print tree2.shape,t3.shape,t2.shape
    except:
	print "no change"

    for i in range(0,len(tree2),bsize):
	print i
        numpy.savez_compressed("batches/Tree_training_"+str(total)+"_"+str(total+bsize),tree2[i:i+bsize], t2[i:i+bsize], t3[i:i+bsize])
	total=total+bsize
         
    print i
    change1=tree2[i:len(tree2)]
    change2=t2[i:len(tree2)]
    change3=t3[i:len(tree2)]
    print change1.shape,change2.shape,change3.shape
