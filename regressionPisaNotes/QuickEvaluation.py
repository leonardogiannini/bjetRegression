#! /usr/bin/env python
from itertools  import combinations
import ROOT
import sys
from math import *
from keras.models import load_model
import numpy

mymodel=load_model("/gpfs/ddn/cms/user/lgiannini/RegressionTraining/PFcands_SVs/h5/mymodel.h5")

def deltaPhi(a,b) :
 r = a-b
 while r>pi : r-=2*pi
 while r<-pi : r+=2*pi
 return r

def deltaR(a,b) :
  dphi=deltaPhi(a.phi(),b.phi())
  return sqrt(dphi*dphi+(a.eta()-b.eta())**2)

def deltaR_Caps(a,b) :
  dphi=deltaPhi(a.Phi(),b.Phi())
  return sqrt(dphi*dphi+(a.Eta()-b.Eta())**2)

from DataFormats.FWLite import Events, Handle

debug=False

output_path="/gpfs/ddn/cms/user/lgiannini/DeepNtupleRegression/"

txt=open('unpodifiles.txt', 'r')
files=txt.readlines()
print files[0:10]
readfile='root://cms-xrd-global.cern.ch/'+files[int(sys.argv[1])]
outname='deepNtupleTTbar_Evaluation_'+sys.argv[1]+'.root'
print readfile

#events = Events (['root://cms-xrd-global.cern.ch//store/mc/RunIISummer17MiniAOD/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8/MINIAODSIM/92X_upgrade2017_realistic_v10_ext1-v1/110000/187F7EDA-0986-E711-ABB3-02163E014C21.root'])

events= Events([readfile])

handleGJ1  = Handle ("std::vector<pat::Jet>")
handleGP  = Handle ("std::vector<reco::GenParticle>")
handleSV = Handle ("std::vector<reco::VertexCompositePtrCandidate>")
handlePV = Handle ("std::vector<reco::Vertex>")
# for now, label is just a tuple of strings that is initialized just
# like and edm::InputTag
labelGJ1 = ("slimmedJets","","PAT")
labelGP = ("prunedGenParticles")
labelSV = ("slimmedSecondaryVertices")
labelPV = ("offlineSlimmedPrimaryVertices")

ROOT.gROOT.SetBatch()        # don't pop up canvases
ROOT.gROOT.SetStyle('Plain') # white background

from array import array

f = ROOT.TFile( output_path+outname, 'recreate' )
t = ROOT.TTree( 'tree', 'tree' )
 
maxn = 100
max2 = 10 
jetpt = array( 'f', [ 0 ] )
jetptraw = array( 'f', [ 0 ] )
jeteta = array( 'f', [ 0 ] )
regressed_pt = array('f', [0])
nPVs = array( 'f', [ 0 ] )
rho = array( 'f', [ 0 ] )
genjetpt = array( 'f', [ 0 ] )
genjetpt_wNu = array( 'f', [ 0 ] )
dr = array( 'f', maxn*[ 0. ] )
deta = array( 'f', maxn*[ 0. ] )
dphi = array( 'f', maxn*[ 0. ] )
id_code = array( 'f', maxn*[ 0. ] ) #g 0, h 1, e 2, mu 3, other 4
charge = array( 'f', maxn*[ 0. ] )
pt = array( 'f', maxn*[ 0. ] )
ptfrac = array( 'f', maxn*[ 0. ] )
ptrel = array( 'f', maxn*[ 0. ] )
pdgid = array( 'f', maxn*[ 0. ] )
sv_pt = array( 'f', max2*[ 0. ] )
sv_ptfrac = array( 'f', max2*[ 0. ] )
sv_deltaR = array( 'f', max2*[ 0. ] )
sv_deltaEta = array( 'f', max2*[ 0. ] )
sv_deltaPhi = array( 'f', max2*[ 0. ] )
sv_mass = array( 'f', max2*[ 0. ] )
sv_Ntracks = array( 'f', max2*[ 0. ] )
sv_chi2 = array( 'f', max2*[ 0. ] )
sv_energyRatio = array( 'f', max2*[ 0. ] )
sv_3dIp = array( 'f', max2*[ 0. ] )
sv_3dIpErr = array( 'f', max2*[ 0. ] )

t.Branch( 'jetpt', jetpt, 'jetpt/F' )
t.Branch( 'jetptraw', jetptraw, 'jetptraw/F' )
t.Branch( 'jeteta', jeteta, 'jeteta/F' )
t.Branch( 'nPVs', nPVs, 'nPVs/F' )
t.Branch( 'rho', rho, 'rho/F' )
t.Branch( 'regressed_pt', regressed_pt, 'regressed_pt/F')
t.Branch( 'genjetpt', genjetpt, 'genjetpt/F' )
t.Branch( 'genjetpt_wNu', genjetpt_wNu, 'genjetpt_wNu/F' )
t.Branch( 'deltaR', dr, 'deltaR[100]/F' )
t.Branch( 'deltaEta', deta, 'deltaEta[100]/F' )
t.Branch( 'deltaPhi', dphi, 'deltaPhi[100]/F' )
t.Branch( 'charge', charge, 'charge[100]/F' )
t.Branch( 'id', id_code, 'id[100]/F' )
t.Branch( 'pt', pt, 'pt[100]/F' )
t.Branch( 'ptfrac', ptfrac, 'ptfrac[100]/F' )
t.Branch( 'pdgid', pdgid, 'pdgid[100]/F' )
t.Branch( 'ptrel', ptrel, 'ptrel[100]/F' )
t.Branch( 'sv_pt', sv_pt, 'sv_pt[5]/F' )
t.Branch( 'sv_ptfrac', sv_ptfrac, 'sv_ptfrac[5]/F' )
t.Branch( 'sv_deltaR', sv_deltaR, 'sv_deltaR[5]/F' )
t.Branch( 'sv_deltaEta', sv_deltaEta, 'sv_deltaEta[5]/F' )
t.Branch( 'sv_deltaPhi', sv_deltaPhi, 'sv_deltaPhi[5]/F' )
t.Branch( 'sv_mass', sv_mass, 'sv_mass[5]/F' )
t.Branch( 'sv_Ntracks', sv_Ntracks, 'sv_Ntracks[5]/F' )
t.Branch( 'sv_chi2', sv_chi2, 'sv_chi2[5]/F' )
t.Branch( 'sv_energyRatio', sv_energyRatio, 'sv_energyRatio[5]/F' )
t.Branch( 'sv_3dIp', sv_3dIp, 'sv_3dIp[5]/F' )
t.Branch( 'sv_3dIpErr', sv_3dIpErr, 'sv_3dIpErr[5]/F' )

# loop over events
count= 0
maxlen=0

jAxis=ROOT.TVector3()
svAxis=ROOT.TVector3()

for event in events:
    count+=1 
    if count % 100 == 0 :
	print count
    #if count > 1000 : break
    event.getByLabel (labelGJ1, handleGJ1)
    event.getByLabel (labelGP, handleGP)
    event.getByLabel (labelSV, handleSV)
    event.getByLabel (labelPV, handlePV)
    # get the product
    jets1 = handleGJ1.product()
    genparticles = handleGP.product()
    secondaryVertices = handleSV.product()
    primaryVertices = handlePV.product()
    pv = primaryVertices[0]
    nPVs[0] = primaryVertices.size()
    
    neutrinos=[]
    for gp in genparticles:
	if abs(gp.pdgId()) in [12,14,16]:
		if debug: print gp.pdgId(), gp.pt(), gp.eta(), gp.phi(), gp.status(), gp.mass()
                if gp.status()==1: neutrinos.append(gp)

    if debug: print neutrinos

    for j1 in jets1  :
            
            jAxis.SetXYZ(j1.px()/j1.p(),j1.py()/j1.p(),j1.pz()/j1.p())
            ptraw=j1.correctedJet("Uncorrected").pt()
            
            if abs(j1.eta()) < 2.5 and j1.pt() > 30 and j1.userInt('pileupJetId:fullId') >0 and j1.genJet() and (abs(j1.hadronFlavour())==5 or abs(j1.partonFlavour()) == 5)	:
		cands=[(j1.daughter(c),deltaR(j1.daughter(c),j1)) for c in range(j1.numberOfDaughters()) ]
		cands.sort(key=lambda c : c[1])
		if debug: print j1.pt(), j1.hadronFlavour(),j1.partonFlavour(),j1.genJet().pt() , [(x[0].pt(),x[0].charge(),x[0].pdgId(),x[1]) for x in cands]
		for i,c in enumerate(cands) :
		    if i == maxn: break
                    dr[i]=c[1]
		    pt[i]=c[0].pt() #non usare mai questa
		    ptfrac[i]=c[0].pt()/ptraw
		    pdgid[i]=c[0].pdgId()
		    charge[i]=c[0].charge()
                    deta[i]=c[0].eta()-j1.eta()
                    dphi[i]=deltaPhi(c[0].phi(),j1.phi())
                    id_code[i]=4
		    my_p4=ROOT.TLorentzVector()
		    my_p4.SetPtEtaPhiM(c[0].pt(),c[0].eta(),c[0].phi(),c[0].mass())
	            ptrel[i]=my_p4.Pt(jAxis)
	            #print pt[i], ptrel[i], j1.pt()
		    if charge[i]!=0:
			id_code[i]=charge[i]
			if abs(pdgid[i])==11: id_code[i]=2*charge[i]
                        elif abs(pdgid[i])==13: id_code[i]=3*charge[i]

		    elif pdgid[i]==22: id_code[i]=0
		                        
		jeteta[0]=j1.eta()
		jetpt[0]=j1.pt()
		genjetpt[0]=j1.genJet().pt() 
		jetptraw[0]=ptraw		
		if debug: print ptraw, jetpt[0]		
                
                p4wNu=j1.genJet().p4()      
	        for nu in neutrinos:
                   if deltaR(nu,j1.genJet())<0.4:
                        if debug: print j1.genJet().eta(), j1.genJet().phi(), nu.eta(), nu.phi(), nu.pt()
                        p4wNu=p4wNu+nu.p4()

		genjetpt_wNu[0]=p4wNu.pt()
		if debug: print genjetpt_wNu[0], genjetpt[0]
                
                j = 0
                
                
                svs=[]
                for sv in secondaryVertices:
		   #print sv.pt(), sv.eta(), sv.phi(), sv.mass()i
		   svAxis.SetXYZ(sv.vertex().x()-pv.x(),sv.vertex().y()-pv.y(),sv.vertex().z()-pv.z()) 
                   svs.append((sv,deltaR_Caps(svAxis, jAxis), svAxis.Mag()))
                svs.sort(key=lambda c : c[1])
                             
               	for sv_tuple in svs:
                   sv=sv_tuple[0]
                   deltaR_Axes=sv_tuple[1]
                   dist=sv_tuple[2]
		   #print sv.pt(), sv.eta(), sv.phi(), sv.mass()i
		   if debug: svAxis.SetXYZ(sv.vertex().x()-pv.x(),sv.vertex().y()-pv.y(),sv.vertex().z()-pv.z())                   
                   if debug: print "deltaR:   ", deltaR(sv, j1), deltaR_Caps(svAxis, jAxis), deltaR_Axes                   
                   if debug: print "matching now"
                   
		   if deltaR_Axes<0.3: #deltaR(sv, j1)< 0.4:
               
			if j == max2: break
        
			sv_pt[j] = sv.pt()
			sv_ptfrac[j] = sv.pt()/ptraw
			sv_deltaR[j] = deltaR(sv, j1)
			sv_deltaEta[j] = sv.eta() - j1.eta()
			sv_deltaPhi[j] = deltaPhi(sv.phi(), j1.phi())
			sv_mass[j] = sv.mass()
			sv_Ntracks[j] = sv.numberOfSourceCandidatePtrs()
			sv_chi2[j] = sv.vertexNormalizedChi2()
			sv_energyRatio[j] = sv.energy()/j1.energy()			
					
			#c++ code https://github.com/cms-sw/cmssw/blob/master/RecoVertex/VertexTools/src/VertexDistance3D.cc
			
			#AlgebraicSymMatrix33 error = vtx1PositionError.matrix() + vtx2PositionError.matrix();
			
                        #vDiff[0] = diff.x();
                        #vDiff[1] = diff.y();
                        #vDiff[2] = diff.z();

                        #double dist=diff.mag();

                        #double err2 = ROOT::Math::Similarity(error,vDiff);
                        #double err = 0.;
                        #if (dist != 0) err  =  sqrt(err2)/dist;
                        
                        err=ROOT.Math.SMatrix('double',3)()
			diff=ROOT.Math.SVector('double',3)()
			err[0][0]=sv.vertexCovariance(0,0)+pv.covariance(0,0)
			err[0][1]=sv.vertexCovariance(0,1)+pv.covariance(0,1)
			err[0][2]=sv.vertexCovariance(0,2)+pv.covariance(0,2)
			err[1][0]=sv.vertexCovariance(1,0)+pv.covariance(1,0)
			err[1][1]=sv.vertexCovariance(1,1)+pv.covariance(1,1)
			err[1][2]=sv.vertexCovariance(1,2)+pv.covariance(1,2)
			err[2][0]=sv.vertexCovariance(2,0)+pv.covariance(2,0)
			err[2][1]=sv.vertexCovariance(2,1)+pv.covariance(2,1)
			err[2][2]=sv.vertexCovariance(2,2)+pv.covariance(2,2)
			diff[0]=sv.vertex().x()-pv.x()
			diff[1]=sv.vertex().y()-pv.y()
			diff[2]=sv.vertex().z()-pv.z()		
			err2 = ROOT.Math.Similarity(err,diff)
			
			if debug: print "ERRs: ",err[0][0],err[0][1],err[0][2],err[1][1],err[1][2],err[2][2]
			if debug: print "mags: ",diff[0], diff[1], diff[2]
			if debug: print err2
		
			error = 0.
                        if (dist != 0): error =(err2)**0.5/dist                        
			
			sv_3dIp[j] = dist
			sv_3dIpErr[j] = error
			
			if debug: print "molto importante: 3dip etc.:  ", svAxis.Mag(), dist, error
			
                        j+=1
                
		print "computation"       
                #[  8.21735992e+01  -5.64785348e-03   2.14355621e+01   8.21762848e+01   8.77624741e+01] 
		#[ 50.88065338   1.16309285   7.74637413  50.66106415  53.48497391]
		print numpy.array([(jetpt[0]-8.21735992e+01)/50.88065338, (jeteta[0]+5.64785348e-03)/1.16309285, (nPVs[0]-2.14355621e+01)/7.74637413]) 
                print numpy.array([(jetpt[0]-8.21735992e+01)/50.88065338, (jeteta[0]+5.64785348e-03)/1.16309285, (nPVs[0]-2.14355621e+01)/7.74637413]).shape
		print numpy.column_stack((ptfrac,ptrel,dr,deta,dphi,charge,id_code)).shape
		print numpy.column_stack((sv_pt,sv_mass,sv_deltaR,sv_deltaEta,sv_deltaPhi,sv_Ntracks,sv_3dIp,sv_3dIpErr)).shape
		regressed_pt[0]=jetpt*mymodel.predict([numpy.array([(jetpt[0]-8.21735992e+01)/50.88065338, (jeteta[0]+5.64785348e-03)/1.16309285, (nPVs[0]-2.14355621e+01)/7.74637413]).reshape(1,3), 
						  numpy.column_stack((ptfrac,ptrel,dr,deta,dphi,charge,id_code)).reshape((1,100,7)), 
						  numpy.column_stack((sv_pt[0:5],sv_mass[0:5],sv_deltaR[0:5],sv_deltaEta[0:5],sv_deltaPhi[0:5],sv_Ntracks[0:5],sv_3dIp[0:5],sv_3dIpErr[0:5])).reshape((1,5,8))])

		print regressed_pt
		t.Fill()
		
		for i in range(maxn) : 
			dr[i]=0
			pt[i]=0
			ptfrac[i]=0
			pdgid[i]=0		          
                        charge[i]=0
                        deta[i]=0
                        dphi[i]=0
                        id_code[i]=0
			ptrel[i]=0

		for j in range(max2) :
			sv_pt[j] = 0
			sv_ptfrac[j] = 0
                        sv_deltaR[j] = 0
                        sv_deltaEta[j] = 0
                        sv_deltaPhi[j] = 0
                        sv_mass[j] = 0
                        sv_Ntracks[j] = 0
                        sv_chi2[j] = 0
                        sv_energyRatio[j] = 0
                        sv_3dIp[j] = 0
                        sv_3dIpErr[j] = 0
		
		if len(cands) > maxlen:
			maxlen=len(cands)
               

print "maxlen",maxlen



f.cd()
t.Write()
#f.Write()
f.Close()
