#! /usr/bin/env python
from itertools  import combinations
import ROOT
import sys
from math import *

def deltaPhi(a,b) :
 r = a-b
 while r>2*pi : r-=2*pi
 while r<-2*pi : r+=2*pi
 return r

def deltaR(a,b) :
  dphi=deltaPhi(a.phi(),b.phi())
  return sqrt(dphi*dphi+(a.eta()-b.eta())**2)
from DataFormats.FWLite import Events, Handle
from math import *

debug=0

events = Events (['root://cms-xrd-global.cern.ch//store/mc/RunIISummer17MiniAOD/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8/MINIAODSIM/92X_upgrade2017_realistic_v10_ext1-v1/110000/187F7EDA-0986-E711-ABB3-02163E014C21.root'])

handleGJ1  = Handle ("std::vector<pat::Jet>")
handleGP  = Handle ("std::vector<reco::GenParticle>")

# for now, label is just a tuple of strings that is initialized just
# like and edm::InputTag
labelGJ1 = ("slimmedJets","","PAT")
labelGP = ("prunedGenParticles")

ROOT.gROOT.SetBatch()        # don't pop up canvases
ROOT.gROOT.SetStyle('Plain') # white background

from array import array

f = ROOT.TFile( 'test.root', 'recreate' )
t = ROOT.TTree( 'tree', 'tree' )
 
maxn = 100
jetpt = array( 'f', [ 0 ] )
jeteta = array( 'f', [ 0 ] )
genjetpt = array( 'f', [ 0 ] )
genjetpt_wNu = array( 'f', [ 0 ] )
dr = array( 'f', maxn*[ 0. ] )
deta = array( 'f', maxn*[ 0. ] )
dphi = array( 'f', maxn*[ 0. ] )
id_code = array( 'f', maxn*[ 0. ] ) #g 0, h 1, e 2, mu 3, other 4
charge = array( 'f', maxn*[ 0. ] )
pt = array( 'f', maxn*[ 0. ] )
pdgid = array( 'f', maxn*[ 0. ] )
t.Branch( 'jetpt', jetpt, 'jetpt/F' )
t.Branch( 'jeteta', jeteta, 'jeteta/F' )
t.Branch( 'genjetpt', genjetpt, 'genjetpt/F' )
t.Branch( 'genjetpt_wNu', genjetpt_wNu, 'genjetpt_wNu/F' )
t.Branch( 'deltaR', dr, 'deltaR[100]/F' )
t.Branch( 'deltaEta', deta, 'deltaEta[100]/F' )
t.Branch( 'deltaPhi', dphi, 'deltaPhi[100]/F' )
t.Branch( 'charge', charge, 'charge[100]/F' )
t.Branch( 'id', id_code, 'id[100]/F' )
t.Branch( 'pt', pt, 'pt[100]/F' )
t.Branch( 'pdgid', pdgid, 'pdgid[100]/F' )


# loop over events
count= 0
maxlen=0
for event in events:
    count+=1 
    if count % 10 == 0 :
	print count
    if count > 1000 :
        break
    event.getByLabel (labelGJ1, handleGJ1)
    event.getByLabel (labelGP, handleGP)
    # get the product
    jets1 = handleGJ1.product()
    genparticles = handleGP.product()
    
    neutrinos=[]
    for gp in genparticles:
	if abs(gp.pdgId()) in [12,14,16]:
		if debug: print gp.pdgId(), gp.pt(), gp.eta(), gp.phi(), gp.status(), gp.mass()
                if gp.status()==1: neutrinos.append(gp)

    if debug: print neutrinos

    for j1 in jets1  :
            if abs(j1.eta()) < 2.5 and j1.pt() > 30 and j1.userInt('pileupJetId:fullId') >0 and j1.genJet() and (abs(j1.hadronFlavour())==5 or abs(j1.partonFlavour()) == 5)	:
		cands=[(j1.daughter(c),deltaR(j1.daughter(c),j1)) for c in range(j1.numberOfDaughters()) ]
		cands.sort(key=lambda c : c[1])
		if debug: print j1.pt(), j1.hadronFlavour(),j1.partonFlavour(),j1.genJet().pt() , [(x[0].pt(),x[0].charge(),x[0].pdgId(),x[1]) for x in cands]
		for i,c in enumerate(cands) :
		    dr[i]=c[1]
		    pt[i]=c[0].pt()
		    pdgid[i]=c[0].pdgId()
		    charge[i]=c[0].charge()
                    deta[i]=c[0].eta()-j1.eta()
                    dphi[i]=deltaPhi(c[0].phi(),j1.phi())
                    id_code[i]=4
		    if charge[i]!=0:
			id_code[i]=charge[i]
			if abs(pdgid[i])==11: id_code[i]=2*charge[i]
                        elif abs(pdgid[i])==13: id_code[i]=3*charge[i]

		    elif pdgid[i]==22: id_code[i]=0
		                        
		jeteta[0]=j1.eta()
		jetpt[0]=j1.pt()
		genjetpt[0]=j1.genJet().pt() 
                
                p4wNu=j1.genJet().p4()      
	        for nu in neutrinos:
                   if deltaR(nu,j1.genJet())<0.4:
                        if debug: print j1.genJet().eta(), j1.genJet().phi(), nu.eta(), nu.phi(), nu.pt()
                        p4wNu=p4wNu+nu.p4()

		genjetpt_wNu[0]=p4wNu.pt()
		if debug: print genjetpt_wNu[0], genjetpt[0]

		t.Fill()
		for i in range(100) : 
			dr[i]=0
			pt[i]=0
			pdgid[i]=0		          
                        charge[i]=0
                        deta[i]=0
                        dphi[i]=0
                        id_code[i]=0
		
		if len(cands) > maxlen:
			maxlen=len(cands)
               

print "maxlen",maxlen



f.cd()
t.Write()
#f.Write()
f.Close()
