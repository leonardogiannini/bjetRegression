#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include "TLorentzVector.h"
#include <TH1F.h>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <TH2F.h>
#include <TH3F.h>
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <TChain.h>
#include <TLorentzVector.h>
#include <TLegend.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <cmath> 



double deltaPhi(double phi1, double phi2)
{
	double PI = 3.14159265;
	double result = phi1 - phi2;
	while (result > PI) result -= 2*PI;
	while (result <= -PI) result += 2*PI;
	return result;
}

float projectionMETOntoJet(float met, float metphi, float jet, float jetphi, bool onlyPositive=true, float threshold=3.14159265/4.0)
{
  float deltaphi = deltaPhi(metphi, jetphi);
  float met_dot_jet = met * jet * TMath::Cos(deltaphi);
  float jetsq = jet * jet;
  float projection = met_dot_jet / jetsq * jet;

  if (onlyPositive && TMath::Abs(deltaphi) >= threshold)
    return 0.0;
  else
    return projection;
}



void treeForRegression()
{
	//std::string inputfilename ="/eos/uscms/store/user/lpchbb/HeppyNtuples/V23/BulkGravTohhTohbbhbb_narrow_M-1000_13TeV-madgraph/VHBB_HEPPY_V23_BulkGravTohhTohbbhbb_narrow_M-1000_13TeV-madgraph__spr16MAv2-puspr16_HLT_80r2as_v14-v3/160716_234415/0000/tree_2.root";
//	std::string inputfilename ="/eos/uscms/store/user/lpchbb/HeppyNtuples/V23/BulkGravTohhTohbbhbb_narrow_M-1400_13TeV-madgraph.root";//allv23all.root";//lpchbb/HeppyNtuples/V22reHLT/TTJets_TuneCUETP8M1_13TeV.root";
//	std::string inputfilename ="/eos/uscms/store/user/cvernier/ttbar_b25.root";
//	 /gpfs/ddn/srm/cms/store/user/arizzi/VHBBHeppyV28/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8/VH_V28_TT_TuneCUETP8M2T4_13TeV-powheg-Py8__RunIISummer16MAv2-PUMoriond17_80r2as_2016_TrancheI
//	 V_v6-v1/171120_094251/0000/
        std::string inputfilename="/eos/uscms/store/user/lgiannin/Radions.root";//"/uscms_data/d3/cvernier/optimizationRegression/RegressionHbb/newTT.root";///eos/uscms/store/user/lpchbb/HeppyNtuples/V25/ZH_HToBB_ZToLL.root";//allgrav.root";//ttbar-g-25.root";
///eos/uscms/store/group/lpchbb/HeppyNtuples/V21/ZH_HToBB_ZToLL_M125_13TeV_powheg_pythia8.root";
        //"/eos/uscms/store/group/lpchbb/HeppyNtuples/V20test/TTJets_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root";
	TChain *tree=new TChain("tree");
	tree->Add(inputfilename.c_str());
	std::cout<<"Opened input file "<<inputfilename<<std::endl;
	tree->SetBranchAddress("*", 0);
	float rho,nPVs;
	float met_pt,met_eta,met_phi;
	float GenJet_wNuPt[15];
	int          nhJCidx, nhJidx, nJet,nGenBQuarkFromTop;
	int          hJCidx[2], hJidx[2],Jet_mult[30],Jet_mcIdx[30],Jet_mcFlavour[30];  
	float      Jet_area[30],Jet_puId[30],Jet_btagCSV[30],Jet_rawPt[30],Jet_corr[30],Jet_pt[30],Jet_eta[30],Jet_phi[30],Jet_mass[30],Jet_chHEF[30],Jet_neHEF[30],Jet_chEmEF[30],Jet_neEmEF[30],Jet_chMult[30],Jet_leadTrackPt[30],Jet_leptonPt[30],Jet_leptonPtRel[30],Jet_leptonPtRelIn[30],Jet_leptonDeltaR[30],Jet_vtxMass[30],Jet_vtxNtracks[30],Jet_vtxPt[30],Jet_vtx3DSig[30],Jet_vtx3DVal[30],GenBQuarkFromTop_pt[4],GenBQuarkFromTop_eta[4],GenBQuarkFromTop_phi[4],GenBQuarkFromTop_mass[4];

	tree->SetBranchAddress("rho", &rho);
	tree->SetBranchAddress("met_pt", &met_pt);
	tree->SetBranchAddress("met_eta", &met_eta);
	tree->SetBranchAddress("met_phi", &met_phi);
	tree->SetBranchAddress("nhJCidx", &nhJCidx);
	tree->SetBranchAddress("hJCidx", &hJCidx);
	tree->SetBranchAddress("nhJidx", &nhJidx);
	tree->SetBranchAddress("hJidx", &hJidx);
	tree->SetBranchAddress("nJet", &nJet);
	tree->SetBranchAddress("Jet_puId", &Jet_puId);
	tree->SetBranchAddress("Jet_btagCSV", &Jet_btagCSV);
	tree->SetBranchAddress("Jet_rawPt", &Jet_rawPt);
	tree->SetBranchAddress("Jet_mcFlavour", &Jet_mcFlavour);
        tree->SetBranchAddress("nPVs", &nPVs);
	tree->SetBranchAddress("Jet_corr", &Jet_corr);
	tree->SetBranchAddress("Jet_pt", &Jet_pt);
	tree->SetBranchAddress("Jet_eta", &Jet_eta);
	tree->SetBranchAddress("Jet_phi", &Jet_phi);
	tree->SetBranchAddress("Jet_mass", &Jet_mass);
	tree->SetBranchAddress("Jet_chHEF", &Jet_chHEF);
	tree->SetBranchAddress("Jet_neHEF", &Jet_neHEF);
	tree->SetBranchAddress("Jet_chEmEF", &Jet_chEmEF);
	tree->SetBranchAddress("Jet_neEmEF", &Jet_neEmEF);
	tree->SetBranchAddress("Jet_chMult", &Jet_chMult);
	tree->SetBranchAddress("Jet_leadTrackPt", &Jet_leadTrackPt);
	tree->SetBranchAddress("Jet_leptonPt", &Jet_leptonPt);
	tree->SetBranchAddress("Jet_leptonPtRel", &Jet_leptonPtRel);
	tree->SetBranchAddress("Jet_leptonDeltaR", &Jet_leptonDeltaR);
	tree->SetBranchAddress("Jet_vtxMass", &Jet_vtxMass);
	tree->SetBranchAddress("Jet_vtxNtracks", &Jet_vtxNtracks);
	tree->SetBranchAddress("Jet_vtxPt", &Jet_vtxPt);
	tree->SetBranchAddress("Jet_vtx3DSig", &Jet_vtx3DSig);
	tree->SetBranchAddress("Jet_vtx3DVal", &Jet_vtx3DVal);
	tree->SetBranchAddress("Jet_mult", &Jet_mult);
	tree->SetBranchAddress("Jet_mcIdx",&Jet_mcIdx);
	//tree->SetBranchAddress("Jet_area",&Jet_area);
	tree->SetBranchAddress("GenJet_wNuPt",&GenJet_wNuPt);
	tree->SetBranchAddress("GenBQuarkFromTop_pt",&GenBQuarkFromTop_pt);
	tree->SetBranchAddress("GenBQuarkFromTop_mass",&GenBQuarkFromTop_mass);
	tree->SetBranchAddress("GenBQuarkFromTop_phi",&GenBQuarkFromTop_phi);
	tree->SetBranchAddress("GenBQuarkFromTop_eta",&GenBQuarkFromTop_eta);
	tree->SetBranchAddress("nGenBQuarkFromTop",&nGenBQuarkFromTop);




	Long64_t nentries =tree->GetEntries();
	std::cout<< " nentries "<<nentries<<std::endl;
	Long64_t nbytes = 0, nb = 0;
	TFile *outfile=new TFile("/eos/uscms/store/user/lgiannin/Processed.root", "recreate");
	int Jet_vtxNtrk;
	float nPVs_=-999;
	float Jet_met_proj_=-999;
	float Jet_area_,Jet_mass_,Jet_corr_, Jet_e_, rho_, Jet_pt_,Jet_eta_,Jet_phi_, Jet_leptonPtRel_,Jet_leptonDeltaR_,Jet_leptonPt_,  Jet_leadTrackPt_, Jet_chHEF_, Jet_chEmEF_, Jet_neHEF_, Jet_neEmEF_, Jet_vtx3dL_, Jet_vtx3deL_, Jet_chMult_, Jet_mcPt_, Jet_mcEta_, Jet_mcPhi_, Jet_mcM_,  Jet_csv_, Jet_flavor_,  met_pt_, met_phi_, Jet_mcFlavour_, Jet_puId_, Jet_met_dPhi_, Jet_mcE_, Jet_btagCSV_, Jet_mt_, Jet_vtxMass_, Jet_vtx3dL, Jet_vtx3deL, Jet_vtxPt_, dR_, Jet_mcPtq_, Jet_rawPt_;
	Jet_mass_=-999,Jet_corr_=-999, Jet_e_=-999, Jet_mt_=-999, rho_=-999, Jet_pt_=-999,Jet_eta_=-999,Jet_phi_=-999, Jet_leptonPtRel_=-999,Jet_leptonDeltaR_=-999,Jet_leptonPt_=-999,Jet_leadTrackPt_=-999, Jet_chHEF_=-999, Jet_chEmEF_=-999, Jet_neHEF_=-999, Jet_neEmEF_=-999, Jet_vtx3dL_=-999, Jet_vtx3deL_=-999, Jet_chMult_=-999, Jet_mcPt_=-999, Jet_mcEta_=-999, Jet_mcPhi_=-999, Jet_mcM_=-999,   Jet_csv_=-999, Jet_flavor_=-999,  met_pt_=-999, met_phi_=-999, Jet_btagCSV_=-999, Jet_mcFlavour_=-999,  Jet_puId_=-999 , Jet_met_dPhi_=-999, Jet_mcE_=-999,  Jet_vtxMass_=-999., Jet_vtx3dL=-999., Jet_vtx3deL=-999., Jet_vtxPt_ =-999., Jet_mcPtq_=-999.,Jet_rawPt_=-999, Jet_area_=-999;
	Jet_vtxNtrk= 999;
	TTree *outtree = new TTree("tree", "tree");
	
	outtree->Branch("nPVs", &nPVs_ ,"nPVs_/F");

	outtree->Branch("Jet_vtxNtrk",            &Jet_vtxNtrk ,"Jet_vtxNtrk/I");
	outtree->Branch("Jet_vtxMass",            &Jet_vtxMass_ ,"Jet_vtxMass_/F");
	outtree->Branch("Jet_vtx3dL",            &Jet_vtx3dL_ ,"Jet_vtx3dL_/F");
	outtree->Branch("Jet_vtx3deL",            &Jet_vtx3deL_ ,"Jet_vtx3deL_/F");
	outtree->Branch("Jet_vtxPt",            &Jet_vtxPt_ ,"Jet_vtxPt_/F");
	outtree->Branch("dR",            &dR_ ,"dR_/F");
	outtree->Branch("Jet_puId",          &Jet_puId_ ,"Jet_puId_/F");
	//outtree->Branch("Jet_area",          &Jet_area_ ,"Jet_area_/F");
	outtree->Branch("Jet_btagCSV",       &Jet_btagCSV_ ,"Jet_btagCSV_/F");
	outtree->Branch("Jet_rawPt",    &Jet_rawPt_ ,"Jet_rawPt_/F");
	outtree->Branch("Jet_corr",    &Jet_corr_ ,"Jet_corr_/F");
	outtree->Branch("Jet_mcPt",    &Jet_mcPt_ ,"Jet_mcPt_/F");
	outtree->Branch("Jet_mcPtq",    &Jet_mcPtq_ ,"Jet_mcPtq_/F");
	outtree->Branch("Jet_mcFlavour",    &Jet_mcFlavour_ ,"Jet_mcFlavour_/F");
	outtree->Branch("Jet_pt",    &Jet_pt_ ,"Jet_pt_/F");
	outtree->Branch("Jet_mt",    &Jet_mt_ ,"Jet_mt_/F");
	outtree->Branch("Jet_eta",    &Jet_eta_ ,"Jet_eta_/F");
	outtree->Branch("Jet_phi",    &Jet_phi_ ,"Jet_phi_/F");
	outtree->Branch("Jet_mass",    &Jet_mass_ ,"Jet_mass_/F");
	outtree->Branch("Jet_chHEF",    &Jet_chHEF_, "Jet_chHEF_/F");
	outtree->Branch("Jet_neHEF",    &Jet_neHEF_, "Jet_neHEF_/F");
	outtree->Branch("Jet_chEmEF",    &Jet_chEmEF_ ,"Jet_chEmEF_/F");
	outtree->Branch("Jet_neEmEF",   &Jet_neEmEF_ ,"Jet_neEmEF_/F");
	outtree->Branch("Jet_chMult",   &Jet_chMult_ ,"Jet_chMult_/F"); 
	outtree->Branch("Jet_leadTrackPt", &Jet_leadTrackPt_ ,"Jet_leadTrackPt_/F");
	outtree->Branch("Jet_mcEta", &Jet_mcEta_, "Jet_mcEta_/F");
	outtree->Branch("Jet_mcPhi", &Jet_mcPhi_ ,"Jet_mcPhi_/F");
	outtree->Branch("Jet_mcM", &Jet_mcM_ ,"Jet_mcM_/F");
	outtree->Branch("Jet_mcE", &Jet_mcE_ ,"Jet_mcE_/F");
	outtree->Branch("Jet_leptonPt", &Jet_leptonPt_ ,"Jet_leptonPt_/F");
	outtree->Branch("Jet_leptonPtRel", &Jet_leptonPtRel_ ,"Jet_leptonPtRel_/F"); 
	outtree->Branch("Jet_leptonDeltaR", &Jet_leptonDeltaR_ ,"Jet_leptonDeltaR_/F"); 
	outtree->Branch("rho", &rho_ ,"rho_/F");
	outtree->Branch("met_pt",  &met_pt_, "met_pt_/F");
	outtree->Branch("met_phi", &met_phi_, "met_phi_/F");
	outtree->Branch("Jet_met_dPhi", &Jet_met_dPhi_, "Jet_met_dPhi_/F");
	outtree->Branch("Jet_met_proj", &Jet_met_proj_, "Jet_met_proj_/F");  
	float dummy=0.;
	float dummy1=1.;
	for (Long64_t jentry=0; jentry<nentries;jentry++) {
		tree->GetEvent(jentry);
		if(jentry%5000==0) std::cout<<"at entry:"<<jentry<<"/"<<nentries<<std::endl;
		for(int i =0 ; i<nJet; i++){

			if(Jet_mcIdx[i]<0 ) continue;
			if(Jet_mcIdx[i]>15 ) continue;
			TLorentzVector hJ0, hJ1;
			Jet_pt_=Jet_pt[i];
			Jet_rawPt_=Jet_rawPt[i];
			Jet_eta_=Jet_eta[i];
			Jet_phi_=Jet_phi[i];
			Jet_corr_=Jet_corr[i];
			Jet_puId_ = Jet_puId[i];
			//Jet_area_ = Jet_area[i];
			Jet_mass_=Jet_mass[i];
			Jet_met_dPhi_=deltaPhi(met_phi,Jet_phi[i]);
			met_pt_= met_pt;
			met_phi_= met_phi;
			Jet_met_proj_=projectionMETOntoJet(met_pt, met_phi,Jet_pt[i], Jet_phi[i]);
			Jet_leptonPtRel_ = TMath::Max(dummy,Jet_leptonPtRel[i]);
			Jet_leptonDeltaR_ = TMath::Max(dummy,Jet_leptonDeltaR[i]);
			Jet_leptonPt_ = TMath::Max(dummy,Jet_leptonPt[i]);
			Jet_leadTrackPt_ = Jet_leadTrackPt[i];
			Jet_chHEF_ =  TMath::Min(dummy1,Jet_chHEF[i]);
			Jet_chEmEF_=  TMath::Min(dummy1, Jet_chEmEF[i]);
			Jet_neHEF_=  TMath::Min(dummy1,Jet_neHEF[i]);
			Jet_neEmEF_ = TMath::Min(dummy1,Jet_neEmEF[i]);

			hJ0.SetPtEtaPhiM(Jet_pt[i],Jet_eta[i],Jet_phi[i],Jet_mass[i]);
			rho_ = rho;
			nPVs_ = nPVs;
			Jet_e_=hJ0.E();
			Jet_mt_=hJ0.Mt();
			Jet_chMult_=Jet_chMult[i];
			Jet_vtx3dL_=TMath::Max(dummy,Jet_vtx3DVal[i]);
			Jet_vtxMass_=TMath::Max(dummy,Jet_vtxMass[i]);
			Jet_vtx3deL_=TMath::Max(dummy,Jet_vtx3DSig[i]);
			Jet_vtxNtrk=TMath::Max(dummy,Jet_vtxNtracks[i]);
			Jet_vtxPt_=TMath::Max(dummy,Jet_vtxPt[i]);
			Jet_mcPt_= GenJet_wNuPt[Jet_mcIdx[i]];
			dR_ =999;
			Jet_mcPtq_=-999.;
			Jet_mcE_=-999.;
			Jet_mcEta_=-999.;
			Jet_mcPhi_=-999.;
			Jet_mcM_= -999.;
			double minDr = 0.4;
			for(int m=0; m<nGenBQuarkFromTop; m++){
				hJ1.SetPtEtaPhiM(GenBQuarkFromTop_pt[m],GenBQuarkFromTop_eta[m],GenBQuarkFromTop_phi[m],GenBQuarkFromTop_mass[m]);       
				if(hJ1.DeltaR(hJ0)<minDr){

					Jet_mcE_=hJ1.E();
					Jet_mcPtq_ = GenBQuarkFromTop_pt[m];
					Jet_mcEta_=  GenBQuarkFromTop_eta[m];
					Jet_mcPhi_=  GenBQuarkFromTop_phi[m];
					Jet_mcM_= GenBQuarkFromTop_mass[m];
					minDr = hJ1.DeltaR(hJ0);         
					dR_ = minDr;

				}
			}




			Jet_mcFlavour_=Jet_mcFlavour[i];
			Jet_btagCSV_=Jet_btagCSV[i];
			outtree->Fill();
		}
	}
	outtree->Write();
	outfile->Close();
}

