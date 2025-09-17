#include <vector>
#include "ROOT/RDataFrame.hxx"
#include <stdio.h>
#include <iostream>
#include "TSystem.h"

void SHMSkim(int run)
{
    ROOT::RDataFrame df("T", TString::Format("/cache/hallc/xem2/analysis/OFFLINE/REPLAYS/SHMS/PRODUCTION/pass2/shms_replay_production_%d_-1.root", run).Data());

    auto clean = df.Filter("P.cal.etottracknorm>0.5").Filter("abs(P.gtr.dp)<30").Filter("P.ngcer.npeSum>0").Filter("P.gtr.index>-1").Filter("abs(P.gtr.th)<0.1").Filter("abs(P.gtr.ph)<0.1");

    clean.Snapshot("T", TString::Format("SHMS_skim_%d.root", run).Data(), {"P.gtr.x", "P.gtr.y", "P.gtr.dp", "P.gtr.p", "P.gtr.ph", "P.gtr.th", "P.gtr.index", "P.kin.x_bj", "P.kin.Q2", "P.kin.nu", "P.react.x", "P.react.y", "P.react.z", "P.dc.x_fp", "P.dc.y_fp", "P.dc.xp_fp", "P.dc.yp_fp", "P.ngcer.npeSum", "P.cal.etottracknorm", "P.rb.raster.frxaRawAdc", "ibcm1", "ibcm2"});
}
