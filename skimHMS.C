#include <vector>
#include "ROOT/RDataFrame.hxx"
#include <stdio.h>
#include <iostream>
#include "TSystem.h"

void skimHMS(int run)
{
    ROOT::RDataFrame df("T", TString::Format("/cache/hallc/xem2/analysis/OFFLINE/REPLAYS/HMS/PRODUCTION/pass2/hms_replay_production_%d_-1.root", run).Data());

    auto clean = df.Filter("H.cal.etottracknorm>0.5").Filter("abs(H.gtr.dp)<20").Filter("H.cer.npeSum>0").Filter("H.gtr.index>-1").Filter("abs(H.gtr.th)<0.1").Filter("abs(H.gtr.ph)<0.1");

    clean.Snapshot("T", TString::Format("HMS_skim_%d.root", run).Data(), {"H.gtr.x", "H.gtr.y", "H.gtr.dp", "H.gtr.p", "H.gtr.ph", "H.gtr.th", "H.gtr.index", "H.kin.x_bj", "H.kin.Q2", "H.kin.nu", "H.dc.x_fp", "H.dc.y_fp", "H.dc.xp_fp", "H.dc.yp_fp", "H.cer.npeSum", "H.cal.etottracknorm", "H.rb.raster.frxaRawAdc", "ibcm1", "ibcm2"});
}
