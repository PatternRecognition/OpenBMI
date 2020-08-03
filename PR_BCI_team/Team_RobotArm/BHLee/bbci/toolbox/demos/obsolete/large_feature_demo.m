file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [-1200 -120]);

fv= proc_baseline(epo, [-1200 -800]);
fv= proc_selectIval(fv, [-150 0]-120);
fv= proc_selectChannels(fv, 'FC#', 'C#', 'CP#');

nTrials= [10 10];
msTrials= [3 10 round(9/10*size(epo.y,2))];
if ~exist('lpenv','var'),  lpenv= cplex_init(1); end

pClassy= {'FDlwqx', lpenv}
param= struct('index',3, 'scale','log', 'value',0:0.25:2);
model= {pClassy, param};
classy= selectModel(fv, msTrials, model{:});
doXvalidation(fv, classy, nTrials);
