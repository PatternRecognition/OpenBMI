fv= proc_baseline(epo, [-1000 -800]);
fv= proc_laplace(fv, 'small', '', 'CP#');
fv= proc_bipolarChannels(fv, {'C3-C2','CP3-CP2'});
fv= proc_selectIval(fv, [-170 -120]);

fv= proc_baseline(epo, [-1000 -800]);
fv= proc_laplace(fv, 'small', '', 'CP#');
fv= proc_bipolarChannels(fv, {'C3-C2','CP3-CP2'});
fv= proc_selectIval(fv, [-200 -120]);

fv= proc_baseline(epo, [-1000 -800]);
fv= proc_bipolarChannels(fv, {'C3-C2','CP3-CP2'});
fv= proc_selectIval(fv, [-200 -120]);

fv= proc_baseline(epo, [-1000 -800]);
fv= proc_laplace(fv, 'small', '', 'CP#');
fv= proc_bipolarChannels(fv, {'C3-C2','CP3-CP2'});
fv= proc_selectIval(fv, [-220 -120]);

fv= proc_baseline(epo, [-1000 -800]);
fv= proc_laplace(fv, 'small', '', 'CP#');
fv= proc_bipolarChannels(fv, {'C3-C2','CP3-CP2','FC3-FC2'});
fv= proc_selectIval(fv, [-200 -120]);

fv= proc_baseline(epo, [-1000 -800]);
fv= proc_laplace(fv, 'small', '', 'CP#');
fv= proc_selectChannels(fv, {'C3','C2','CP3','CP2'});
fv= proc_selectIval(fv, [-200 -120]);
