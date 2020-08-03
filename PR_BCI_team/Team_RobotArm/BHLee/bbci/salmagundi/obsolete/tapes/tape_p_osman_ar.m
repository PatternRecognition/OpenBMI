fv= proc_selectChannels(epo, 'C5-1','C2-6', 'CP3','CP4');
fv= proc_selectIval(fv, [500 2000]);
fv= proc_arCoefs(fv, 5);

fv= proc_selectChannels(epo, 'C5-1','C2-6', 'CP3','CP4');
fv= proc_selectIval(fv, [500 2000]);
fv= proc_arCoefs(fv, 6);

fv= proc_selectChannels(epo, 'C5-1','C2-6', 'CP3','CP4');
fv= proc_selectIval(fv, [500 2000]);
fv= proc_arCoefs(fv, 7);

fv= proc_selectChannels(epo, 'C5-1','C2-6', 'CP3','CP4');
fv= proc_selectIval(fv, [500 2000]);
fv= proc_arCoefs(fv, 9);

fv= proc_selectChannels(epo, 'C3','C4', 'CP3','CP4');
fv= proc_selectIval(fv, [500 2000]);
fv= proc_arCoefs(fv, 8);

fv= proc_selectChannels(epo, 'C5-1','C2-6', 'CP3','CP4');
fv= proc_selectIval(fv, [500 2000]);
fv= proc_arCoefs(fv, 8);

fv= proc_selectChannels(epo, 'C5-1','C2-6', 'CP3','CP4');
fv= proc_selectIval(fv, [500 1500]);
fv= proc_arCoefs(fv, 8);

fv= proc_selectChannels(epo, 'C5-1','C2-6', 'CP3','CP4');
fv= proc_selectIval(fv, [300 2200]);
fv= proc_arCoefs(fv, 8);

laplace.grid= getGrid('visible_128');
laplace.filter= 'small';
epo_lap= proc_laplace(epo, laplace);
fv= proc_selectChannels(epo_lap, 'C3','C4');
fv= proc_selectIval(fv, [500 2000]);
fv= proc_arCoefs(fv, 5);

laplace.grid= getGrid('visible_128');
laplace.filter= 'small';
epo_lap= proc_laplace(epo, laplace);
fv= proc_selectChannels(epo_lap, 'C3','C4');
fv= proc_selectIval(fv, [500 2000]);
fv= proc_arCoefs(fv, 6);

laplace.grid= getGrid('visible_128');
laplace.filter= 'small';
epo_lap= proc_laplace(epo, laplace);
fv= proc_selectChannels(epo_lap, 'C3-1','C2-4', 'CP3','CP4');
fv= proc_selectIval(fv, [500 2000]);
fv= proc_arCoefs(fv, 5);
