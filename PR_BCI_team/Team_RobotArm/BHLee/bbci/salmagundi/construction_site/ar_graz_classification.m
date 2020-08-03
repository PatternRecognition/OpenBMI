file= 'graz/graz_f3';
%file= 'graz/graz_f5';
 
[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [3000 6000]);

fprintf('[classification on ar coefficients]\n');
fv= proc_selectIval(epo, [4500 6000]);
fv= proc_arCoefs(fv, 6);

doXvalidation(fv, 'FisherDiscriminant', [5 10]);




laplace.grid= getGrid('graz56');
laplace.filter= [0 -1; 2 0; 0 1; -2 0]';  %% small laplacian
%laplace.filter= [0 -2; 4 0; 0 2; -4 0]';  %% large laplacian
epo_lap= proc_laplace(epo, laplace, '');

fv= proc_selectIval(epo_lap, [4500 6000]);
fv= proc_arCoefs(fv, 6);

doXvalidation(fv, 'FisherDiscriminant', [5 10]);




band= [8 30];
[b,a]= getButterFilter(band, cnt.fs, [3 5]);
epo= makeSegments(cnt, mrk, [0 7990]);
epo_flt= proc_filtfilt(epo, b, a);      %% acausal(!) filtering

fv= proc_selectIval(epo_flt, [4500 6000]);
fv= proc_arCoefs(fv, 6);

doXvalidation(fv, 'FisherDiscriminant', [5 10]);
