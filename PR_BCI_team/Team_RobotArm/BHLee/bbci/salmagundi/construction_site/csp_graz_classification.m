file= 'graz/graz_f3';
%file= 'graz/graz_f5';
 
[cnt, mrk, mnt]= loadProcessedEEG(file);
band= [8 30];
[b,a]= getButterFilter(band, cnt.fs, [3 5]);
epo= makeSegments(cnt, mrk, [0 7990]);
epo_flt= proc_filtfilt(epo, b, a);      %% acausal(!) filtering

fprintf('calculating csp on all trials (cheating)\n');
fv= proc_selectIval(epo_flt, [4500 6000]);
fv= proc_csp(fv, 1);
fv= proc_variance(fv);
fv= proc_logNormalize(fv);

doXvalidation(fv, 'FisherDiscriminant', [5 10]);


fprintf('calculating csp on training set only\n');
fv= proc_selectIval(epo_flt, [4500 6000]);
fv.proc=['fv= proc_csp(epo, 1); ' ...
         'fv= proc_variance(fv); ' ...
         'fv= proc_logNormalize(fv); '];
doXvalidationPlus(fv, 'FisherDiscriminant', [5 10]);




laplace.grid= getGrid('graz56');
laplace.filter= [0 -2; 4 0; 0 2; -4 0]';  %% large laplacian
%laplace.filter= [0 -1; 2 0; 0 1; -2 0]';  %% small laplacian
epo_lap= proc_laplace(epo_flt, laplace, '');

fv= proc_selectIval(epo_lap, [4500 6000]);
fv.proc=['fv= proc_csp(epo, 1); ' ...
         'fv= proc_variance(fv); ' ...
         'fv= proc_logNormalize(fv); '];
doXvalidationPlus(fv, 'FisherDiscriminant', [5 10]);
