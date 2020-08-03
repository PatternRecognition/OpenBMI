file= 'graz/graz_f3';
%file= 'graz/graz_f5';
 
[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [3000 6000]);

fprintf('[classification on aar coefficients]\n');

laplace.grid= getGrid('graz56');
laplace.filter= [0 -2; 4 0; 0 2; -4 0]';  %% large laplacian
%laplace.filter= [0 -1; 2 0; 0 1; -2 0]';  %% small laplacian
epo_lap= proc_laplace(epo, laplace, '');

fv= proc_selectIval(epo_lap, [4500 6000]);
fv= proc_aar_graz(fv, 6, 0, 0);
fv.x= squeeze(fv.x(:,end,:,:));

doXvalidation(fv, 'FisherDiscriminant', [5 10]);
