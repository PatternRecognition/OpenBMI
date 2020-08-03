sub_dir= 'bci_competition_ii/';
subject= 'AA';
%subject= 'BB';
%subject= 'CC';
xTrials= [5 10];

file= sprintf('%salbany_%s_train', sub_dir, subject);
[cnt, mrk, mnt]= loadProcessedEEG(file);

mrk= mrk_selectClasses(mrk, {'top','bottom'});
epo= makeEpochs(cnt, mrk, [-1000 4500]);


%% with common spatial patterns
band= [10 14]; 
csp_ival= [1000 3000];
[b,a]= getButterFixedOrder(band, epo.fs, 6);
epo_flt= proc_filt(epo, b, a);

fv= proc_selectIval(epo_flt, csp_ival);
fv.proc=['fv= proc_csp(epo, 2); ' ...
         'fv= proc_variance(fv); '];
doXvalidationPlus(fv, 'LDA', xTrials);
%% 7.5%



%% with band power
band= [11 13.5];
%band= [23 27];
fv= proc_selectIval(epo, [1000 3000]);
fv= proc_commonAverageReference(fv);
fv= proc_selectChannels(fv, 'FC3,4','C5-1','C2-6','CP5-6','P5,3,4,6');
fv= proc_fourierBandEnergy(fv, band, 320);
doXvalidation(fv, 'LDA', xTrials);
%% 9%



%% with power spectrum
band= [10 13.5];
%band= [7 35];
fv= proc_selectIval(epo, [1000 3000]);
fv= proc_commonAverageReference(fv);
fv= proc_selectChannels(fv, 'FC3,4','C5-1','C2-6','CP5-6','P5,3,4,6');
fv= proc_fourierBandMagnitude(fv, band, 320);
fv= proc_jumpingMeans(fv, 2);
doXvalidation(fv, 'LDA', xTrials);

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.01 0.1 0.5 0.75];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidation(fv, classy, xTrials);
%doXvalidationPlus(fv, model, xTrials); %% the real thing
%% 5.5%



%% with AR coefficients
band= [5 30]; 
[b,a]= getButterFixedOrder(band, epo.fs, 6);
%fv= proc_laplace(epo);
fv= proc_commonAverageReference(epo);
fv= proc_filtfilt(fv, b, a);
fv= proc_selectIval(fv, [1000 3000]);
fv= proc_selectChannels(fv, 'C5,3,4,6','CP5-6','P5,3,4,6');
fv= proc_arCoefsPlusVar(fv, 6);
doXvalidation(fv, 'LDA', xTrials);

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.01 0.1 0.5 0.75];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidation(fv, classy, xTrials);
