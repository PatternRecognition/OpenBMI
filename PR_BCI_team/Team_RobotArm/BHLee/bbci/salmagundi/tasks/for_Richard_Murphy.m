subdir= 'VPgcf_11_02_09';
ff= 1;  %% index of condition, see filelist

filelist= {'NoColor116ms', 'Color116ms', 'Color83ms'};
prefix= 'RSVP_';

clab_cfy= {'F3-4','FC5-6','C5-6','CP5-6','P7-8','PO#','O#'};
ival_cfy= [200 250; 250 300; 300 500];

sbj= subdir(1:find(subdir=='_',1,'first')-1);

file= [subdir '/' prefix filelist{ff} sbj];
clear fv
[cnt, mrk]= eegfile_loadMatlab(file);

%% split data in training (calibration) and test (copy- /free-spelling) set
itrain= find(mrk.mode(1,:));
itest= find(any(mrk.mode([2 3],:),1));
mrk_tr= mrk_chooseEvents(mrk, itrain);
mrk_te= mrk_chooseEvents(mrk, itest);

%% process training data 
fv= cntToEpo(cnt, mrk_tr, [-200 ival_cfy(end)], 'clab',clab_cfy);
fv= proc_baseline(fv, [-200 0]);
fv= proc_jumpingMeans(fv, ival_cfy);
C= trainClassifier(fv, 'RLDAshrink');
clear fv

%% process test data
fv= cntToEpo(cnt, mrk_te, [-200 ival_cfy(end)], 'clab',clab_cfy);
clear cnt
fv= proc_baseline(fv, [-200 0]);
fv= proc_jumpingMeans(fv, ival_cfy);

%% evaluate performance of binary (target vs nontarget) classification
% The following line is only required for symbolwise evaluation (see next cell)
[fv, nBlocks, nClasses]= proc_sortWrtStimulus(fv);
out= applyClassifier(fv, 'RLDAshrink', C);
perf= 100-100*mean(loss_classwiseNormalized(fv.y, out));
fprintf('Binary: %s - %s:  %.1f%%.\n', subdir, filelist{ff}, perf);

%% evaluate performance of symbol selection
out_bl= proc_subtrialFeatures(fv, out, nClasses);
yy_hex= [1:nClasses] * out_bl.y(:,1:10:end);
nTests= length(unique(out_bl.trial_idx));
ave= squeeze(mean(reshape(out_bl.x, [nClasses 10 nTests]), 2));
[dmy,ihex]= min(ave);
perf= 100*mean(ihex==yy_hex);
fprintf('Symbol selection: %s - %s:  %.1f%%.\n', subdir, filelist{ff}, perf);
