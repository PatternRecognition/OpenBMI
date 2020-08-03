%subject= 'AA'; 
%subject= 'BB'; 
%subject= 'CC'; 
sub_dir= 'bci_competition_ii/';
cd([BCI_DIR 'tasks/' sub_dir]);
if ~exist('subject','var'), error('please define subject'); end 
model= struct('classy','RLDA', 'msDepth',2, 'inflvar',1);
model.param= [0 0.01 0.1 0.5 0.8];

load(['albany_csp_' subject]);
fprintf('%s:  band1= [%d %d],  band2= [%d %d],  csp_ival= [%d %d]\n', ...
        subject, dscr_band(1,:), dscr_band(2,:), csp_ival);
file= sprintf('%salbany_%s_train', sub_dir, subject);

[cnt, Mrk, mnt]= loadProcessedEEG(file);
mrk= mrk_selectClasses(Mrk, {'top','bottom'});

cnt= proc_linearDerivation(cnt, csp_w);
cnt.clab= csp_clab;
classy= 'LDA';


epo= makeEpochs(cnt, mrk, [1000 4000]);
%epo= makeEpochs(cnt, Mrk, [1000 4000]);


fv1= proc_fourierCourseOfBandEnergy(epo, dscr_band(1,:), 160, 80);
fv2= proc_fourierCourseOfBandEnergy(epo, dscr_band(2,:), 160, 80);
fv= proc_catFeatures(fv1, fv2);
doXvalidationPlus(fv1, classy, [5 10]);
doXvalidationPlus(fv2, classy, [5 10]);
doXvalidationPlus(fv, classy, [5 10]);
%% top vs bot:  8.5%  -> 8% mit RLDA
%% all classes: 30%



fv1= proc_selectChannels(epo, 'band1*');
fv1= proc_fourierCourseOfBandMagnitude(fv1, dscr_band(1,:), 160, 80);
fv2= proc_selectChannels(epo, 'band2*');
fv2= proc_fourierCourseOfBandMagnitude(fv2, dscr_band(2,:), 160, 80);
fv= proc_catFeatures(fv1, fv2);

doXvalidationPlus(fv1, classy, [5 10]);
%% top vs bot: 5%,  all classes: 26%
doXvalidationPlus(fv2, classy, [5 10]);
cl= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, cl, [5 10]);
%% top vs bot: 4%
%% all classes: 22%



fv1= proc_fourierCourseOfBandMagnitude(epo, dscr_band(1,:), 160, 80);
fv2= proc_fourierCourseOfBandMagnitude(epo, dscr_band(2,:), 160, 80);
fv= proc_catFeatures(fv1, fv2);

doXvalidationPlus(fv1, classy, [5 10]);
%% top vs bot: 5%,  all classes: 26%
doXvalidationPlus(fv2, classy, [5 10]);
cl= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, cl, [5 10]);
%% top vs bot:  4%  (5.5% ohne Reg.)
%% all classes: 24%






epo= makeEpochs(cnt, mrk, [1000 3000]);
%epo= makeEpochs(cnt, Mrk, [1000 3000]);

fv1= proc_fourierBandMagnitude(epo, dscr_band(1,:), 320);
fv1= proc_jumpingMeans(fv1, 2);
fv2= proc_fourierBandMagnitude(epo, dscr_band(2,:), 320);
fv2= proc_jumpingMeans(fv2, 2);
fv= proc_catFeatures(fv1, fv2);

doXvalidationPlus(fv1, classy, [5 10]);
%% top vs bot: 5%,  all classes: 33%
doXvalidationPlus(fv2, classy, [5 10]);
doXvalidationPlus(fv, classy, [5 10]);
%% top vs bot:  5%
%% all classes: 31%



cl= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, cl, [5 10]);
%% top vs bot:  5%
%% all classes: 31%



classy= {'multiClass', 'all-pairs', 'hamming', 'LDA'};

[em,es,out]= doXvalidationPlus(fv, cl, [5 10], 1);
[mc, me, ms]= calcConfusionMatrix(fv, out)
imagesc(me)
