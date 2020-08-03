sub_dir= 'bci_competition_ii/';
cd([BCI_DIR 'tasks/' sub_dir]);
classy= 'LDA';
model= struct('classy','RLDA', 'msDepth',2, 'inflvar',1);
model.param= [0 0.01 0.1 0.5 0.8];


for Subject= {'AA', 'BB', 'CC'},
 
subject= Subject{1};

load(['albany_csp_' subject]);
fprintf('%s:  band1= [%d %d],  band2= [%d %d],  csp_ival= [%d %d]\n', ...
        subject, dscr_band(1,:), dscr_band(2,:), csp_ival);
file= sprintf('%salbany_%s_train', sub_dir, subject);

[cnt, Mrk, mnt]= loadProcessedEEG(file);

cnt= proc_linearDerivation(cnt, csp_w);
cnt.clab= csp_clab;

mrk= mrk_selectClasses(Mrk, {'top','bottom'});
epo= makeEpochs(cnt, Mrk, [1000 4000]);
%epo= makeEpochs(cnt, mrk, [1000 4000]);  %% 2 classes only

fv1= proc_selectChannels(epo, 'band1*');
fv1= proc_fourierCourseOfBandMagnitude(fv1, dscr_band(1,:), 160, 80);
fv2= proc_selectChannels(epo, 'band2*');
fv2= proc_fourierCourseOfBandMagnitude(fv2, dscr_band(2,:), 160, 80);
fv= proc_catFeatures(fv1, fv2);

%doXvalidationPlus(fv1, classy, [5 10]);
cl= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))], 0);
doXvalidationPlus(fv, cl, [5 10]);

end
