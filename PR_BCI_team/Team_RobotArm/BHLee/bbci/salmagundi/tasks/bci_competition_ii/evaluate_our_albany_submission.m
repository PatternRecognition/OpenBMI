data_dir= [DATA_DIR 'eegImport/bci_competition_ii/albany/'];

clab= {'FC5','FC3','FC1','FCz','FC2','FC4','FC6', ...
       'C5','C3','C1','Cz','C2','C4','C6', ...
       'CP5','CP3','CP1','CPz','CP2','CP4','CP6', ...
       'Fp1','Fpz','Fp2', 'AF7','AF3','AFz','AF4','AF8', ...
       'F7','F5','F3','F1','Fz','F2','F4','F6','F8', ...
       'FT7','FT8','T7','T8','T9','T10','TP7','TP8', ...
       'P7','P5','P3','P1','Pz','P2','P4','P6','P8', ...
       'PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz'};
testset_runs= 7:10;

sub_dir= 'bci_competition_ii/';
cd([BCI_DIR 'tasks/' sub_dir]);
classy= 'LDA';
model= struct('classy','RLDA', 'msDepth',3, 'inflvar',1);
model.param= [0 0.01 0.1 0.5 0.8];



subject_list=  {'AA','BB','CC'};

for ss= 1:length(subject_list),
 
subject= subject_list{ss};

load(['albany_csp_' subject]);
fprintf('%s:  band1= [%d %d],  band2= [%d %d],  csp_ival= [%d %d]\n', ...
        subject, dscr_band(1,:), dscr_band(2,:), csp_ival);
file= sprintf('%salbany_%s_train', sub_dir, subject);

[cnt, Mrk, mnt]= loadProcessedEEG(file);

cnt= proc_linearDerivation(cnt, csp_w);
cnt.clab= csp_clab;

epo= makeEpochs(cnt, Mrk, [1000 4000]);

fv1= proc_selectChannels(epo, 'band1*');
fv1= proc_fourierCourseOfBandMagnitude(fv1, dscr_band(1,:), 160, 80);
fv2= proc_selectChannels(epo, 'band2*');
fv2= proc_fourierCourseOfBandMagnitude(fv2, dscr_band(2,:), 160, 80);
fv= proc_catFeatures(fv1, fv2);

classy= selectModel(fv, model, [10 10]);
err= doXvalidationPlus(fv, classy, [10 10]);
err_xval(ss)= err(1);
C= trainClassifier(fv, classy);

test_file=  sprintf('%salbany_%s_test_withLabels', sub_dir, subject);
[cnt, mrk, mnt]= loadProcessedEEG(test_file);
cnt= proc_linearDerivation(cnt, csp_w);
cnt.clab= csp_clab;

epo= makeEpochs(cnt, mrk, [1000 4000]);
fv1= proc_selectChannels(epo, 'band1*');
fv1= proc_fourierCourseOfBandMagnitude(fv1, dscr_band(1,:), 160, 80);
fv2= proc_selectChannels(epo, 'band2*');
fv2= proc_fourierCourseOfBandMagnitude(fv2, dscr_band(2,:), 160, 80);
fv= proc_catFeatures(fv1, fv2);

out= applyClassifier(fv, classy, C);
[dummy, predtargetpos]= max(out);
[dummy, truetargetpos]= max(mrk.y);
valid= find(any(mrk.y));
predtargetpos= predtargetpos(valid);
truetargetpos= truetargetpos(valid);
albanypred= mrk.albany_pred(valid);
err_berlin(ss)= 100*mean(predtargetpos~=truetargetpos);
err_albany(ss)= 100*mean(albanypred~=truetargetpos);

fprintf('%s> berlin: %5.1f%%  [xval: %5.1f%%], albany: %5.1f%%\n', ...
        subject, err_berlin(ss), err_xval(ss), err_albany(ss));

end

for ss= 1:length(subject_list),
  fprintf('%s:4classes> berlin: %5.1f%%  [xval: %5.1f%%], albany: %5.1f%%\n', ...
        subject_list{ss}, err_berlin(ss), err_xval(ss), err_albany(ss));

end




subject_list=  {'AA','BB','CC'};

for ss= 1:length(subject_list),
 
subject= subject_list{ss};

load(['albany_csp_' subject]);
fprintf('%s:  band1= [%d %d],  band2= [%d %d],  csp_ival= [%d %d]\n', ...
        subject, dscr_band(1,:), dscr_band(2,:), csp_ival);
file= sprintf('%salbany_%s_train', sub_dir, subject);

[cnt, Mrk, mnt]= loadProcessedEEG(file);

cnt= proc_linearDerivation(cnt, csp_w);
cnt.clab= csp_clab;

mrk= mrk_selectClasses(Mrk, {'top','bottom'});
epo= makeEpochs(cnt, mrk, [1000 4000]);

fv1= proc_selectChannels(epo, 'band1*');
fv1= proc_fourierCourseOfBandMagnitude(fv1, dscr_band(1,:), 160, 80);
fv2= proc_selectChannels(epo, 'band2*');
fv2= proc_fourierCourseOfBandMagnitude(fv2, dscr_band(2,:), 160, 80);
fv= proc_catFeatures(fv1, fv2);

classy= selectModel(fv, model, [10 10]);
err= doXvalidationPlus(fv, classy, [10 10]);
err_xval(ss)= err(1);
C= trainClassifier(fv, classy);

test_file=  sprintf('%salbany_%s_test_withLabels', sub_dir, subject);
[cnt, Mrk, mnt]= loadProcessedEEG(test_file);
cnt= proc_linearDerivation(cnt, csp_w);
cnt.clab= csp_clab;

[mrk, tb_ev]= mrk_selectClasses(Mrk, {'top','bottom'});
epo= makeEpochs(cnt, mrk, [1000 4000]);
fv1= proc_selectChannels(epo, 'band1*');
fv1= proc_fourierCourseOfBandMagnitude(fv1, dscr_band(1,:), 160, 80);
fv2= proc_selectChannels(epo, 'band2*');
fv2= proc_fourierCourseOfBandMagnitude(fv2, dscr_band(2,:), 160, 80);
fv= proc_catFeatures(fv1, fv2);

out= applyClassifier(fv, classy, C);
if size(out,1)==1,
  predtargetpos= sign(out)/2 + 1.5;
else
  [dummy, predtargetpos]= max(out);
end
valid= find(any(mrk.y));
predtargetpos= predtargetpos(valid);
[dummy, truetargetpos]= max(mrk.y(:,valid));
albanypred= mrk.albany_pred(tb_ev(valid));
albanypred(find(ismember(albanypred,[3 4])))= 4;
albanypred(find(ismember(albanypred,[1 2])))= 1;
albanypred(find(albanypred==4))= 2;
err_berlin(ss)= 100*mean(predtargetpos~=truetargetpos);
err_albany(ss)= 100*mean(albanypred~=truetargetpos);

fprintf('%s:T vs B> berlin: %5.1f%%  [xval: %5.1f%%], albany: %5.1f%%\n', ...
        subject, err_berlin(ss), err_xval(ss), err_albany(ss));

end


for ss= 1:length(subject_list),
  fprintf('%s:T vs B> berlin: %5.1f%%  [xval: %5.1f%%], albany: %5.1f%%\n', ...
        subject_list{ss}, err_berlin(ss), err_xval(ss), err_albany(ss));

end
