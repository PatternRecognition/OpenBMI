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
model= struct('classy','RLDA', 'msDepth',2, 'inflvar',1);
model.param= [0 0.01 0.1 0.5 0.8];



for Subject= {'AA','BB','CC'},
 
subject= Subject{1};

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
C= trainClassifier(fv, classy);

cnt= struct('clab',{clab}, 'fs',160);
for rr= testset_runs,
  file= sprintf('%s%03d', subject, rr),
  load([data_dir file]);
  cnt.x= signal;
  
  if sum(trial==max(trial))<sum(trial==1),  %% delete incomplete last trials
    trial(find(trial==max(trial)))= -1;
  end

  pos= find(diff([-1; trial])>0) + 176;  %% time of feedback presentation
  mrk= struct('pos',pos', 'fs',cnt.fs);
  mrk.className= {'top','upper','lower','bottom'};

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
  nTrials= length(predtargetpos);
  runnr= run(pos);
  trialnr= trial(pos);
  
  save_file= sprintf('%s%03dRES', subject, rr);
  save(save_file, 'runnr', 'trialnr', 'predtargetpos');
  
end


end
