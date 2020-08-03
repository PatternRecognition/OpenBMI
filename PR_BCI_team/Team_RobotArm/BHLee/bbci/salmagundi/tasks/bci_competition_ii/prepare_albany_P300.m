data_dir= [DATA_DIR 'eegImport/bci_competition_ii/albany/P300_data_set/'];

clab= {'FC5','FC3','FC1','FCz','FC2','FC4','FC6', ...
       'C5','C3','C1','Cz','C2','C4','C6', ...
       'CP5','CP3','CP1','CPz','CP2','CP4','CP6', ...
       'Fp1','Fpz','Fp2', 'AF7','AF3','AFz','AF4','AF8', ...
       'F7','F5','F3','F1','Fz','F2','F4','F6','F8', ...
       'FT7','FT8','T7','T8','T9','T10','TP7','TP8', ...
       'P7','P5','P3','P1','Pz','P2','P4','P6','P8', ...
       'PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz'};

clear cnt mrk
cnt.x= zeros(0, 64);
cnt.clab= clab;
cnt.fs= 240;
cnt.title= 'albany P300';
fl= zeros(0, 1);
st_type= zeros(0, 1);
st_code= zeros(0, 1);
ph= zeros(0, 1);
nRuns= [5 6];
word_length= [];
for ss= 10:11,
  for rr= 1:nRuns(ss-9),
    file= sprintf('AAS%03dR%02d', ss, rr),
    load([data_dir file]);
    cnt.x= cat(1, cnt.x, signal);
    
    if ss==11 & rr==6,
      Flashing(5335:5340)= 1;  %% repair artifactual data
    end
    fl= cat(1, fl, Flashing);
    st_type= cat(1, st_type, StimulusType);
    st_code= cat(1, st_code, StimulusCode);
    ph= cat(1, ph, PhaseInSequence);
    word_length= [word_length; trialnr(end)/15/12];
  end
end
pos= find([0; diff(fl)==1]);
mrk.pos= pos';
mrk.toe= 2-st_type(pos)';
mrk.y= [mrk.toe==1; mrk.toe==2];
mrk.fs= cnt.fs;
mrk.className= {'deviant', 'standard'};
mrk.code= st_code(pos)';
mrk.breakPos= find(ismember(ph, [1,3]) & [0; diff(ph)~=0])';
mrk.indexedByEpochs= {'code'};
mrk.wordLength= word_length;

mnt= setElectrodeMontage(cnt.clab);
grd= sprintf('F3,Fz,F4\nC3,Cz,C4\nlegend,Pz,P8');
mnt= setDisplayMontage(mnt, grd);

saveProcessedEEG('bci_competition_ii/albany_P300_train', cnt, mrk, mnt);




clear cnt mrk
cnt.x= zeros(0, 64);
cnt.clab= clab;
cnt.fs= 240;
cnt.title= 'albany P300 test data';
fl= zeros(0, 1);
st_code= zeros(0, 1);
ph= zeros(0, 1);
word_length= zeros(8, 1);
for ss= 12,
  for rr= 1:8,
    file= sprintf('AAS%03dR%02d', ss, rr),
    load([data_dir file]);
    cnt.x= cat(1, cnt.x, signal);
    
    fl= cat(1, fl, Flashing);
    st_code= cat(1, st_code, StimulusCode);
    ph= cat(1, ph, PhaseInSequence);
    word_length(rr)= trialnr(end)/15/12;
  end
end
pos= find([0; diff(fl)==1]);
mrk.pos= pos';
mrk.fs= cnt.fs;
mrk.className= {'deviant', 'standard'};
mrk.code= st_code(pos)';
mrk.breakPos= find(ismember(ph, [1,3]) & [0; diff(ph)~=0])';
mrk.indexedByEpochs= {'code'};
mrk.wordLength= word_length;

saveProcessedEEG('bci_competition_ii/albany_P300_test', cnt, mrk, mnt);
