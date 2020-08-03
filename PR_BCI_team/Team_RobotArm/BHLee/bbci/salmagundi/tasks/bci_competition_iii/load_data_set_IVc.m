bcdir= [DATA_DIR 'eegImport/bci_competition_iii/'];

train_file= 'berlin/100Hz/data_set_IVb_al_train';
test_file= 'berlin/100Hz/data_set_IVb_al_test';

load([bcdir train_file]);

cnt= struct('x',double(cnt)/10, 'clab',{nfo.clab}, 'fs',nfo.fs, ...
            'name',nfo.name);
mrk.y= [mrk.y==1; mrk.y==2];
mrk.className= {'left', 'foot'};
mnt= projectElectrodePositions(cnt.clab);

idxTr= 1:length(mrk.pos);

S= load([bcdir test_file]);
