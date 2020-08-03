paceStr= '1s';
ival= [-490 0]-130;
export_dir= '/home/tensor/blanker/Daten/eegImport/bci_competition_ii/berlin/';

%% load data
file= ['Gabriel_01_07_24/selfpaced' paceStr 'Gabriel'];
[dummy, mrk]= loadProcessedEEG(file, [], 'mrk');

cd([BCI_DIR 'tasks/bci_competition_ii']);

if exist('our_export.mat', 'file'),
  load('our_export');

else
  %% select equilibrated subset and make epochs
  pairs= getEventPairs(mrk, paceStr);
  equi= equiSubset(pairs);
  mrk= pickEvents(mrk, [equi{:}]);

  %% split randomly in test and training data
  nEvents= length(mrk.pos);
  perm_idx= randperm(nEvents);
  test_idx= perm_idx(1:100);
  train_idx= perm_idx(101:end);
  
  save('our_export', 'equi', 'mrk', 'perm_idx', 'train_idx', 'test_idx');
end

cnt= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'not','E*');

mrk_tr= pickEvents(mrk, train_idx);
epo_tr= makeSegments(cnt, mrk_tr, ival);
mrk_te= pickEvents(mrk, test_idx);
epo_te= makeSegments(cnt, mrk_te, ival);

clab= epo.clab;
x_train= epo_tr.x;
y_train= [0 1]*epo_tr.y;
x_test= epo_te.x;

save([export_dir 'sp1s_aa'], 'clab','x_train','y_train','x_test');

[T, nChans, nEvents]= size(x_train);
X= [y_train; reshape(x_train, [T*nChans nEvents])]';
save([export_dir 'sp1s_aa_train.txt'], '-ascii', 'X');
nEvents= size(x_test, 3);
X= reshape(x_test, [T*nChans nEvents])';
save([export_dir 'sp1s_aa_test.txt'], '-ascii', 'X');



%% export 1000 Hz data
ival= [-499 0]-130;

cnt= readGenericEEG(file, [], 1000);
cnt= proc_selectChannels(cnt, 'not','E*');
mrk= readMarkerTable(file, cnt.fs);
classDef= {[65 70],[74 192]; 'left','right'};
mrk= makeClassMarkers(mrk, classDef);
mrk= pickEvents(mrk, [equi{:}]);
mrk_tr= pickEvents(mrk, train_idx);
epo_tr= makeSegments(cnt, mrk_tr, ival);
mrk_te= pickEvents(mrk, test_idx);
epo_te= makeSegments(cnt, mrk_te, ival);

clab= epo.clab;
x_train= epo_tr.x;
y_train= [0 1]*epo_tr.y;
x_test= epo_te.x;

save([export_dir 'sp1s_aa_1000Hz'], 'clab','x_train','y_train','x_test');

[T, nChans, nEvents]= size(x_train);
X= [y_train; reshape(x_train, [T*nChans nEvents])]';
save([export_dir 'sp1s_aa_train_1000Hz.txt'], '-ascii', 'X');
nEvents= size(x_test, 3);
X= reshape(x_test, [T*nChans nEvents])';
save([export_dir 'sp1s_aa_test_1000Hz.txt'], '-ascii', 'X');
