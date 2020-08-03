%% get labels for test set
load([BCI_DIR 'tasks/bci_competition_ii/our_export']);
y_truth= mrk.y(:,test_idx);

%% load competition data set
file= 'berlin/sp1s_aa';
load([EEG_IMPORT_DIR 'bci_competition_ii/' file]);
train_idx= 1:size(x_train,3);
test_idx= length(train_idx) + [1:size(x_test,3)];

%% transform data to epo structure
epo= struct('file', file);
epo.x= cat(3, x_train, x_test);
nEpochs= size(epo.x,3);
epo.y= cat(2, [y_train==0; y_train==1], y_truth);
epo.fs= 100;
epo.t= linspace(-620, -130, size(epo.x,1));

%% do classification
fv= proc_filtBruteFFT(epo, [0.8 3], 128, 150);
fv= proc_jumpingMeans(fv, 5);

C= trainClassifier(fv, 'LDA', train_idx);
out= applyClassifier(fv, 'LDA', C, test_idx);

err= 100*mean(sign(out)~=[-1 1]*fv.y(:,test_idx));
fprintf('LDA: %2.1f%%\n', err);

fv_train= proc_selectEpochs(fv, train_idx);
model= struct('classy','RLDA', 'msDepth',3, 'inflvar',1);
model.param= [0 0.0005 0.001 0.005 0.01 0.05];
classy= selectModel(fv_train, model, [10 10]);
C= trainClassifier(fv, classy, train_idx);
out= applyClassifier(fv, 'RLDA', C, test_idx);

err= 100*mean(sign(out)~=[-1 1]*fv.y(:,test_idx));
fprintf('RLDA: %2.1f%%\n', err);
