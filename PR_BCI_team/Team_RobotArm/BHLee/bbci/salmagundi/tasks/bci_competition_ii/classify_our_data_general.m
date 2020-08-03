paceStr= '1s';
ival= [-490 0]-130;

%% load data
file= ['Gabriel_01_07_24/selfpaced' paceStr 'Gabriel'];
[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'not','E*');

%% select equilibrated subset and make epochs
pairs= getEventPairs(mrk, paceStr);
equi= equiSubset(pairs);
mrk= pickEvents(mrk, [equi{:}]);
epo= makeSegments(cnt, mrk, ival);

%% split randomly in test and training data
nEvents= length(mrk.pos);
perm_idx= randperm(nEvents);
test_idx= perm_idx(1:100);
train_idx= perm_idx(101:end);


fv= proc_filtBruteFFT(epo, [0.8 3], 128, 150);
fv= proc_jumpingMeans(fv, 5);

C= trainClassifier(fv, 'LDA', train_idx);
out= applyClassifier(fv, 'LDA', C, test_idx);

err= 100*mean(sign(out)~=[-1 1]*fv.y(:,test_idx));
fprintf('LDA: %2.1f%%\n', err);


model= struct('classy','RLDA', 'msDepth',3, 'inflvar',1);
model.param= [0 0.0005 0.001 0.005 0.01 0.05];
classy= selectModel(fv, model, [10 10]);
C= trainClassifier(fv, classy, train_idx);
out= applyClassifier(fv, 'LDA', C, test_idx);

err= 100*mean(sign(out)~=[-1 1]*fv.y(:,test_idx));
fprintf('RLDA: %2.1f%%\n', err);
