file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
motoJits= [0, -40, -80, -120, -160];
epo= makeEpochs(cnt, mrk, [-1300 0], motoJits);
nMotos= size(epo.y, 2);
no_moto= makeEpochs(cnt, mrk, [-1300 0]+1200, [0, -60, -120, -180, -240]);
%no_moto= makeEpochs(cnt, mrk, [-1300 0]-800, [0, -80, -160, -240]);
noMotos= size(no_moto.y, 2);
epo= proc_appendEpochs(epo, no_moto);



%% via two classes
epo.className= {'motor', 'no motor'};
epo.y= [repmat([1;0], 1, nMotos) repmat([0;1], 1, noMotos)];

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 200);
fv= proc_jumpingMeans(fv, 5);

nTrials= [10 10];
doXvalidationPlus(fv, 'QDA', nTrials);

doXvalidationPlus(fv, 'LDA', nTrials);

%% via three classes
epo.y= zeros(3, size(epo.y,2));                 %% labels for l/r motor events
epo.y(1:2,1:nMotos)= repmat(mrk.y, [1 length(motoJits)]);  
epo.y(3,nMotos+1:end)= 1;                       %% labels for no motor events
epo.className= {'left','right','none'};

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 200);
fv= proc_jumpingMeans(fv, 5);

fv.loss= [0 0 1; 0 0 1; 1 1 0];    %% loss just for confusing motor / no_motor
doXvalidationPlus(fv, 'QDA', nTrials);

doXvalidationPlus(fv, 'LDA', nTrials);


% and now with combineClasses, epo contains three class setup
fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 200);
fv= proc_jumpingMeans(fv, 5);


% the following order combines classes!!! Therefore fv is two class
% after it, train_combineClasses can extend this afterwards in
% three classes and combine it in probability by given the order
% 'exp' as next argument
fv = proc_combineClasses(fv,'left','right');
doXvalidationPlus(fv, {'combineClasses','exp'}, nTrials);


model.classy = {'combineClasses','exp','RLDA'};
model.param.index = 4;
model.param.value = (0:0.2:1).^3;
model.msDepth = 5;
classy = selectModel(fv,model,[3 10 round(0.9*size(fv.y,2))])
doXvalidationPlus(fv, classy, nTrials);

model.classy = {'combineClasses','exp','RDA'};
model.param(1).index = 4;
model.param(1).value = 0:0.2:1;
model.param(2).index = 5;
model.param(2).value = (0:0.2:1).^3;
model.msDepth = 5;
classy = selectModel(fv,model,[3 10 round(0.9*size(fv.y,2))])
doXvalidationPlus(fv, classy, nTrials);



% via four classes
epo= makeEpochs(cnt, mrk, [-1300 0], motoJits);
nMotos= size(epo.y, 2);
no_moto= makeEpochs(cnt, mrk, [-1300 0]+1200, [0, -60, -120, -180, -240]);
%no_moto= makeEpochs(cnt, mrk, [-1300 0]-800, [0, -80, -160, -240]);
noMotos= size(no_moto.y, 2);
no_moto.className = {'no_moto after left','no_moto after right'};
epo= proc_appendEpochs(epo, no_moto);



fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 200);
fv= proc_jumpingMeans(fv, 5);
fv.loss= [0 0 1 1; 0 0 1 1; 1 1 0 0; 1 1 0 0];    

doXvalidationPlus(fv, 'QDA', nTrials);

doXvalidationPlus(fv, 'LDA', nTrials);



fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 200);
fv= proc_jumpingMeans(fv, 5);

fv = proc_combineClasses(fv,{1,2},{3,4});
doXvalidationPlus(fv, {'combineClasses','exp'}, nTrials);




% now we forget what we know about the classes and do it with a
% mixture of Gaussian

epo.className= {'motor', 'no motor'};
epo.y= [repmat([1;0], 1, nMotos) repmat([0;1], 1, noMotos)];

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 200);
fv= proc_jumpingMeans(fv, 5);


% the second argument describes the number of clusters for each class
classy = {'mixGauss',[2,1]};

% mixGauss is very slow:
nTrials = [3 10];

doXvalidationPlus(fv,classy,nTrials);

% regularisation works like RDA (the following for example is RLDA)
model.classy = {'mixGauss',[2,1],1};
model.param.value = (0:0.25:1).^3;
model.param.index = 4;
model.msDepth = 3;
classy = selectModel(fv,model,[3 10 round(0.9*size(fv.y,2))]);
doXvalidationPlus(fv,classy,nTrials);


% alternatively (4 classes)
classy = {'mixGauss',2};

% mixGauss is very slow:
nTrials = [3 10];

doXvalidationPlus(fv,classy,nTrials);

% regularisation works like RDA 
model.classy = {'mixGauss',[2,1]};
model.param(1).value = (0:0.25:1).^3;
model.param(1).index = 3;
model.param(2).value = (0:0.25:1).^3;
model.param(2).index = 4;
model.msDepth = 3;
classy = selectModel(fv,model,[3 10 round(0.9*size(fv.y,2))]);
doXvalidationPlus(fv,classy,nTrials);


