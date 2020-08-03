file= 'Gabriel_00_09_05/selfpaced2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file);
motoJits= [0, -40, -80, -120, -160];
epo= makeEpochs(cnt, mrk, [-1300 0], motoJits);
nMotos= size(epo.y, 2);
no_moto= makeEpochs(cnt, mrk, [-1300 0]-800, [0, -60, -120, -180, -240]);
%no_moto= makeSegments(cnt, mrk, [-1300 0]-800, [0, -80, -160, -240]);
noMotos= size(no_moto.y, 2);
epo= proc_appendEpochs(epo, no_moto);
clear no_moto
epo.className= {'motor', 'no motor'};
epo.y= [repmat([1;0], 1, nMotos) repmat([0;1], 1, noMotos)];

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 200);
fv= proc_jumpingMeans(fv, 5);

nTrials= [10 10];
doXvalidationPlus(fv, 'QDA', nTrials);
%% -> 7.1 (doXvalidation), 13.8 (doXvalidationPlus)

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_selectIval(fv, 240);
fv= proc_jumpingMeans(fv, 8);
doXvalidationPlus(fv, 'LDA', nTrials);
%% -> 15.0 (doXvalidation), 24.5 (doXvalidationPlus)

%% via three classes
epo.y= zeros(3, size(epo.y,2));                 %% labels for l/r motor events
epo.y(1:2,1:nMotos)= repmat(mrk.y, [1 length(motoJits)]);  
epo.y(3,nMotos+1:end)= 1;                       %% labels for no motor events
epo.className= {'left','right','none'};

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 200);
fv= proc_subsampleByMean(fv, 5);

fv.loss= [0 0 1; 0 0 1; 1 1 0];    %% loss just for confusing motor / no_motor
doXvalidationPlus(fv, 'QDA', nTrials);
%% -> 11.5

%model.classy= {'RDA', 0, '*lin'};
%model.param= 0:0.00001:0.0001;
%classy= selectModel(fv, model, [5 10], 1);


fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_selectIval(fv, 240);
fv= proc_subsampleByMean(fv, 8);
fv.loss= [0 0 1; 0 0 1; 1 1 0];
doXvalidationPlus(fv, 'LDA', nTrials);
%% -> 19.7



return



epo.y= zeros(3, size(epo.y,2));                 %% labels for l/r motor events
epo.y(1:2,1:nMotos)= repmat(mrk.y, [1 length(motoJits)+1]);  
epo.y(3,nMotos+1:end)= 1;                       %% labels for no motor events
epo.className= {'left','right','none'};

fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 200);
fv= proc_subsampleByMean(fv, 5);

fv.y= epo.y;
[dummy,label]= max(fv.y);
fv= proc_flaten(fv);
[divTr, divTe]= sampleDivisions(fv.y, [1 10]);
C= train_LSR(fv.x(:,divTr{1}{1}), fv.y(:,divTr{1}{1}));

test_idx= divTe{1}{1};
out= apply_separatingHyperplane(C, fv.x);
[dummy,out]= max(out);
100*mean( out(test_idx) ~= label(test_idx) )


fv.y= [-1 1 0]*fv.y;
C= train_LSR(fv.x(:,divTr{1}{1}), fv.y(:,divTr{1}{1}));

test_idx= divTe{1}{1};
out= apply_separatingHyperplane(C, fv.x);
ml= mean(out(fv.y==-1));
mn= mean(out(fv.y==0));
mr= mean(out(fv.y==1));
t1= (ml+mn)/2;
t2= (mr+mn)/2;
oo= [(out<t1) + 3*(out>t1 & out<t2) + 2*(out>t2)];

100*mean( oo(test_idx) ~= label(test_idx) )


idx= setdiff(test_idx, find(label==3));
ooo= [(out<0) + 2*(out>0)];
100*mean( ooo(idx) ~= label(idx) )
