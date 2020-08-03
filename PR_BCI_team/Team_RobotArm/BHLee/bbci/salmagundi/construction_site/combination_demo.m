% DEMONSTRATE feature Combination, see NIPS2002
% DUDU, 03.07.02

% load cnt, filtering and segmentation
file = 'Gabriel_01_10_15/imagGabriel';

[cnt,mrk,mnt] = loadProcessedEEG(file);

xTrials = [10 10];
msTrials = [3 10 9/10*length(mrk.pos)];

epo = makeSegments(cnt,mrk,[-1000 2000]);

epo = proc_baseline(epo,[-1000 -300]);


%preprocessing BP
fvbp = proc_selectChannels(epo,'C5-6','CP5-6','FC5-6');

fvbp = proc_selectIval(fvbp,[0 1000]);
fvbp = proc_jumpingMeans(fvbp,33);

% model for BP
modelbp.classy = 'RLSR';
modelbp.param.index = 2;
modelbp.param.scale = 'log';
modelbp.param.value = -5:5;
modelbp.msDepth = 2;

% model selektion and validation

modelbp = selectModel(fvbp,modelbp,msTrials,0);
te = doXvalidationPlus(fvbp,modelbp,xTrials,0);
fprintf('X-validation error for BP: %2.1f\n',te(1));


% preprocessing AR
band = [3 45];
cnt_flt = proc_filtForth(cnt,band);
epo = makeSegments(cnt_flt,mrk,[-1000 2000]);
epo = proc_laplace(epo,'small');
epo = proc_baseline(epo,[-1000 -300]);
fvar = proc_selectChannels(epo, 'C5-6','CP5-6' );

fvar = proc_selectIval(fvar,[500 1500]);

fvar = proc_arCoefsPlusVar(fvar,8);


% model for AR
modelar.classy = 'RLSR';
modelar.param.index = 2;
modelar.param.scale = 'log';
modelar.param.value = -5:5;
modelar.msDepth = 2;

% model selektion and validation

modelar = selectModel(fvar,modelar,msTrials,0);
te = doXvalidationPlus(fvar,modelar,xTrials,0);
fprintf('X-validation error for AR: %2.1f\n',te(1));


% preprocessing CSP

band = [7 30];
cnt_flt = proc_filtForth(cnt,band);

epo = makeSegments(cnt_flt,mrk,[-1000 2000]);
epo = proc_baseline(epo,[-1000 -300]);

fvcsp = proc_selectIval(epo,[500 1500]);

fvcsp.proc = ['fv = proc_csp(epo, 4); fv = proc_variance(fv); fv = proc_logNormalize(fv);' ];

% model for CSP
modelcsp.classy = 'RLDA';
modelcsp.param.index = 2;
modelcsp.param.scale = 'lin';
modelcsp.param.value = 0:0.2:1;
modelcsp.msDepth = 2;

% model selektion and validation

modelcsp = selectModel(fvcsp,modelcsp,msTrials,0);
te = doXvalidationPlus(fvcsp,modelcsp,xTrials,0);
fprintf('X-validation error for CSP: %2.1f\n',te(1));

% get some free space
clear cnt mrk epo mnt

% now combination
% the most important function here is proc_appendFeatures. The call
% is very simple:

fvbpar = proc_appendFeatures(fvbp,fvar);
fvall = proc_appendFeatures(fvbp,fvar,fvcsp);

% the following is possible too, if you want to concat two features
% and then use another combination
fvMD = proc_appendFeatures(fvbp,proc_appendFeatures(fvar,fvcsp));

% first of all the demonstration of the CONCAT, important is here
% the classifier catFeatures, as second argument the name of the
% CONCAT classifier is needed

model.classy = {'catCombiner', 'RLSR'};
model.param.index = 3;
model.param.scale = 'log';
model.param.value = -5:5;
model.msDepth = 2;

model = selectModel(fvbpar,model,msTrials,0);
te = doXvalidationPlus(fvbpar,model,xTrials,0);
fprintf('X-validation error for CONCAT BP-AR: %2.1f\n',te(1));

% the same is possible for fvall, fvMD (only for the last done
% below)
model.classy = {'catCombiner', 'RLSR'};
model.param.index = 3;
model.param.scale = 'log';
model.param.value = -5:5;
model.msDepth = 2;

model = selectModel(fvMD,model,msTrials,0);
te = doXvalidationPlus(fvMD,model,xTrials,0);
fprintf('X-validation error for CONCAT BP-AR+CSP: %2.1f\n',te(1));



% now the second combination method, PROB

% THIS is simple, too... , different approaches possible
% only LDA (QDA is equal)
te = doXvalidationPlus(fvbpar,'probCombiner',xTrials,0);
% or te = doXvalidationPlus(fvbpar,{'probCombiner','LDA'},xTrials,0);
fprintf('X-validation error for PROB with LDA for BP and AR: %2.1f\n',te(1));

% We can do it with RLDA (if only one paramater used, it is used
% for all features equally)
model.classy = {'probCombiner','RLDA'};
model.param(1).index = 3;
model.param(1).scale = 'lin';
model.param(1).value = 0:0.2:1;
model.param(2).index = 4;   
model.param(2).scale = 'lin';
model.param(2).value = 0:0.2:1;
model.msDepth = 2;
model = selectModel(fvbpar,model,msTrials,0);
te = doXvalidationPlus(fvbpar,model,xTrials,0);
fprintf('X-validation error for PROB with RLDA for BP and AR: %2.1f\n',te(1));


% The same is possible for RDA, then paramter are given feature for
% feature as lambda,gamma,...
% Combination of different types is possible, too, e. g. If you want to
% do BP as RLDA and AR as RLDA, do the following
model.classy = {'probCombiner',{'RLDA','RDA'}};
model.param(1).index = 3;  % for BP only one parameter
model.param(1).scale = 'lin';
model.param(1).value = 0:0.2:1;
model.param(2).index = 4;   % lambda for AR
model.param(2).scale = 'lin';
model.param(2).value = 0:0.2:1;
model.param(3).index = 5;   % gamma for AR
model.param(3).scale = 'lin';
model.param(3).value = 0:0.2:1;
model.msDepth = 2;
model = selectModel(fvbpar,model,msTrials,0);
te = doXvalidationPlus(fvbpar,model,xTrials,0);
fprintf('X-validation error for PROB with RLDA for BP and RDA for AR: %2.1f\n',te(1));

% If you want to concat one part and then make PROB you have to go
% the same way for fvMD. probCombiner only takes the first level as
% independent, other levels will be concatenated
model.classy = {'probCombiner','RLDA'};
model.param(1).index = 3;
model.param(1).scale = 'lin';
model.param(1).value = 0:0.2:1;
model.param(2).index = 4;   
model.param(2).scale = 'lin';
model.param(2).value = 0:0.2:1;
model.msDepth = 2;
model = selectModel(fvMD,model,msTrials,0);
te = doXvalidationPlus(fvMD,model,xTrials,0);
fprintf('X-validation error for PROB with RLDA for BP and CONCAT AR and CSP: %2.1f\n',te(1));


% THe third method for combination is META, the simplest case is the
% following: (with FisherDiscriminant as metaclassifier
model = {'metaCombiner','FisherDiscriminant',{modelbp, ...
		    modelar}};

te = doXvalidationPlus(fvbpar,model,xTrials,0);
fprintf('X-validation error for META with FD for BP and AR: %2.1f\n',te(1));

% modelbp and so on ban be structs. Then the selectModel will be
% done in the training... This is very heavy for
% computation.... (selectModel in a cross validation)

% You can do META regularised:
model.classy = {'metaCombiner','RLDA',{modelbp, modelar,modelcsp}};
model.param.index = 4;
model.param.scale = 'lin';
model.param.value = 0:0.2:1;
model.msDepth = 2;
model = selectModel(fvall,model,msTrials,0);
te = doXvalidationPlus(fvall,model,xTrials,0);
fprintf('X-validation error for META with RLSR for BP, AR and CSP: %2.1f\n',te(1));

% Interesting is again the case, where you want to concat AR and
% CSP first, and then make BP. Do it so: (regularisation is
% possible, too)


model = {'metaCombiner','FisherDiscriminant',{modelbp,{'catCombiner','FisherDiscriminant'}}};
te = doXvalidationPlus(fvMD,model,xTrials,0);
fprintf('X-validation error for META with FisherDiscriminant for BP and CONCAT AR and CSP: %2.1f\n',te(1));








