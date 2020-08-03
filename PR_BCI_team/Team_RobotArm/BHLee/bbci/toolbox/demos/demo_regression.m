%% IN CONSTRUCTION

fv= load_features_from_ascii([DATA_DIR 'uci/housing/housing.data'], ...
                             'regression',1, 'label_idx', 14);

model= struct('classy', 'regr_nuLPM');
model.param(1)= struct('index',2, 'scale','log', ...
                       'value',[-3:0]);
model.param(2)= struct('index',3, 'scale','log', ...
                       'value',[-1:3]);

% Regression is evaluated with mean-square loss. The standard sampling
% functions in xvalidation.m try to balance classes - which we obviously
% don't have here. Use kfold sampling instead, both in the xvalidation
% and in the nested model-selection step (ms_sample_fcn)
opt = struct('loss','meanSquared');
opt.sample_fcn = {'kfold', [10 5]};
opt.ms_sample_fcn = {'kfold', [3 5]};

%% you shouldn't do this: outer model selection
classy= select_model(fv, model, opt);
xvalidation(fv, classy, opt);

%% evaluate on training data
CFY= trainClassifier(fv, classy);
out= applyClassifier(fv, classy, CFY);
plot([fv.y' out']);
legend('truth','regression');


%% the real thing
xvalidation(fv, model, opt);
