%% This demo shows how to define different feature processings for
%% training and test set.

%% bb ida.first.fhg.de 07/2004


S= load([DATA_DIR 'ida_training/bnb_task1']);
Fv= struct('x',S.xTr, 'y',S.yTr);
fv_test= struct('x',S.xTe);

%% Imagine you want to normalize your data and then train a classifier
%% on your training data. To apply this classifer to future 'test data'
%% you should use the same transform, that you applied to the training
%% data:

fv= Fv;
[fv, opt_sm]= proc_subtractMean(fv);
[fv, opt_nr]= proc_normalize(fv);
C= trainClassifier(fv, 'LDA');

fv_test= proc_subtractMean(fv_test, opt_sm);
fv_test= proc_normalize(fv_test, opt_nr);
out= applyClassifier(fv_test, 'LDA', C);


%% To estimate the generalization error of such a procedure you could do:
fv= Fv;
fv= proc_subtractMean(fv);
fv= proc_normalize(fv);
xvalidation(fv, 'LDA');

%% But here the normalization of the data is calculated on the whole
%% data set fv, i.e., the transform is determined also on data that
%% is used as test data in the cross-validation.

%% In this case this might not really matter, but to make an unbiased
%% validation of the above method, you would have to define different
%% processings for the training and the test set (of the cross-validation).
%% This is done by the fields .train and .apply of the variable proc.
%% Quantities that have to be determine from the training data alone
%% have to be transported to the processing of the test data. This is
%% accomplished by the field .memo of variable proc.

fv= Fv;
proc= [];
proc.train= ['[fv, opt_sm]= proc_subtractMean(fv); ' ...
             '[fv, opt_nr]= proc_normalize(fv); '];
proc.apply= ['fv= proc_subtractMean(fv, opt_sm); ' ...
             'fv= proc_normalize(fv, opt_nr); '];
proc.memo= {'opt_sm', 'opt_nr'};

xvalidation(fv, 'LDA', 'proc', proc);

% In the next example, there is an additional input variable dim to
% determine the form of normalization. This is piece of code may not be of
% particular practical use, I put it here mainly to demonstrate how
% variables can be passed to the proc-routines without coding them directly
% into the proc-string.
fv = Fv;
proc= [];
proc.train= ['[fv, opt_sm]= proc_subtractMean(fv); ' ...
             '[fv, opt_nr]= proc_normalize(fv, ''policy'', policy);'];
proc.param.var = 'policy';
% Mind that the value here must be in a cell array.
proc.param.value = {'max'};
proc.apply= ['fv= proc_subtractMean(fv, opt_sm); ' ...
             'fv= proc_normalize(fv, opt_nr); '];
proc.memo= {'opt_sm', 'opt_nr'};

xvalidation(fv, 'LDA', 'proc',proc);


%% Note: In this example both strategies lead to (approx.) the same
%% generalization error. That means, that in this simple case the
%% 'possibly biased' validation methods make no bias. This may well
%% be different with other preprocessings, see demo_feature_selection.
