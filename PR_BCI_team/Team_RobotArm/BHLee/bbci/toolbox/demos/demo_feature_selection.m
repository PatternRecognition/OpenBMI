%% This demo demonstrates how to validate a feature selection method
%% It shows by no means a sophisticated way to select features, the
%% focus is just to show how to do the validation.

%% bb ida.first.fhg.de 07/2004


S= load([DATA_DIR 'ida_training/bnb_task1']);
fv= struct('x',S.xTr, 'y',S.yTr);


%% This is the 'fake' variant:
ff= proc_fs_FisherCriterion(fv, 95, 'policy','perc_of_score');
xvalidation(ff, 'LDA');



%% This is the correct validation: select features WITHIN the cross-validation
%% on each training set (-> fidx are the indices of the selected features)
%% and evaluate this selection on the test set. To do this you have to
%% have separate processings for the training set (proc.train) and for the
%% test set (proc.apply). The field proc.memo is used to transport
%% information that has be determined on the training set by the
%% procudure proc.train (here the variable fidx) to the processing on the
%% test set. See also demo_proc_trainApply
proc= [];
proc.train= ['[fv, fidx]= proc_fs_FisherCriterion(fv, 95, ' ...
                               '''policy'',''perc_of_score'');'];
proc.apply= 'fv.x= fv.x(fidx,:);';
proc.memo= {'fidx'};

xvalidation(fv, 'LDA', 'proc',proc);



%% You can also have free variables that have do be chosen within the
%% cross-validation, e.g., a threshold for the feature selection:


%% This is the 'fake' variant:
thresh_list= [85 90 95];
for ii= 1:length(thresh_list),
  thresh= thresh_list(ii);
  ff= proc_fs_FisherCriterion(fv, thresh, 'policy','perc_of_score');
  err(ii)= xvalidation(ff, 'LDA', ...
                       'out_prefix', sprintf('thresh= %d -> ', thresh));
end
[min_err,mi]= min(err);
fprintf('using thresh= %d yields %.1f%% error\n', thresh_list(mi), ...
        100*min_err);



%% This is the correct validation:
proc= [];
proc.train= ['[fv, fidx]= proc_fs_FisherCriterion(fv, thresh, ' ...
                               '''policy'',''perc_of_score'');'];
proc.apply= 'fv.x= fv.x(fidx,:);';
proc.param.var= 'thresh';
proc.param.value= {85, 90, 95};
proc.memo= {'fidx'};

[l,s,o,m]= xvalidation(fv, 'LDA', 'proc',proc, 'save_proc_params',{'thresh'});
fprintf('chosen thresholds: %s\n', vec2str([m.thresh]));

