%% This demo demonstrates how to validate an outlier removal method.
%% It shows by no means an intelligent way to detect outliers, the
%% focus is just to show how to do the validation.

%% bb ida.first.fhg.de 07/2004


S= load([DATA_DIR 'ida_training/bnb_task1']);
fv= struct('x',S.xTr, 'y',S.yTr);


%% This is the wrong way:
ff= proc_outl_distToClassMean(fv, 10, 'policy','perc_of_gauss');
xvalidation(ff, 'LDA');



%% This is the correct validation: remove outliers WITHIN the cross-validation
%% from each training set and evaluate the classifier trained on this
%% cleaned set on the test set. To do this you have to specify a processing
%% for the training set (proc.train).
proc= [];
proc.train= ['fv= proc_outl_distToClassMean(fv, 10, ' ...
                               '''policy'',''perc_of_gauss'', ' ...
                               '''remove_outliers'', 0);'];
xvalidation(fv, 'LDA', 'proc',proc);



%% You can also have free variables that have do be chosen within the
%% cross-validation, e.g., a threshold for the outlier removal:


%% This is the wrong way:
thresh_list= [5 10 15];
for ii= 1:length(thresh_list),
  thresh= thresh_list(ii);
  ff= proc_outl_distToClassMean(fv, thresh, 'policy','perc_of_gauss');
  err(ii)= xvalidation(ff, 'LDA', ...
                       'out_prefix', sprintf('thresh= %d -> ', thresh));
end
[min_err,mi]= min(err);
fprintf('using thresh= %d yields %.1f%% error\n', thresh_list(mi), ...
        100*min_err);



%% This is the correct validation:
proc= [];
proc.train= ['fv= proc_outl_distToClassMean(fv, thresh, ' ...
                               '''policy'',''perc_of_gauss'', ' ...
                               '''remove_outliers'', 0);'];
proc.param.var= 'thresh';
proc.param.value= {5, 10, 15};

[l,s,o,m]= xvalidation(fv, 'LDA', 'proc',proc, 'save_proc_params',{'thresh'});
fprintf('chosen thresholds: %s\n', vec2str([m.thresh]));

