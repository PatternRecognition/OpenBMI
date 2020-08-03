function [divTr, divTe]= sample_chronKfold(label, folds)
%[divTr, divTe]= sample_chronKfold(label, folds)

nSamples= size(label,2);

divTr= {cell(folds, 1)};
divTe= {cell(folds, 1)};
div= round(linspace(0, nSamples, folds+1));
for kk= 1:folds,
  test_set= div(kk)+1:div(kk+1);
  train_set= setdiff(1:nSamples, test_set);
  %% check that all classes are inhabited
  if ~all(sum(label(:,train_set),2)),
    warning('empty classes in training set');
  end
  divTe{1}{kk}= test_set;
  divTr{1}{kk}= train_set;
end
