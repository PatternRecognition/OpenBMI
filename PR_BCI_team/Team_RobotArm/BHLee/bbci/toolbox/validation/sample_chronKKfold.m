function [divTr, divTe]= sample_chronKKfold(label, folds)
%[divTr, divTe]= sample_chronKKfold(label, folds)
nSamples= size(label,2);

if length(folds)==1
    folds = [1 folds];
end

divTr= {cell(1,folds(2))};
divTe= {cell(1,folds(2))};

for nn= 1:folds(1)
    div = round(linspace(0+(nn-1), nSamples-(folds(1)-nn), folds(2)+1));
    for kk= 1:folds(2)
        test_set= div(kk)+1:div(kk+1);
        train_set= setdiff(nn:nSamples-(folds(1)-nn), test_set);
        %% check that all classes are inhabited
        if ~all(sum(label(:,train_set),2)),
            warning('empty classes in training set');
        end
    divTe{nn}{kk}= test_set;
    divTr{nn}{kk}= train_set;
    end
end
