function [ loss01 loss ] = eval_crossValidation( dat, varargin )
%PROC_CROSS_VALIDATION Summary of this function goes here
%   Detailed explanation goes here
CV=varargin{:};

%% Prepocessing procedure, input 'dat' sould not be epoched
if isfield(dat, 'x')  %if not epoching
    in=dat;
    %Consider the number of output parameter
    for i=1:length(CV.prep)
        myFun=str2func(CV.prep{i});
        switch CV.prep{i}
            case 'prep_filter'
                [out]= feval(myFun, in, CV.prep{i,2},0);
            case 'prep_segmentation'
                [out]=feval(myFun, in, CV.prep{i,2},0);
            otherwise
                [out]=feval(myFun, in, CV.prep{i,2},0);
        end
        in=out;
    end
    dat=in;
end
%%

[NofClass NofTrial]=size(dat.y_logical);
switch lower(CV.option{1})
    case 'kfold'
        CVO = cvpartition(dat.y,'k',str2num(CV.option{2}));
end


for i = 1:CVO.NumTestSets
    idx=CVO.training(i);
    idx2=CVO.test(i);
    train_dat=prep_selectTrials(dat,idx);
    test_dat=prep_selectTrials(dat,idx2);
    in=train_dat;
    %Consider the number of output parameter
    for k=1:length(CV.train)
        myFun=str2func(CV.train{k});
        switch CV.train{k}
            case 'func_csp'
                param.CSP_W=[];
                [out CSP_W, CSP_D]= feval(myFun, in, CV.train{k,2},0);
            case 'func_featureExtraction'
                out=feval(myFun, in, CV.train{k,2},0);
            case 'classifier_trainClassifier'
                param.CF_PARAM=[];
                CF_PARAM=feval(myFun, in, CV.train{k,2},0);
            otherwise
                out=feval(myFun, in, CV.train{k,2},0);
        end
        in=out;
    end
    
    in=test_dat;
    for k=1:length(CV.test)
        myFun=str2func(CV.test{k});
        switch CV.test{k}
            case 'func_projection'
                if isfield(param,'CSP_W');
                    [out]=feval(myFun, in, CSP_W);
                else
                    disp('check CSP parameter');
                end
            case 'func_featureExtraction'
                [out]=feval(myFun, in, CV.test{k,2},0);
            case 'classifier_applyClassifier'
                if isfield(param, 'CF_PARAM')
                    [out]=feval(myFun, in, CF_PARAM,0);
                else
                    disp('check classifier parameter');
                end
            otherwise
                out=feval(myFun, in, CV.test{k,2},0);
        end
        in=out;
    end
    loss(i,:)=eval_calLoss(test_dat.y_logical, in);
end

loss01=mean(loss);

end

