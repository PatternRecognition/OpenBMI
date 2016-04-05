function [ loss01 loss ] = eval_crossValidation( dat, varargin )
%PROC_CROSS_VALIDATION Summary of this function goes here
%   Detailed explanation goes here
CV=varargin{:};

%% Prepocessing procedure, input 'dat' sould not be epoched
if isfield(varargin{:},'prep')
    %Consider the number of output parameter
    param=struct;
    for i=1:length(CV.prep)
        [out_param str_function in_dat in_param] = opt_funcParsing(CV.prep{i});        
        save=cell(1,length(out_param)-1);
        nFunc=str2func(str_function);
        if i==1
            in=dat; % do at first iteration
        else
            in=param.(in_dat{1});
        end
        [out save{:}]= feval(nFunc, in, in_param);
        if length(out_param)==1  % save an actual output parameter with its real variable name
            param.(out_param{1})=out;
        else
            param.(out_param{1})=out;
            for j=2:length(out_param)
                param.(out_param{j})=save{j-1};
            end
        end
    end
end
% param is the structure which have a outputs(nomally output data) from each preprocessing functions
%% training phase, dat.y_dec is nacessary

[out_param str_function in_dat in_param] = opt_funcParsing(CV.train{1}); % find a first training input parameter to assign a data
if isfield(varargin{:},'prep')
    dat=param.(in_dat{1}); % conect the prep and training proc. with its real-function name
else
    % use initial "dat" parameter
end


[NofClass NofTrial]=size(dat.y_logic);
switch lower(CV.option{1})
    case 'kfold'
        CVO = cvpartition(dat.y_dec,'k',str2num(CV.option{2})); 
end


for i = 1:CVO.NumTestSets
    idx=CVO.training(i);
    idx2=CVO.test(i);
    train_dat=prep_selectTrials(dat,{'index', idx});
    test_dat=prep_selectTrials(dat,{'index', idx});


    %Consider the number of output parameter
    for k=1:length(CV.train)
        [out_param str_function in_dat in_param] = opt_funcParsing(CV.train{k});        
        if k==1
            in=train_dat; % do at first iteration
        else
            in=param.(in_dat{1}); % 이후 k가 증가할 경우 실제 input output 데이터를 따라서 
        end
        save=cell(1,length(out_param)-1);
        nFunc=str2func(str_function)
        [out save{:}]= feval(nFunc, in, in_param);
        if length(out_param)==1  % save an actual output parameter with its real variable name
            param.(out_param{1})=out;
        else
            param.(out_param{1})=out;
            for j=2:length(out_param)
                param.(out_param{j})=save{j-1};
            end
        end     

    end
    %..
    in=test_dat;
    for k=1:length(CV.test)
        [out_param str_function in_dat in_param] = opt_funcParsing(CV.test{k});
        
        
        
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

