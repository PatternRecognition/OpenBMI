function [ loss01 loss ] = eval_crossValidation_( dat, varargin )
%PROC_CROSS_VALIDATION Summary of this function goes here
%   Detailed explanation goes here
CV=varargin{:};

%% Prepocessing procedure, input 'dat' sould not be epoched
if isfield(varargin{:},'prep')
    %Consider the number of output parameter
    param=struct;
    for i=1:length(CV.prep)
        [out_param str_function in_dat in_param in_type] = opt_funcParsing(CV.prep{i});
        save=cell(1,length(out_param)-1);
        nFunc=str2func(str_function);
        if i==1
            in=dat; % do at first iteration
        else
            in=param.(in_dat{1});
        end
        for k=1:length(in_type)  % assigned a input parameter 
            if strcmp(in_type{k}, 'unassigned_variable')
               in_param{k}=CV.var.(in_param{k});
            end
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
% param is the structure which have a output parameters from each
% preprocessing steps
%% training phase, dat.y_dec is nacessary

[out_param str_function in_dat in_param in_type] = opt_funcParsing(CV.train{1}); % find a first training input parameter to assign a data
if isfield(varargin{:},'prep') && isfield(param, in_dat{1})  %% prep 절차가 있고, 전처리 아웃풋 결과를 가지고 올때 
    dat=param.(in_dat{1}); % conect the prep and training proc. with its real-function name
else
    dat=dat;
    % use initial "dat" parameter
end

        

[NofClass NofTrial]=size(dat.y_logic);
switch lower(CV.var.(CV.option{1}))
    case 'kfold'
        CVO = cvpartition(dat.y_dec,'k',CV.var.(CV.option{2}));
    case 'leaveout'
        CVO = cvpartition(dat.y_dec,'Leaveout')
end


for i = 1:CVO.NumTestSets
    idx=CVO.training(i);
    idx2=CVO.test(i);
    train_dat=prep_selectTrials(dat,{'Index', idx});
    test_dat=prep_selectTrials(dat,{'Index', idx2});
    
    
    %Consider the number of output parameter
    for k=1:length(CV.train)
        [out_param str_function in_dat in_param type] = opt_funcParsing(CV.train{k});
        save=cell(1,length(out_param)-1);
        if k==1
            in=train_dat; % do at first iteration
        else
            if iscell(in_dat)
                in=param.(in_dat{1}); % 이후 k가 증가할 경우 실제 input output 데이터를 따라서
            else  % string
                in=param.(in_dat);
            end
        end
        for kk=1:length(type)  % assigned a input parameter
            if strcmp(type{kk}, 'unassigned_variable')
                in_param{kk}=CV.var.(in_param{kk});
            end
        end
        for j=1:length(in_param)
            switch type{j}
                case 'string'
                case 'numeric'
                case 'variable' % find a parameter from inner CV procedures
                    if ~isempty(param)
                        if isfield(param, in_param{j})
                            in_param{j} = param.(in_param(j));
                        end
                    end
            end
        end
        
        
        nFunc=str2func(str_function);
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
    %..  들어가는 부분에서 pair로 넣을건지, 단일로 넣을건지 결정,
    in=test_dat;
    for k=1:length(CV.test)
        [out_param str_function in_dat in_param type] = opt_funcParsing(CV.test{k});
        if k==1
            in=test_dat; % do at first iteration
        else
            if iscell(in_dat)
            in=param.(in_dat{1}); % 이후 k가 증가할 경우 실제 input output 데이터를 따라서
            else  % string
                in=param.(in_dat);
            end
        end        
        for kk=1:length(type)  % assigned a input parameter
            if strcmp(type{kk}, 'unassigned_variable')
                in_param{kk}=CV.var.(in_param{kk});
            end
        end
        nFunc=str2func(str_function);
        
        for j=1:length(in_param)
            switch type{j}
                case 'string'
                case 'numeric'
                case 'variable' % find a parameter from inner CV procedures
                    if ~isempty(param)
                        if isfield(param, in_param{j})
                            in_param{j} = param.(in_param{j});
                        end
                    end
            end
        end        
        [out save{:}]= feval(nFunc, in, in_param);
        param.(out_param{1})=out;        
    end
    loss(i,:)=eval_calLoss(test_dat.y_logic, out);
end
loss01=mean(loss);

end
