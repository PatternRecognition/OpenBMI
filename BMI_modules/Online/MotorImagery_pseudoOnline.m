function [ cf_out ] = MotorImagery_pseudoOnline( eeg, eegFb, online )
%MOTORIMAGERY_PSEUDOONLINE Summary of this function goes here
%   Detailed explanation goes here
if ~isfield(online,'train'); error('training module is not exist'); end
if ~isfield(online,'apply'); error('applying module is not exist'); end
if ~isfield(online,'option'); disp('applying module is not exist');
    opt={
        'windowSize', '1000' % 1s
        'paradigm', 'MotorImagery'
        'Feedback','off'
        }; % default setting
    opt=opt_CellToStruct(online.option{:});
else
    opt=opt_CellToStruct(online.option{:});
end
if ~isfield(opt,'ite')  %iteration in while loop
    opt.ite='1';
end

in=eeg; %training
if isfield(online,'train')
    for k=1:length(online.train)
        myFun=str2func(online.train{k});
        switch online.train{k}
            case 'func_csp'
                [out CSP_W, CSP_D]= feval(myFun, in, online.train{k,2},0);
            case 'func_featureExtraction'
                out=feval(myFun, in, online.train{k,2},0);
            case 'classifier_trainClassifier'
                CF_PARAM=feval(myFun, in, online.train{k,2},0);
            otherwise
                out=feval(myFun, in, online.train{k,2},0);
        end
        in=out;
    end
end

time=func_getTimeMarker(eegFb, opt);
cf_out=zeros(1,time.ite);
% eegFb.x=eegFb.cnt;  eegFb=rmfield(eegFb, 'cnt'); %% change dat.cnt to dat.x
run=true;
while run,
%     str2num(opt.ite)
    [fbData]=func_getData(eegFb, time, opt); %opt for iteration++
    in=fbData;
    if isfield(online,'apply')
        for k=1:length(online.apply)
            myFun=str2func(online.apply{k});
            switch online.apply{k}
                case 'func_projection'
                    [out]=feval(myFun, in, CSP_W);
                case 'func_featureExtraction'
                    [out]=feval(myFun, in, online.apply{k,2},0);
                case 'classifier_applyClassifier'
                    [out]=feval(myFun, in, CF_PARAM,0);
                otherwise
                    out=feval(myFun, in, online.apply{k,2},0);
            end
            in=out;
        end
        cf_out(str2num(opt.ite))=in;
        if strcmp(opt.Feedback,'on')
        end
    else
        waring('online.apply is not exist');
    end
    if str2num(opt.ite) == time.ite
        run=false;
    end
    opt.ite=func_countInc(opt.ite);
end
end

