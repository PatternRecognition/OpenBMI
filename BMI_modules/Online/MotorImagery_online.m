function [ out ] = MotorImagery_online( eeg, online )
%MOTORIMAGERY_ONLINE Summary of this function goes here
%   Detailed explanation goes here
if ~isfield(online,'train'); warning('training module is not exist'); end
if ~isfield(online,'apply'); warning('applying module is not exist'); end

if ~isfield(online,'option'); disp('applying module is not exist');
    opt={
        'device', 'BrainVision'
        'paradigm', 'MotorImagery'
        'Feedback','off'
        'host', 'JohnKim-PC'
        'port','51244'
        }; % default setting
    opt=opt_CellToStruct(online.option{:});
else
    opt=opt_CellToStruct(online.option{:});
end

in=eeg;
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

switch lower(opt.device)
    case 'brainvision'
        H = bv_open(opt.host);
        
end

while (true)
    dat=bv_read(H);
    in=dat';
    if ~isempty(dat)
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
            if strcmp(opt.Feedback,'on')
            end
        end
    end
end

end

