function [out] = func_predict( dat, varargin)
% FUNC_PREDICT - predict the features of data based on trained classifier
% func_predict (function procedure):
%
% Synopsis:
%     [out] = func_predict(DAT, <OPT>)
% 
% Example:
%     func_predict(fv, cf_param);
%     func_predict(fv, {'classifier', 'lda'; 'w', weight_matrix; 'b', bias});
% 
% Arguments:
%     dat - Structure or feature data
%     varargin - struct or property/value list of optional properties:
%         : cf_param - A results of func_train
%         : classifeir - choose classifier what you want
% Retuns:
%     out - result of testing output
% 
% Description:
%     This function predicts the features of data based on trained classifier.
%     In this version, only producing the lda classifier. 
%     Other classifier algorithm will be updated. Also, finding func_train.
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Hong-kyung Kim, 09-2018
% hk_kim@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin{:})
    error('OpenBMI: func_predict must have options');
end

if isstruct(varargin{:})
    opt=varargin{:};
elseif isnumeric(varargin{1})
    opt.cf_param = varargin{1};
else
    opt=varargin{:}{:}; % cross-validation procedure
end

if ~isfield(opt, 'cf_param') || isempty(opt.cf_param)
    if isfield(opt, 'w')
        opt.cf_param.w = opt.w;
    else
        opt.cf_param.w = 1;
    end
    if isfield(opt, 'b')
        opt.cf_param.b = opt.b;
    else
        opt.cf_param.b = 0;
    end
end

if ~isfield(opt, 'classifier')
    opt.classifier = 'lda';
end

if isstruct(dat) && isfield(dat, 'x')
    dat=dat.x;
elseif isnumeric(dat)
    dat = dat;
else
    error('OpenBMI: Unexpected data type');
end

switch opt.classifier
    case 'lda'
        [row, col]=size(dat);
        if col ~= size(opt.cf_param.w,1) && row == size(opt.cf_param.w, 1)
            dat = dat';
        end
        out= real(dat*opt.cf_param.w+b);
end
end

