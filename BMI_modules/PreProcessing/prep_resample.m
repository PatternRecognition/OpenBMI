function [out] = prep_resample(dat, fs, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prep_resample (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_resample(DAT, fs, <OPT>)
%
% Example :
%     [out] = prep_resample(dat, fs, {'Nr', n_samples})
%
% Arguments:
%     dat - Structure. Continuous data or epoched data
%         - Data which channel is to be selected     
%     fs[Hz] - Desired sampling frequency (scalar)
%     varargin - struct or property/value list of optional properties:
%          : Nr(scalar) - Remove the first and last Nr samples from the
%            resampled dat to avoid edge effects (default is 0)
%           
% Returns:
%     out - Data structure which changed sampling freqeuncy 
%
%
% Description:
%     This function changes the sampling rate of the given EEG signal.
%     It can do both up/downsampling, considering frequency up to 3 digits.
%     continuous data should be [time * channels]
%     epoched data should be [time * channels * trials]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 12-2017
% minho_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    disp('The number of removed samples is 0 (default)');
    opt.Nr = 0;
elseif ~iscell(varargin)
    warning('OpenBMI: Number of samples to be removed should be in a correct form, cell type')
    return
else
    opt = opt_cellToStruct(varargin{:});
end

if ~isfield(dat,'x') || ~isfield(dat,'fs')
    warning('OpenBMI: Data must have fields named ''x'',''fs''')
    return
end

p = round(1000*fs);
q = round(1000*dat.fs);

if ndims(dat.x) == 3
    n_t = size(dat.x,1);
    n_tr = size(dat.x,2);
    n_ch = size(dat.x,3);
    x = zeros(ceil(n_t*p/q),n_tr,n_ch);
    for i = 1:n_tr
        xt = reshape(dat.x(:,i,:),[n_t,n_ch]);
        xt = resample(xt,p,q);
        x(:,i,:) = reshape(xt,[size(xt,1),1,n_ch]);
    end
    if isfield(dat,'ival')
        out = rmfield(dat,{'x','fs','ival'});
        out.ival = linspace(dat.ival(1),dat.ival(end),size(x,1));
        out.ival = out.ival((opt.Nr+1):end-opt.Nr);
    end
    x = x((opt.Nr+1):end-opt.Nr,:,:);
elseif ismatrix(dat.x)
    out = rmfield(dat,{'x','fs'});
    x = resample(dat.x,p,q);
    x = x((opt.Nr+1):end-opt.Nr,:);
else
    warning('Check for the dimension of input data')
    return
end

out.fs = fs;
out.x = x;
