function [out,r]=func_cca(dat, varargin)
% FUNC_CCA - calculate correlation parameters base on CCA
% Synopsis:
%     [out] = func_cca(DAT,<OPT>)
%Example:
%     [out] = func_cca(SMT, {'refer', [5 7 9 11]})
%     [out] = func_cca(SMT, {'refer', [5 7 9 11]; 'harmonic', 2});
%     [out] = func_cca(SMT.x, {'refer', [5 7 9 11]; 'fs', 100});
% Input:
%     dat - Data structure, epoched data
%
% Option:
%     refer - Making reference frequency
%     harmonic - Number of harmonic frequncy
% 	  fs - Setting the data sampling rate. If the dat is not a struct
% 	  property, option 'fs' is necessary.
%
% Returns:
%     out - Calculating classification result of class which maximum r value  
%     r - Calculating r values 
%
% Description:
%     This function calculating correlation parameters based on 
%     canonical correlation analysis(CCA).  CCA is a method for calculating 
%     the relationships between two multivariate signal. 
%     Epoched data should be [time * trials * channels]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Hong-kyung Kim, 09-2018
% hk_kim@korea.ac.kr

%% channel must be segemented
% function proc_CCA(dat,time, frequency, harmonic)
%

out = dat;

if isempty(varargin)
    warning('OpenBMI: Options should be specified')
    return
end

if isnumeric(varargin{1})
    opt.refer = varargin{1};
elseif iscell(varargin{1})
    opt = opt_cellToStruct(varargin{:});
end

if ~isfield(opt, 'refer')
    error('OpenBMI: Reference frequency must be needed');
end
if ~isfield(opt, 'harmonic')
    warning(sprintf('OpenBMI: parameter "harmonic" is missing\nThe default harmonic value is 2'))
    opt.harmonic=2;
end
if isstruct(dat)
    if ~all(isfield(dat, {'x', 'fs', 'chan'}))
        error('OpenBMI: Data must have a field named ''x'', ''fs'' ,and ''chan''');
    end
    if ~(ndims(dat.x)==3 || (ismatrix(dat) && dat.chan == 1))
        error('OpenBMI: Data must be segmented');
    end
    comp_dat = dat.x;
    fs = dat.fs;
    if isfield(dat, 'ival')
        t = dat.ival/1000;
    else
        t = linspace(0, floor(size(comp_dat, 1)/fs), size(comp_dat,1));
    end
elseif isnumeric(dat)
    if ~isfield(opt, 'fs')
        error('OpenBMI: Data must have a field named ''fs''');
    end
    comp_dat = dat;
    fs = opt.fs;
    t = linspace(0, floor(size(comp_dat, 1)/fs), size(comp_dat,1));
else
    error('OpenBMI: ');
end

Y = cell(1,length(opt.refer));
corr_r = cell(size(comp_dat, 2), length(opt.refer));

ref = arrayfun(@(x) 2*pi*t*x, opt.refer, 'Uni', false);
for i = 1:opt.harmonic
    Y = cellfun(@(x,y) vertcat(x, [sin(y*i);cos(y*i)]),Y, ref,'Uni',false);
end

for i = 1:size(comp_dat, 2)
    X = squeeze(comp_dat(:,i,:));
    [~,~,corr_r(i,:)] = cellfun(@(y) canoncorr(X,y'), Y, 'Uni', false);
end

r = cellfun(@max, corr_r)';
[~, out] = max(r);
end