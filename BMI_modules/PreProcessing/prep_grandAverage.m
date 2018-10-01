function [out] = prep_grandAverage(cell_dat, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_GRANDAVERAGE - average the data across subjects
% prep_grandAverage (Pre-processing procedure):
%
% Synopsis:
%   [out] = prep_grandAverage(dat,<var>)
%
% Example :
%   [out] = prep_grandAverage({dat1,dat2,...,datN}, {'method', 'weighted');
%   [out] = prep_grandAverage({dat1,dat2,...,datN}, {'method', 'casual');
%
% Arguments:
%     dat - Segmented data in a cell type
%     varargin = property/value list of optional properties:
%         : 'methods' - 'weigthed' or 'casual'
%
% Returns:
%     out - Data structure averaged across subjects
%
% Description:
%    This function averages the data across subjects. It is recommended that
%    input data be separated by each class in advance.
%     continuous data should be [time * channels]
%     epoched data should be [time * trials * channels]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 09-2018
% mh_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~iscell(cell_dat)
    error('OpenBMI: cell_dat should be a cell type');
end

if ~all(cellfun(@(x) isfield(x, 'x'), cell_dat))
    error('OpenBMI: Data must have a field named ''x''');
else

if isempty(varargin)
    opt.method = 'casual';
else
    opt = opt_cellToStruct(varargin{:});
end

non_average = cellfun(@(x) size(x.x, 2) ~= size(x.class, 1), cell_dat);
cell_dat(non_average) = cellfun(@prep_average, cell_dat(non_average), 'Uni', false);

fields_check = {'class', 'chan', 'ival'};
check_dat = cellfun(@(x) getfield([cell_dat{:}], x),{1:3},fields_check, 'Uni', false);

% for i = 1:length(cell_dat)
%     check_dat(i).(fields_check{:}) = 
% end

for i = 1:length(cell_dat)
    dat = cell_dat{i};
    sgnr_cls = ismember(dat.class(:,2), 'sgn r^2');
    if any(sgnr_cls)
        dat.x(:,sgnr_cls,:) = atanh(sqrt(abs(dat.x(:,sgnr_cls,:))).*sign(dat.x(:,sgnr_cls,:)));        
    end
    cell_dat{i} = dat;
end

[time, trials, channels] = size(cell_dat{1}.x);
dat_av = zeros(time, trials, channels);
se = dat_av;

for cls = 1:size(cell_dat{1}.class, 1)
    sW = 0;
    swV = 0;
    for v = 1:length(cell_dat)
        switch opt.method
            case 'weighted'
                W = 1./cell_dat{v}.se(:, cls,:).^2;
            case 'casual'
                W = 1;
        end
        sW = sW + W;
        swV = swV + W.^2.*cell_dat{v}.se(:, cls, :).^2;
        dat_av(:, cls, :) = dat_av(:, cls,:) + W.*cell_dat{v}.x(:,cls,:);
    end
    dat_av(:, cls,:) = dat_av(:, cls,:)./sW;
    se(:, cls,:) = sqrt(swV)./sW;
end

if isequal(dat.class{1,2}, 'sgn r^2')
    dat_av = tanh(dat_av).*abs(tanh(dat_av));
end

out = cell_dat{1};
out.x = dat_av;
out.se = se;

out = opt_history(out, mfilename, opt);
end