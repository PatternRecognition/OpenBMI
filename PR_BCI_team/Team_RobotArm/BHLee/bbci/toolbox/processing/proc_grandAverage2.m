function ga= proc_grandAverage2(varargin)
% PROC_GRANDAVERAGE2 - calculates the grand average of the given data.
%
%Usage:
%ga= proc_grandAverage2(erps)
%ga= proc_grandAverage2(erps, <OPT>)
%ga= proc_grandAverage2(erp1, <erp2, ..., erpN>, <Prop1>, <Val1>, ...)
%
% IN   erps - cell array of erp-like structures, or
%      erpn - erp-like structure
%      Supported data types (yUnit) are: ERPs/ERDs/ERSs, power spectra (dB),
%      r-values, signed r^2-values, z-values and auc-values.
%
% <OPT> - struct or property/value list of optional fields/properties:
%   average - If 'weighted', each ERP is weighted for the number of
%             epochs/trials, giving less weight to ERPs made from a small
%             number of epochs/trials. (default: 'unweighted')
%   must_be_equal - Fields in the erp structs that must be equal across
%                   different structs, otherwise an error is issued.
%                   (default: {'fs','y','className'})
%   should_be_equal - Fields in the erp structs that should be equal across
%                   different structs, gives a warning otherwise.
%                   (default: {'yUnit'})
%   Bonferroni - If 1 and for grand averages yielding p-values a Bonferroni
%                multiple comparisons correction is performed. (default: 0)
%   Bonferroni_alpha - The significance level used in the Bonferroni
%                      correction. (default: 0.01)
%   Bonferroni_value - Value to which not significant ga.x values are set.
%                      (default: 0)
%Output:
% ga: Grand average
%
% Authors: haufe, schultze-kraft


if iscell(varargin{1}),
    erps= varargin{1};
    opt= propertylist2struct(varargin{2:end});
else
    iserp= apply_cellwise2(varargin, inline('~ischar(x)','x'));
    nerps= find(iserp, 1, 'last');
    erps= varargin(1:nerps);
    opt= propertylist2struct(varargin{nerps+1:end});
end

opt= set_defaults(opt, ...
                  'average', 'unweighted', ...
                  'Bonferroni', false, ...
                  'Bonferroni_alpha', .01, ...
                  'Bonferroni_value', 0, ...
                  'must_be_equal', {'fs','y','className'}, ...
                  'should_be_equal', {'yUnit'});

valid_erp= apply_cellwise2(erps, 'isstruct');
if any(~valid_erp),
    fprintf('%d non valid ERPs removed.\n', sum(~valid_erp));
    erps= erps(valid_erp);
end

clab= erps{1}.clab;
for ii= 2:length(erps),
    clab= intersect(clab, erps{ii}.clab);
end
if isempty(clab),
    error('intersection of channels is empty');
end

ga= copy_struct(erps{1}, 'not','x', 'V', 'p', 'log10p');

if isfield(ga, 'yUnit') && strcmp(opt.average, 'weighted')
    switch ga.yUnit
        case {'r','sgn r^2','z','auc'}
            error('weighted averaging not supported for yUnits ''r'', ''sgn r^2'', ''z'' or ''auc''.')
    end
end

if isfield(ga, 'yUnit') && opt.Bonferroni
    if ~any(strcmp(ga.yUnit,{'r','sgn r^2','z','auc'}))
        error('Bonferroni correction only for yUnits ''r'', ''sgn r^2'', ''z'' and ''auc''.')
    end
end

must_be_equal= intersect(opt.must_be_equal, fieldnames(ga));
should_be_equal= intersect(opt.should_be_equal, fieldnames(ga));

if isfield(erps{1}, 'f')
    F= size(erps{1}.x, 1);
    T= size(erps{1}.x, 2);
    E= size(erps{1}.x, 4);
    X= zeros([F T length(clab) E length(erps)]);
    iV= zeros([F T length(clab) E length(erps)]);
    Z= zeros([F T length(clab) E length(erps)]);
    dim2avg= 5;
else
    T= size(erps{1}.x, 1);
    E= size(erps{1}.x, 3);
    X= zeros([T length(clab) E length(erps)]);
    iV= zeros([T length(clab) E length(erps)]);
    Z= zeros([T length(clab) E length(erps)]);
    dim2avg= 4;
end
if isfield(erps{1},'N')
    Ns= apply_cellwise(erps,@(s)getfield(s,'N'));
    if ~isfield(erps{1}, 'f')
        N= zeros([T length(clab) E length(erps)]);
        for ii= 1:E
            for jj= 1:length(erps)
                N(:,:,ii,jj)= Ns{jj}(ii);
            end
        end
    else
        N= zeros([F T length(clab) E length(erps)]);
        for ii= 1:E
            for jj= 1:length(erps)
                N(:,:,:,ii,jj)= Ns{jj}(ii);
            end
        end
    end
end

for ii= 1:length(erps),
    for jj= 1:length(must_be_equal),
        fld= must_be_equal{jj};
        if ~isequal(getfield(ga,fld), getfield(erps{ii},fld)),
            error(sprintf('inconsistency in field %s.', fld));
        end
    end
    for jj= 1:length(should_be_equal),
        fld= should_be_equal{jj};
        if ~isequal(getfield(ga,fld), getfield(erps{ii},fld)),
            warning(sprintf('inconsistency in field %s.', fld));
        end
    end
    ci= chanind(erps{ii}, clab);
    if isfield(erps{ii}, 'f')
        X(:,:,:,:,ii)= erps{ii}.x(:,:,ci,:);
        if isfield(erps{ii}, 'V')
            iV(:,:,:,:,ii)= 1./erps{ii}.V(:,:,ci,:);
        end
        if isfield(erps{ii}, 'z')
            Z(:,:,:,:,ii)= erps{ii}.z(:,:,ci,:);
        end
    else
        X(:,:,:,ii)= erps{ii}.x(:,ci,:);
        if isfield(erps{ii}, 'V')
            iV(:,:,:,ii)= 1./erps{ii}.V(:,ci,:);
        end
        if isfield(erps{ii}, 'z')
            Z(:,:,:,ii)= erps{ii}.z(:,ci,:);
        end
    end
end

if isfield(ga, 'yUnit')
    switch ga.yUnit
        case 'r'
            ga.V= 1./sum(iV, dim2avg);
            ga.z= sum(atanh(X).*iV, dim2avg).*sqrt(ga.V);
            ga.x= tanh(ga.z.*sqrt(ga.V));
            ga.p= reshape(2*normal_cdf(-abs(ga.z(:)), zeros(size(ga.z(:))), ones(size(ga.z(:)))), size(ga.z));
            ga.sgn_log10_p= reshape(((log(2)+normcdfln(-abs(ga.z(:))))./log(10)), size(ga.z)).*-sign(ga.z);
        case 'sgn r^2'
            ga.V= 1./sum(iV, dim2avg);
            ga.z= sum(atanh(sqrt(abs(X)).*sign(X)).*iV, dim2avg).*sqrt(ga.V);
            ga.x= tanh(ga.z.*sqrt(ga.V));
            ga.x= ga.x.^2.*sign(ga.x);
            ga.p= reshape(2*normal_cdf(-abs(ga.z(:)), zeros(size(ga.z(:))), ones(size(ga.z(:)))), size(ga.z));
            ga.sgn_log10_p= reshape(((log(2)+normcdfln(-abs(ga.z(:))))./log(10)), size(ga.z)).*-sign(ga.z);
        case 'z'
            ga.x= mean(X, dim2avg).*sqrt(size(X, dim2avg));
            ga.p= reshape(2*normal_cdf(-abs(ga.x(:)), zeros(size(ga.x(:))), ones(size(ga.x(:)))), size(ga.x));
            ga.sgn_log10_p= reshape(((log(2)+normcdfln(-abs(ga.x(:))))./log(10)), size(ga.x)).*-sign(ga.x);
        case 'auc'
            ga.z= mean(Z, dim2avg).*sqrt(size(Z, dim2avg));
            ga.p= reshape(2*normal_cdf(-abs(ga.z(:)), zeros(size(ga.z(:))), ones(size(ga.z(:)))), size(ga.z));
            ga.sgn_log10_p= reshape(((log(2)+normcdfln(-abs(ga.z(:))))./log(10)), size(ga.z)).*-sign(ga.z);
            ga.x=  mean(X, dim2avg);
        case 'dB'
            if strcmp(opt.average, 'weighted')
                ga.x= nansum(N.*10.^(X/10), dim2avg)./sum(N, dim2avg);
            else
                ga.x= nanmean(10.^(X/10), dim2avg);
            end
            ga.x= 10*log10(ga.x);
        otherwise
            if strcmp(opt.average, 'weighted')
                ga.x= nansum(X.*N, dim2avg)./sum(N, dim2avg);
            else
                ga.x= nanmean(X, dim2avg);
            end
    end
    if opt.Bonferroni
        ga.x(ga.p>opt.Bonferroni_alpha/numel(ga.V))= opt.Bonferroni_value;
    end
else
    if strcmp(opt.average, 'weighted')
        ga.x= nansum(X.*N, dim2avg)./sum(N, dim2avg);
    else
        ga.x= nanmean(X, dim2avg);
    end
end

ga.clab= clab;
ga.title= 'grand average';
ga.N= length(erps);
