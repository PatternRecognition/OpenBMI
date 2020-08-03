function ga= proc_grandAverage(varargin)
% PROC_GRANDAVERAGE -  calculates the grand average ERPs or ERD/ERS from given set of
% data.
%
%Usage:
%ga= proc_grandAverage(erps)
%ga= proc_grandAverage(erps, <OPT>)
%ga= proc_grandAverage(erp1, <erp2, ..., erpN>, <Prop1>, <Val1>, ...)
%
% IN   erps  -  cell array of erp structures
%      erpn  -  erp structure
%
% <OPT> - struct or property/value list of optional fields/properties:
%   average   - If 'weighted', each ERP is weighted for the number of
%               epochs (giving less weight to ERPs made from a small number
%               of epochs). Default 'unweighted'
%   must_be_equal - fields in the erp structs that must be equal across
%                   different structs, otherwise an error is issued
%                   (default {'fs','y','className'})
%   should_be_equal - fields in the erp structs that should be equal across
%                   different structs, gives a warning otherwise
%                   (default {'yUnit'})
%Output:
% ga: Grand average
%

%% Process input arguments
if iscell(varargin{1}),
  erps= varargin{1};
  opt= propertylist2struct(varargin{2:end});
else
  iserp= apply_cellwise2(varargin, inline('~ischar(x)','x'));
  nerps= find(iserp,1,'last');
  erps= varargin(1:nerps);
  opt= propertylist2struct(varargin{nerps+1:end});
end
%% Options
opt= set_defaults(opt, ...
                  'average', 'unweighted', ...
                  'must_be_equal', {'fs','y','className'}, ...
                  'should_be_equal', {'yUnit'});

valid_erp= apply_cellwise2(erps, 'isstruct');
if any(~valid_erp),
  fprintf('%d non valid ERPs removed.\n', sum(~valid_erp));
  erps= erps(valid_erp);
end

%% Get common electrode set
clab= erps{1}.clab;
for ii= 2:length(erps),
  clab= intersect(clab, erps{ii}.clab);
end
if isempty(clab),
  error('intersection of channels is empty');
end

%% Define ga data field
datadim = unique(apply_cellwise2(erps, 'getDataDimension'));
if numel(datadim) > 1
  error('Datasets have different dimensionalities');
end

ga= copy_struct(erps{1}, 'not','x');
must_be_equal= intersect(opt.must_be_equal, fieldnames(ga));
should_be_equal= intersect(opt.should_be_equal, fieldnames(ga));

if datadim==2
  T= size(erps{1}.x, 1);
  E= size(erps{1}.x, 3);
  ci= chanind(erps{1}, clab);
  X= zeros([T length(ci) E length(erps)]);
else
  ci= chanind(erps{1}, clab);
  % Timefrequency data
  if ndims(erps{1}.x)==3   % no classes, erps are averages over single class
    F= size(erps{1}.x, 1);
    T= size(erps{1}.x, 2);
    X= zeros([F T length(ci) length(erps)]);
  elseif ndims(erps{1}.x)==4
    F= size(erps{1}.x, 1);
    T= size(erps{1}.x, 2);
    E= size(erps{1}.x, 4);
    X= zeros([F T length(ci) E length(erps)]);
  end
end

%% Store all erp data in X
for ii= 1:length(erps),
  for jj= 1:length(must_be_equal),
    fld= must_be_equal{jj};
    if ~isequal(getfield(ga,fld), getfield(erps{ii},fld)),
      error('inconsistency in field %s.', fld);
    end
  end
  for jj= 1:length(should_be_equal),
    fld= should_be_equal{jj};
    if ~isequal(getfield(ga,fld), getfield(erps{ii},fld)),
      warning('inconsistency in field %s.', fld);
    end
  end
  ci= chanind(erps{ii}, clab);
  if isfield(ga, 'V')
    iV(:,:,:,ii)= 1./erps{ii}.V;  
  end

  if datadim==2
    X(:,:,:,ii)= erps{ii}.x(:,ci,:);
  else
    if ndims(erps{1}.x)==3
      X(:,:,:,ii)= erps{ii}.x(:,:,ci);
    elseif ndims(erps{1}.x)==4
      X(:,:,:,:,ii)= erps{ii}.x(:,:,ci,:);
    end
  end
end

%% Perform averaging
if isfield(ga, 'yUnit') && strcmp(ga.yUnit, 'dB'),
  X= 10.^(X/10);
end
if isfield(ga, 'yUnit') && strcmp(ga.yUnit, 'r'),
  if strcmp(opt.average, 'weighted'),
    % does it make sense to use weighting here?
    warning('weighted averaging not implemented for this case - ask Stefan');
  end
  ga.V = 1./sum(iV, 4);
  z = sum(atanh(X).*iV, 4).*ga.V;
  ga.x= tanh(z);
  ga.p = reshape(2*normal_cdf(-abs(z(:)), zeros(size(z(:))), sqrt(ga.V(:))), size(z));
else
  if strcmp(opt.average, 'weighted'),
    if datadim==2
      ga.x= zeros([T length(ci) E]);
      for cc= 1:size(X, 3),  %% for each class
        nTotalTrials= 0;
        for vp= 1:size(X, 4),  %% average across subjects
          % TODO: sort out NaNs
          ga.x(:,:,cc)= ga.x(:,:,cc) + erps{vp}.N(cc)*X(:,:,cc,vp);
          nTotalTrials= nTotalTrials + erps{vp}.N(cc);
        end
        ga.x(:,:,cc)= ga.x(:,:,cc)/nTotalTrials;
      end
    else
      % Timefrequency data
      if ndims(erps{1}.x)==3   % only one class
        ga.x= zeros([F T length(ci)]);
        nTotalTrials= 0;
        for vp= 1:size(X, 4),  %% average across subjects
          % TODO: sort out NaNs
          ga.x = ga.x + erps{vp}.N*X(:,:,:,vp);
          nTotalTrials= nTotalTrials + erps{vp}.N;
        end
        ga.x = ga.x/nTotalTrials;

      elseif ndims(erps{1}.x)==4
        ga.x= zeros([F T length(ci) E]);
        for cc= 1:size(X, 4),  %% for each class
          nTotalTrials= 0;
          for vp= 1:size(X, 5),  %% average across subjects
            % TODO: sort out NaNs
            ga.x(:,:,:,cc)= ga.x(:,:,:,cc) + erps{vp}.N(cc)*X(:,:,:,cc,vp);
            nTotalTrials= nTotalTrials + erps{vp}.N(cc);
          end
          ga.x(:,:,:,cc)= ga.x(:,:,:,cc)/nTotalTrials;
        end
      end

    end
  else
    % Unweighted
    if datadim==2 || ndims(erps{1}.x)==3
      ga.x= nanmean(X, 4);
    else
      ga.x= nanmean(X, 5);
    end
  end
end
if isfield(ga, 'yUnit') && strcmp(ga.yUnit, 'dB'),
  ga.x= 10*log10(ga.x);
end

ga.clab= clab;
%% TODO should allow for weighting accoring to field N
%% (but this has to happen classwise)

ga.title= 'grand average';
ga.N= length(erps);
