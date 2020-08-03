function erp_ga= proc_grandAverage3(varargin)
%erp_ga= proc_grandAverage(erps)
%erp_ga= proc_grandAverage(erps, <OPT>)
%erp_ga= proc_grandAverage(erp1, <erp2, ..., erpN>, <Prop1>, <Val1>, ...)
%
% IN   erps  -  cell array of erp structures
%      erpn  -  erp structure
%
% an erp structure is the output of proc_average


if iscell(varargin{1}),
  erps= varargin{1};
  opt= propertylist2struct(varargin{2:end});
else
  iserp= apply_cellwise2(varargin, inline('~ischar(x)','x'));
  nerps= max(find(iserp));
  erps= varargin(1:nerps);
  opt= propertylist2struct(varargin{nerps+1:end});
end
opt= set_defaults(opt, ...
                  'average', 'unweighted');

must_be_equal= {'fs','y','className'};
should_be_equal= {'yUnit'};

valid_erp= apply_cellwise2(erps, 'isstruct');
if any(~valid_erp),
  fprintf('%d non valid ERPs removed.\n', sum(~valid_erp));
  erps= erps(find(valid_erp));
end

clab= erps{1}.clab;
for ii= 2:length(erps),
  clab= intersect(clab, erps{ii}.clab);
end
if length(clab)==0,
  error('intersection of channels is empty');
end

erp_ga= copy_struct(erps{1}, 'not','x', 'z', 'V', 'p', 'log10p');
% if isfield(erps{1}, 'V')
%   if isfield(erps{1}, 'f')
%     iV(:,:,:,:,1)= 1./erps{1}.V;  
%   else
%     iV(:,:,:,1)= 1./erps{1}.V; 
%   end
% end
% if isfield(erps{1}, 'z')
%     if isfield(erps{1}, 'f')
%       Z(:,:,:,:,1)= erps{1}.z;  
%     else
%       Z(:,:,:,1)= erps{1}.z;  
%     end
% end
must_be_equal= intersect(must_be_equal, fieldnames(erp_ga));
should_be_equal= intersect(should_be_equal, fieldnames(erp_ga));
if isfield(erps{1}, 'f')
  F= size(erps{1}.x, 1);
  T= size(erps{1}.x, 2);
  E= size(erps{1}.x, 4);
  X= zeros([F T length(clab) E length(erps)]);
else
  T= size(erps{1}.x, 1);
  E= size(erps{1}.x, 3);
  X= zeros([T length(clab) E length(erps)]);  
end
for ii= 1:length(erps),
  for jj= 1:length(must_be_equal),
    fld= must_be_equal{jj};
    if ~isequal(getfield(erp_ga,fld), getfield(erps{ii},fld)),
      error(sprintf('inconsistency in field %s.', fld));
    end
  end
  for jj= 1:length(should_be_equal),
    fld= should_be_equal{jj};
    if ~isequal(getfield(erp_ga,fld), getfield(erps{ii},fld)),
      warning(sprintf('inconsistency in field %s.', fld));
    end
  end
  ci= chanind(erps{ii}, clab);
  if isfield(erps{ii}, 'f')
    X(:,:,:,:,ii)= erps{ii}.x(:,:,ci,:);
  else
    X(:,:,:,ii)= erps{ii}.x(:,ci,:);
  end
  if isfield(erps{ii}, 'V')
    if isfield(erps{ii}, 'f')
      iV(:,:,:,:,ii)= 1./erps{ii}.V(:,:,ci,:);  
    else
      iV(:,:,:,ii)= 1./erps{ii}.V(:,ci,:);  
    end
  end
  if isfield(erps{ii}, 'z')
    if isfield(erps{ii}, 'f')
      Z(:,:,:,:,ii)= erps{ii}.z(:,:,ci,:);  
    else
      Z(:,:,:,ii)= erps{ii}.z(:,ci,:);  
    end
  end
end
if isfield(erps{1}, 'f')
    if isfield(erp_ga, 'yUnit') 
      switch erp_ga.yUnit
          case 'dB'
            erp_ga.x= nanmean(10.^(X/10), 5);
            erp_ga.x= 10*log10(erp_ga.x);
          case 'r'
            erp_ga.V = 1./sum(iV, 5);
            erp_ga.z = sum(atanh(X).*iV, 5).*sqrt(erp_ga.V);
            erp_ga.x= tanh(erp_ga.z.*sqrt(erp_ga.V));
            erp_ga.p = reshape(2*normal_cdf(-abs(erp_ga.z(:)), zeros(size(erp_ga.z(:))), ones(size(erp_ga.z(:)))), size(erp_ga.z));
            erp_ga.sgn_log10_p = reshape(((log(2)+normcdfln(-abs(erp_ga.z(:))))./log(10)), size(erp_ga.z)).*-sign(erp_ga.z);
          case 'z'
            erp_ga.x= mean(X, 5).*sqrt(size(X, 5));
            erp_ga.p = reshape(2*normal_cdf(-abs(erp_ga.x(:)), zeros(size(erp_ga.x(:))), ones(size(erp_ga.x(:)))), size(erp_ga.x));
            erp_ga.sgn_log10_p = reshape(((log(2)+normcdfln(-abs(erp_ga.x(:))))./log(10)), size(erp_ga.x)).*-sign(erp_ga.x);
          case 'auc'
            erp_ga.z = mean(Z, 5).*sqrt(size(Z, 5));
            erp_ga.p = reshape(2*normal_cdf(-abs(erp_ga.z(:)), zeros(size(erp_ga.z(:))), ones(size(erp_ga.z(:)))), size(erp_ga.z));
            erp_ga.sgn_log10_p = reshape(((log(2)+normcdfln(-abs(erp_ga.z(:))))./log(10)), size(erp_ga.z)).*-sign(erp_ga.z);
%             erp_ga.V = 1./sum(iV, 5);
%             erp_ga.x =  sum(X.*iV, 5).*erp_ga.V;
            erp_ga.x =  mean(X, 5);
          otherwise
            erp_ga.x= nanmean(X, 5);
      end
    else
      erp_ga.x= nanmean(X, 4);
  end  
else
  if isfield(erp_ga, 'yUnit') 
      switch erp_ga.yUnit
          case 'dB'
            erp_ga.x= nanmean(10.^(X/10), 4);
            erp_ga.x= 10*log10(erp_ga.x);
          case 'r'
            erp_ga.V = 1./sum(iV, 4);
            erp_ga.z = sum(atanh(X).*iV, 4).*sqrt(erp_ga.V);
            erp_ga.x= tanh(erp_ga.z.*sqrt(erp_ga.V));
            erp_ga.p = reshape(2*normal_cdf(-abs(erp_ga.z(:)), zeros(size(erp_ga.z(:))), ones(size(erp_ga.z(:)))), size(erp_ga.z));
            erp_ga.sgn_log10_p = reshape(((log(2)+normcdfln(-abs(erp_ga.z(:))))./log(10)), size(erp_ga.z)).*-sign(erp_ga.z);
          case 'z'
            erp_ga.x= mean(X, 4).*sqrt(size(X, 4));
            erp_ga.p = reshape(2*normal_cdf(-abs(erp_ga.x(:)), zeros(size(erp_ga.x(:))), ones(size(erp_ga.x(:)))), size(erp_ga.x));
            erp_ga.sgn_log10_p = reshape(((log(2)+normcdfln(-abs(erp_ga.x(:))))./log(10)), size(erp_ga.x)).*-sign(erp_ga.x);
          case 'auc'
            erp_ga.z = mean(Z, 4).*sqrt(size(Z, 4));
            erp_ga.p = reshape(2*normal_cdf(-abs(erp_ga.z(:)), zeros(size(erp_ga.z(:))), ones(size(erp_ga.z(:)))), size(erp_ga.z));
            erp_ga.sgn_log10_p = reshape(((log(2)+normcdfln(-abs(erp_ga.z(:))))./log(10)), size(erp_ga.z)).*-sign(erp_ga.z);
%             erp_ga.V = 1./sum(iV, 4);
%             erp_ga.x =  sum(X.*iV, 4).*erp_ga.V;
            erp_ga.x =  mean(X, 4);
          otherwise
            erp_ga.x= nanmean(X, 4);
      end
    else
      erp_ga.x= nanmean(X, 4);
  end  
end
if isfield(erp_ga, 'yUnit') & strcmp(erp_ga.yUnit, 'dB'),
  erp_ga.x= 10*log10(erp_ga.x);
end

erp_ga.clab= clab;
%% TODO should allow for weighting accoring to field N
%% (but this has to happen classwise)

erp_ga.title= 'grand average';
erp_ga.N= length(erps);
