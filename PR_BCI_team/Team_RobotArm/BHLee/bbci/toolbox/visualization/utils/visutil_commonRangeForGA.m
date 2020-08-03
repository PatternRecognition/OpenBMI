function range= visutil_commonRangeForGA(erp, varargin)

opt= propertylist2struct(varargin{:});
[opt, isdefault]= set_defaults(opt, ...
                               'clab_erp', '', ...
                               'ival_erp', [], ...
                               'sym_erp', 0, ...
                               'nice_range_erp', 0, ...
                               'enlarge_range_erp', 0.02, ...
                               'clab_scalp', '', ...
                               'ival_scalp', [], ...
                               'sym_scalp', 1);

if opt.nice_range_erp && isdefault.enlarge_range_erp,
  opt.enlarge_range_erp= 0;
end

range.erp= [inf -inf];
if ~isempty(opt.ival_scalp),
  range.scalp= [inf -inf];
end

for ff= 1:size(erp,2),
  erp_ga= proc_grandAverage2(erp{:,ff});
  
  tmp= proc_selectChannels(erp_ga, opt.clab_erp);
  if ~isempty(opt.ival_erp),
    tmp= proc_selectIval(tmp, opt.ival_erp);
  end
  range.erp(1)= min(range.erp(1), min(tmp.x(:))); 
  range.erp(2)= max(range.erp(2), max(tmp.x(:))); 
  
  if ~isempty(opt.ival_scalp),
    tmp= proc_jumpingMeans(erp_ga, opt.ival_scalp);
    tmp= proc_selectChannels(tmp, opt.clab_scalp);
    range.scalp(1)= min(range.scalp(1), min(tmp.x(:))); 
    range.scalp(2)= max(range.scalp(2), max(tmp.x(:))); 
  end

end

if opt.sym_erp,
  range.clab= [-1 1]*max(abs(range.clab));
end
if opt.enlarge_range_erp>0,
  range.erp= range.erp + [-1 1]*opt.enlarge_range_erp*diff(range.erp);
end
if opt.nice_range_erp>0,
  res= opt.nice_range_erp;
  range.erp(1)= floor(res*range.erp(1))/res;
  range.erp(2)= ceil(res*range.erp(2))/res;
end
if ~isempty(opt.ival_scalp) && opt.sym_scalp,
  range.scalp= [-1 1]*max(abs(range.scalp));
end
