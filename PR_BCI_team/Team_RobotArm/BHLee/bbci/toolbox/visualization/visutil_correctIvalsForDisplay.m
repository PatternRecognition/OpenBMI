function ival= visutil_correctIvalsForDisplay(ival, varargin)

if length(varargin)==1,
  varargin= {'fs', varargin{1}};
end
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'fs', 100, ...
                  'sort', 1, ...
                  'only_point_intervals', 0);

if opt.sort,
  [so,si]= sort(ival(:,1));
  ival= ival(si,:);
end

if opt.only_point_intervals,
  to_be_checked= find(diff(ival, 1, 2)==0)';
else
  to_be_checked= 1:size(ival, 1);
end

for ii= to_be_checked,
  if ii==1 | ival(ii-1,2)<ival(ii,1),
    ival(ii,1)= ival(ii,1) - 1000/opt.fs/2;
  end
  if ii==size(ival,1) | ival(ii,2)<ival(ii+1,2),
    ival(ii,2)= ival(ii,2) + 1000/opt.fs/2;
  end
end
