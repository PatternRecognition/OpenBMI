function S= strukt(varargin)

C= varargin;
for ii= 2:2:length(C),
  if iscell(C(ii)),
    C(ii)= {C(ii)};
  end
end
S= struct(C{:});
