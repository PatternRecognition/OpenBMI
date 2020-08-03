function C= cprintf(fmt, varargin)

if isempty(varargin),
  C= {};
  return;
end

len= apply_cellwise2(varargin, 'length');
if any(diff(len)),
  error('all arguments must have the same length');
end

N= len(1);
C= cell(N, 1);
Nv= length(varargin);
args= cell(1, Nv);
for n= 1:N,
  for m= 1:Nv,
    v= varargin{m};
    if iscell(v),
      args{m}= v{n};
    else
      args{m}= v(n);
    end
  end
  C{n}= sprintf(fmt, args{:});
end
