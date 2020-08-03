function out= ml_union(varargin)

if nargin==2,
  a= varargin{1};
  b= varargin{2};
  iab= ismember(a, b);
  iba= ismember(b, a);
  if all(iab),
    out= b;
  elseif all(iba),
    out= a;
  else
    if ~iab(1),
      c= a; a= b; b= c;      
      iab= ismember(a, b);
      iba= ismember(b, a);
    end
    out= a;
    im= find(iba),
    ptr= min(find(iba==0));
    for k= 1:length(im),
      if im(k)>ptr,
        iposa= find(ismember(out, b(im(k))));
        out= [out(1:iposa-1), b(ptr:im(k)-1), out(iposa:end)];
        ptr= min(find(iba==0 & 1:length(iba)>im(k)));
      end
    end
    if ~isempty(ptr),
      out= [out, b(ptr:end)];
    end
  end
elseif nargin==1,
  out= varargin{1};
else
  out= ml_union(ml_union(varargin{1}, varargin{2}), varargin{3:end});
end
