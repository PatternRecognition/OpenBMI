function qq= fractileValues(x, varargin)
%qq= fractileValues(x, <p=25, dim>)
%qq= fractileValues(x, <opt>)
%
% vector of x values at [0 p 50 100-p 100] percent fractiles.

if ~isempty(varargin),
  if ischar(varargin{1}),
    opt= propertylist2struct(varargin{:});
  else
    p= varargin{1};
    opt= struct('perc', [0 p 50 100-fliplr(p) 100]);
    if length(varargin)>1,
      opt.dim= varargin{2};
    els
    end
  end
end

xs= size(x);
opt= set_defaults(opt, ...
                  'perc', [0 25 50 75 100], ...
                  'dim', min(find(xs~=1)));

if isempty(opt.dim), 
  qq= x; 
  return;
end                            


if prod(xs)>xs(opt.dim),
  otherDims= setdiff(1:length(xs), opt.dim);
  x= permute(x, [opt.dim otherDims]);
  K= prod(xs(otherDims));
  qq= zeros(3+2*length(opt.perc), K);
  for k= 1:K,
    qq(:,k)= fractileValues(x(:,k), opt, 'dim',1);
  end
  qq= ipermute(qq, [opt.dim otherDims]);
  return;
end

intersects= 1+round((xs(opt.dim)-1)*opt.perc/100);
[xso, xsi]= sort(x);
qq= x(xsi(intersects));
