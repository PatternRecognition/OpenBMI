function shiftAxesCenter(f, hc)
%shiftAxesCenter(<f=0.1, hc>)

if ~exist('f','var') | isempty(f), f=0.1; end
if ~exist('hc','var'), hc= get(gcf, 'children'); end

for hi= 1:length(hc);
  if isempty(get(hc(hi), 'tag')),  %% otherwise it might be a legend or such
    pos= get(hc(hi), 'position');
    pos([1 2])= 0.5 + (pos([1 2])-0.5)*(1-f/2);
    pos([3 4])= pos([3 4])*(1-f);
    set(hc(hi), 'position', pos);
  end
end
