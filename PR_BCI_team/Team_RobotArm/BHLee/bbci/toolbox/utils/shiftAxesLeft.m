function shiftAxesLeft(left, hc)
%shiftAxesLeft(<left=0.1, hc>)

if ~exist('left','var'), left=0.1; end
if ~exist('hc','var'), hc= get(gcf, 'children'); end

for hi= 1:length(hc);
%  if isempty(get(hc(hi), 'tag')),  %% otherwise it might be a legend or such
    pos= get(hc(hi), 'position');
    pos([1 3])= pos([1 3])*(1-left);
    set(hc(hi), 'position', pos);
%  end
end
