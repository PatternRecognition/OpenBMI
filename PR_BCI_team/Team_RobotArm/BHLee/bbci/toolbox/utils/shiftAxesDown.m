function shiftAxesDown(down, hc)
%shiftAxesDown(<down=0.1, hc>)

if ~exist('down','var') | isempty(down), down=0.1; end
if ~exist('hc','var'), hc= get(gcf, 'children'); end

for hi= 1:length(hc);
  if isempty(get(hc(hi), 'tag')),  %% otherwise it might be a legend or such
    pos= get(hc(hi), 'position');
    pos(2)= (pos(2))*(1-down);
    pos(4)= pos(4)*(1-down);
    set(hc(hi), 'position', pos);
  end
end
