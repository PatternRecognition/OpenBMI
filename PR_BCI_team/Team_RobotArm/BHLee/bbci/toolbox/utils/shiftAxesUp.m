function shiftAxesUp(up, hc)
%shiftAxesUp(<up=0.1, hc>)

if ~exist('up','var') | isempty(up), up=0.1; end
if ~exist('hc','var'), hc= get(gcf, 'children'); end

for hi= 1:length(hc);
  tag= get(hc(hi), 'tag');
  if isempty(tag) || strcmp(tag, 'Colorbar'),
    pos= get(hc(hi), 'position');
    if length(pos)==4,  %% otherwise it might be a uicontextmenu
      pos(2)= 1 - (1-pos(2))*(1-up);
      pos(4)= pos(4)*(1-up);
      set(hc(hi), 'position', pos);
    end
  end
end
