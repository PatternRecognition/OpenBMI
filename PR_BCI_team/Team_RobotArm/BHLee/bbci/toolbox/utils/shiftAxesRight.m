function shiftAxesRight(shift, hc)
%shiftAxesRight(<shift=0.1, hc>)

if ~exist('shift','var'), shift=0.1; end
if ~exist('hc','var'), hc= get(gcf, 'children'); end

for hi= 1:length(hc);
%  if isempty(get(hc(hi), 'tag')),  %% otherwise it might be a legend or such
    pos= get(hc(hi), 'position');
    pos(1)= 1 - (1-pos(1))*(1-shift);
    pos(3)= pos(3)*(1-shift);
    set(hc(hi), 'position', pos);
%  end
end
