function ql = get_ql(Vnames)
if ~iscell(Vnames)
  Vnames = {Vnames};
end
ql = NaN(length(Vnames),1);
for n = 1:length(Vnames)
  vidname = Vnames{n};
  if length(vidname)==33
    ql(n) = 33;  % HQ
  else
    if strcmpi(vidname(1:9),'bigWaffle')
      ql(n) = 16;
    else
      if vidname(38)=='.'
        ql(n) = str2double(vidname(37));
      else
        ql(n) = str2double(vidname(37:38));
      end
    end
  end
end
