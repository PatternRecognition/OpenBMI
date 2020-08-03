function hsp= grid_getSubplots(chans)
%hsp= grid_getSubplots(<chans>)
%
% returns the handles of the subplots of an grid plot (cf. grid_plot)
% that correspond to given channels
%
% IN  - channels, [] means all, default []

if ~exist('chans','var'), chans=[]; end
if ~isempty(chans) & ~iscell(chans),
  chans= {chans};
end

if length(chans)>0 & ischar(chans{1}) & ...
      strcmp(chans{1},'plus'),
  search_type= 'ERP*';
  chans= chans(2:end);
else
  search_type= 'ERP';
end

hc= get(gcf, 'children');
isERPplot= zeros(size(hc));

for ih= 1:length(hc),
  ud= get(hc(ih), 'userData');
  if isstruct(ud) & isfield(ud,'type') & ischar(ud.type) & ...
        strpatterncmp(search_type, ud.type), 
    if isempty(chans) | ~isempty(chanind(chans, ud.chan)),
      isERPplot(ih)= 1;
    end
  end
end

hsp= hc(find(isERPplot))';
