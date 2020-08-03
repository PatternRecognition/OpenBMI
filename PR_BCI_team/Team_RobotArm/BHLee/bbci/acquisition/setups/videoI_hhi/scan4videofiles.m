% List all AVI files from video directory
% and save filenames for different conditions in cell arrays
%

%% Get all avi files
s = dir(fullfile(videodir,'*.avi'));
s = struct2cell(rmfields(s,'date','bytes','isdir','datenum'));
s = sort(s); % sort alphabetically

%% Get HQ vids
hqvideos = {};
hqidx = regexp(s,'.*HQ_only.*');
hqidx = apply_cellwise(hqidx,'isempty');
hqidx = ~cell2mat(hqidx);
hqvideos = s(hqidx);
s(hqidx) = [];

%% Get according LQ vids
videos = cell(1,nQuality);
cnt = 1;
for ii=1:numel(hqvideos) 
    for q=1:nQuality
      cvid = videos{1,q};
      if ~iscell(cvid), cvid={}; end
      videos{1,q} = {cvid{:} s{cnt}};
      cnt=cnt+1;
    end
end
