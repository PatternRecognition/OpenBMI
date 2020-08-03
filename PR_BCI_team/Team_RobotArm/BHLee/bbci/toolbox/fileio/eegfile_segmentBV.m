function eegfile_segmentBV(file, varargin)
%  
% is this working? (SH)

global EEG_RAW_DIR

if length(varargin)==1,
  opt= struct('segment_groups',varargin{1});
else
  opt= propertylist2struct(varargin{:});
end

seg= readSegmentBorders(file, 'raw');
nSegments= length(seg);

opt= set_defaults(opt, ...
                  'segment_groups', num2cell(1:nSegments));
if ~iscell(opt.segment_groups),
  opt.segment_groups= {opt.segment_groups};
end

hdr= eegfile_readBVheader(file);
mrk= eegfile_readBVmarkers(file);
mrkpos= [mrk.pos];
for ii= 1:length(opt.segment_groups),
  cnt= [];
  idx= [];
  for jj= 1:length(opt.segment_groups{ii}),
    kk= opt.segment_groups{ii}(jj);
    ival_ms= seg.ival(kk,:)/seg.fs*1000;
    clear cnt_new;                         %% this is just to save memory
    cnt_new= eegfile_loadBV(file, 'ival', ival_ms);
    idx_new= find(mrkpos>=seg.ival(kk,1) & mrkpos<=seg.ival(kk,2));
    cnt= proc_appendCnt(cnt, cnt_new);
    idx= cat(2, idx, idx_new);
  end
  cnt.title= strcat(EEG_RAW_DIR, file, num2str(ii));
  eegfile_writeBV(cnt, mrk(idx), hdr.scale);
end
