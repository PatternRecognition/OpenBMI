function blk= getSegmentBorders(file, fs)

[mrk_orig, fs_orig, nfo]= ...
    eegfile_loadMatlab(file, 'vars',{'mrk_orig','fs_orig','nfo'});
blk= struct('fs', nfo.fs);
lag= fs_orig/blk.fs;

is= strmatch('New Segment', mrk_orig.type, 'exact');
seg_start= ceil([mrk_orig.pos(is)]/lag);
nSeg= length(seg_start);
iv= [1:nSeg-1; 2:nSeg]';
blk.ival= [seg_start(iv); seg_start(end) nfo.T+1];
blk.ival(:,2)= blk.ival(:,2)-1;
