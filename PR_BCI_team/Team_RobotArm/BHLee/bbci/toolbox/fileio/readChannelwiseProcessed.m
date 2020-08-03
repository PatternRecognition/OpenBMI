function cnt_all= readChannelwiseProcessed(file, proc, varargin)
%cnt= readChannelwiseProcessed(file, proc, args, ...)
%
% IN   file  - file name (no extension),
%              relative to EEG_RAW_DIR unless beginning with '/'
%      proc  - 
%      args  - passed to eegfile_loadBV
%
% OUT  cnt      struct for contiuous signals
%         .x    - EEG signals (time x channels)
%         .clab - channel labels
%         .fs   - sampling interval

nChans= 0;
for ig= 1:length(proc.chans),
  nChans= nChans + length(proc.chans{ig});
end

cc= 0;
if iscell(file),
  hdr= eegfile_readBVheader(file{1});
else
  hdr= eegfile_readBVheader(file);
end
for ig= 1:length(proc.chans),
  chans= hdr.clab(chanind(hdr,proc.chans{ig}));
%  [func,param]= getFuncParam(proc.exec{ig});
  
  for ic= 1:length(chans),
    cc= cc+1; fprintf('\r%d ', cc);
    cnt= eegfile_loadBV(file, 'clab',chans(ic), varargin{:});
%    cnt= feval(func, cnt, param{:});
    eval(proc.eval{ig});
    if cc==1,
      cnt_all= copyStruct(cnt, 'x','clab');
      cnt_all.x= zeros(size(cnt.x,1), nChans);
      cnt_all.clab= cell(1, nChans);
    end
    cnt_all.x(:,cc)= cnt.x;
    cnt_all.clab{cc}= cnt.clab{1};
  end
end
fprintf('\r    \n');
