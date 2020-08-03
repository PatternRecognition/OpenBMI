function eegfile_onlinesaveMatlab_old(mode,varargin)
% eegfile_onlinesaveMatlab(mode,<dat,mrkPos,mrkToe>)
% 
% mode - 'init': A new FID is generated and memorized.
%        'store': write data to FID
%        'close': close FID.
% dat  - data array TIMExCHANNELS
% mrkPos - marker position array
% mrkToe - marker toe array
% opt  - contains a field 'subdir' (relative to EEG_RAW_DIR),
%        a field 'fs' and a field 'nChans'.
%        NOTE: This assumes that fs of mrk and data are the same.
% 
% init:
%      eegfile_onlinesaveMatlab('init',opt)
% store:
%      eegfile_onlinesaveMatlab('store',dat,mrkPos,mrkToe)
% close:
%      eegfile_onlinesaveMatlab('close')

% kraulem 10/06
persistent fid_mrk fid_dat counter
global EEG_RAW_DIR

switch mode
 case 'init'
  % generate a new FID
  opt = varargin{1};
  state= varargin{2};
  if ~exist([EEG_RAW_DIR opt.subdir],'dir')
    mkdir([EEG_RAW_DIR opt.subdir]);
  end
  dir = [EEG_RAW_DIR opt.subdir];
  poi = 1;
  ind = strfind(opt.subdir,'_');
  if isempty(ind),
    name= 'anonymos';
  else
    ind = ind(end-2);
    name = opt.subdir(1:ind-1);
  end
  while exist(sprintf('%seeg_%s_%03i.eeg', ...
		      dir,name,poi),'file'),
    poi = poi+1;
  end
  fid_mrk = fopen(sprintf('%seeg_%s_%03i.mrk', ...
		      dir,name,poi),'a+');
  fid_dat = fopen(sprintf('%seeg_%s_%03i.eeg', ...
		      dir,name,poi),'a+');
  % save some additional information:
  save(sprintf('%seeg_%s_%03i.mat', ...
		      dir,name,poi),'opt','state');
  counter = 0;
 case 'store'
  dat = varargin{1};
  bvdat = varargin{2};
  mrkPos = varargin{3};
  mrkToe = varargin{4};
  % write to FID
  fwrite(fid_dat, dat,'double');
  counter = counter+size(dat,1);
  % store marker information
  %get the offset (assume the last sample is the current one):
  offset = size(bvdat,1)-mrkPos;
  for ii = 1:length(mrkPos),
    toe= mrkToe{ii};
    sgn= (toe(1)=='S') - (toe(1)=='R');
    fwrite(fid_mrk,[counter-offset(ii),sgn*eval(toe(2:end))],'double');
  end
 case 'close'
  % close FID
  fclose(fid_mrk);
  fclose(fid_dat);
end
