function eegfile_onlinesaveMatlab(mode,varargin)
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
%      eegfile_onlinesaveMatlab('init',opt,state)
% store:
%      eegfile_onlinesaveMatlab('store',dat,mrkPos,mrkToe)
% close:
%      eegfile_onlinesaveMatlab('close')

% kraulem 10/06
persistent fid_mrk fid_dat counter mrk_counter
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
    name= 'anonymous';
  else
    ind = ind(end-2);
    name = opt.subdir(1:ind-1);
  end
  while exist(sprintf('%seeg_%s_%03i.eeg', ...
		      dir,name,poi),'file'),
    poi = poi+1;
  end
  filename = sprintf('%seeg_%s_%03i', ...
		     dir,name,poi);
  % Markers:
  fid_mrk = fopen([filename '.vmrk'],'a+');
  fprintf(fid_mrk, ['Brain Vision Data Exchange Marker File, Version 1.0' 13 10]);
  fprintf(fid_mrk, [13 10 '[Common Infos]' 13 10]);
  fprintf(fid_mrk, ['DataFile=%s.eeg' 13 10], filename);
  fprintf(fid_mrk, [13 10 '[Marker Infos]' 13 10]);
  fprintf(fid_mrk, ['Mk1=New Segment,,1,1,0,00000000000000000000' 13 10]);

  % Data:
  fid_dat = fopen([filename '.eeg'],'a+');
  
  % Header:
  fid_hdr = fopen([filename '.vhdr'],'w');
  if fid_hdr==-1, 
    error(sprintf('cannot write to %s.vhdr',filename)); 
  end
  fprintf(fid_hdr, ['Brain Vision Data Exchange Header File Version 1.0' 13 10]);
  fprintf(fid_hdr, ['; Data exported from BBCI Matlab Toolbox' 13 10]);
  fprintf(fid_hdr, [13 10 '[Common Infos]' 13 10]);
  fprintf(fid_hdr, ['DataFile=%s.eeg' 13 10], filename);
  fprintf(fid_hdr, ['MarkerFile=%s.vmrk' 13 10], filename);
  fprintf(fid_hdr, ['DataFormat=BINARY' 13 10]);
  fprintf(fid_hdr, ['DataOrientation=MULTIPLEXED' 13 10]);
  fprintf(fid_hdr, ['NumberOfChannels=%d' 13 10], opt.nChans);
  fprintf(fid_hdr, ['DataPoints=%d' 13 10], -1);% not known yet.
  fprintf(fid_hdr, ['SamplingInterval=%g' 13 10], 1000000/opt.fs);
  fprintf(fid_hdr, [13 10 '[Binary Infos]' 13 10]);
  fprintf(fid_hdr, ['BinaryFormat=DOUBLE' 13 10]);
  fprintf(fid_hdr, ['UseBigEndianOrder=NO' 13 10]);
  fprintf(fid_hdr, [13 10 '[Channel Infos]' 13 10]);
  for ic= 1:opt.nChans,
    fprintf(fid_hdr, ['Ch%d=%s,,%g' 13 10], ic, state.clab{ic}, 1);
  end
  fprintf(fid_hdr, ['' 13 10]);
  fclose(fid_hdr);
  % save some additional information:
  save([filename '.mat'],'opt','state');
  % initialize counter:
  counter = 0;
  mrk_counter = 1;
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
    mt = sgn*eval(toe(2:end));
    %fwrite(fid_mrk,[counter-offset(ii),sgn*eval(toe(2:end))],'double');
    %fwrite(fid_mrk,[counter-offset(ii),sgn*eval(toe(2:end))],'double');
    if mt>0,
      fprintf(fid_mrk, ['Mk%d=Stimulus,S%3d,%d,1,0' 13 10], ...
	      mrk_counter+1, mt, counter-offset(ii));
    else
      fprintf(fid_mrk, ['Mk%d=Response,R%3d,%d,1,0' 13 10], ...
	      mrk_counter+1, mt, counter-offset(ii));
    end
  end
 case 'close'
  % close FID
  fclose(fid_mrk);
  fclose(fid_dat);
end
