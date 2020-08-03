function [dat] = proc_specgram(epo, band, chan, varargin)
%   [dat] = proc_specgram(epo, band, chan, varargin)
% 
%   IN: epo    - struct of epoched data
%       band   - range of the frequency range [minF maxF]Hz
%       chan   - cellarray/indexvalues/string with the selection of channels
%    
%       varargin - list of additional optional values, by default those
%                  values are 'Nfft' = 64, 'db_scaled' = 1 
%
%  OUT: dat   - struct different from classical epo or cnt struct,
%               but very similar sizeof(dat.x) = (freqs X time X chans)
%               dat.x contains the averaged spectrograms
%
% in case of open questions read the source code or contact
%
% stl, Berlin Aug. 2004
  
%
% see also SPECGRAM, FFT
    
warning('This is function is deprecated and will be removed soon: please use proc_spectrogram instead')
  
  
  if ~isempty(varargin) & isnumeric(varargin{1}),
    %% arguments given as <win/N, step>
    opt.win= varargin{1};
    if length(varargin)>=2 & isnumeric(varargin{2}),
      opt.step= varargin{2};
    end
  else
    %% arguments given as <opt>
    opt= propertylist2struct(varargin{:});
  end
  opt= set_defaults(opt, 'win', [],  'db_scaled', 1,  'Nfft',64);

  [T, nChan, nEvt] = size(epo.x) ;

  if ~exist('chan','var') | isempty(chan),
    chan = 1: nChan;
  else
    chan = chanind(epo,chan) ;
  end;

  
  [B,F,T] = specgram([zeros(opt.Nfft/2,1);epo.x(:,1,1); zeros(opt.Nfft/2-1,1)], opt.Nfft, epo.fs, [], opt.Nfft-1);
  if ~exist('band','var') | isempty(band),
    bandIdx = 1:length(F) ;
  else
    bandIdx = find(F>=band(1) & F<=band(2)) ;
  end;
  
  dat   = copyStruct(epo,'x','clab','t');
  dat.clab = epo.clab(chan) ;
  dat.t = epo.t ;
  dat.f = F(bandIdx);
  dat.zUnit = 'Hz';
  
% begin sthf
  dat.x = zeros(length(bandIdx), length(T), length(chan));
  dat.x = zeros(length(bandIdx), length(T), length(chan), nEvt);
% end sthf  

  if opt.db_scaled,
    for chIdx = 1: length(chan),
      ch = chan(chIdx) ;
      for evt = 1: nEvt,
	[B,F,T] = specgram([mean(epo.x(1:min(end,opt.Nfft),ch,evt))*ones(opt.Nfft/2,1); epo.x(:,ch,evt);mean(epo.x(max(1,end-opt.Nfft):end,ch,evt))*ones(opt.Nfft/2-1,1)], opt.Nfft, epo.fs, [], opt.Nfft-1);
% begin sthf        
	dat.x(:,:,chIdx, evt) = abs(B(bandIdx, :)).^2;
      end ;
      dat.x(:,:,chIdx, :) = 10*log10( dat.x(:,:,chIdx, :)+eps ) ;
% end sthf
    end ;
    dat.yUnit= 'dB';
  else
    for ch = chan,
      for evt = 1: nEvt,
	[B,F,T] = specgram([zeros(opt.Nfft/2,1);epo.x(:,ch,evt);zeros(opt.Nfft/2-1,1)], opt.Nfft, epo.fs, [], opt.Nfft-1);
% begin sthf        
        dat.x(:,:,chIdx, evt) = abs(B(bandIdx, :)).^2;
% end sthf
      end ;
    end ;
    dat.yUnit= 'power';
  end

