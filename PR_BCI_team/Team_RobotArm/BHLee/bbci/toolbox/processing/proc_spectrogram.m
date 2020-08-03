function [dat] = proc_spectrogram(epo, band, chan, varargin)
%   [dat] = proc_spectrogram(epo, band, chan, varargin)
% 
%   IN: epo    - struct of epoched data
%       band   - range of the frequency range [minF maxF]Hz
%       chan   - cellarray/indexvalues/string with the selection of channels
%    
%       varargin - list of additional optional values, by default those
%                  values are 'Nfft' = 64, 'db_scaled' = 1, 'Window' =
%                  2*ceil(size(epo.x, 1)/10), 'Noverlap' = ceil(size(epo.x,
%                  1)/10)
%
%  OUT: dat   - struct different from classical epo or cnt struct,
%               but very similar sizeof(dat.x) = (freqs X time X chans X epochs)
%               
%               the number of time samples is determined by the parameters
%               Window und Noverlap. Each time sample in dat is the spectral
%               response within a window of length 'Window'. Window centers
%               start from the first sample and are shifted by 'Noverlap' 
%               samples. Spectral response for border windows are
%               calculated by padding the signal.
%               Per default, the Window length is set
%               to 1/10 of the signal length and the Overlap is 1/20 of the
%               signal length, leading to 11 windows with 1/2 overlap.
%
%               The number of frequency samples is determined by Nfft,
%               which should be less or equal than 'Window'. If it is
%               less than 'Window', the end of the window is not used 
%               by the fft. 
%               If somebody is interested, he or she could implement an
%               overlap-and-add scheme for within-window spectral power 
%               estimation (if 'db_scaled=1' option is set) that uses all data.
%
%
% in case of open questions read the source code or contact
%
% stl, haufe, Berlin 2004, 2010
%
% See also SPECTROGRAM, FFT, PROC_WAVELETS
  
  if ~isempty(varargin) && isnumeric(varargin{1}),
    %% arguments given as <win/N, step>
    opt.win= varargin{1};
    if length(varargin)>=2 && isnumeric(varargin{2}),
      opt.step= varargin{2};
    end
  else
    %% arguments given as <opt>
    opt= propertylist2struct(varargin{:});
  end
  opt= set_defaults(opt, 'Window', 2*floor(size(epo.x, 1)/10),  'db_scaled', 1,  'Nfft', 64, 'Noverlap', floor(size(epo.x, 1)/10));

  [T, nChan, nEvt] = size(epo.x) ;

  if ~exist('chan','var') || isempty(chan),
    chan = 1: nChan;
  else
    chan = chanind(epo,chan) ;
  end;

  
  [B,F,T] = spectrogram([zeros(opt.Window/2,1);epo.x(:,1,1); zeros(opt.Window/2-1,1)], opt.Window, opt.Noverlap, opt.Nfft, epo.fs);
  if ~exist('band','var') || isempty(band),
    bandIdx = 1:length(F) ;
  else
    bandIdx = find(F>=band(1) & F<=band(2)) ;
  end;
  
  dat   = copyStruct(epo,'x','clab','t');
  dat.clab = epo.clab(chan) ;
  dat.t = (T-opt.Window/2/epo.fs)*1000 + epo.t(1);
  dat.f = F(bandIdx);
  dat.zUnit = 'Hz';
  
  dat.x = zeros(length(bandIdx), length(T), length(chan));
  dat.x = zeros(length(bandIdx), length(T), length(chan), nEvt);

  if opt.db_scaled,
    for chIdx = 1: length(chan),
      ch = chan(chIdx) ;
      for evt = 1: nEvt,
        [B,F,T] = spectrogram([mean(epo.x(1:min(end,opt.Window),ch,evt))*ones(opt.Window/2,1); epo.x(:,ch,evt);mean(epo.x(max(1,end-opt.Window):end,ch,evt))*ones(opt.Window/2-1,1)], opt.Window, opt.Noverlap, opt.Nfft, epo.fs);     
        dat.x(:,:,chIdx, evt) = abs(B(bandIdx, :)).^2;
      end ;
      dat.x(:,:,chIdx, :) = 10*log10( dat.x(:,:,chIdx, :)+eps ) ;
    end ;
    dat.yUnit= 'dB';
  else
    for ch = chan,
      for evt = 1: nEvt,
        [B,F,T] = spectrogram([zeros(opt.Window/2,1);epo.x(:,ch,evt);zeros(opt.Window/2-1,1)], opt.Window, opt.Noverlap, opt.Nfft, epo.fs);      
        dat.x(:,:,chIdx, evt) = abs(B(bandIdx, :)).^2;
      end ;
    end ;
    dat.yUnit= 'power';
  end

