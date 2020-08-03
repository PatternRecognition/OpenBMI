function out= proc_coherence(dat, freq, varargin)
%PROC_COHERENCE - Event-Related Coherence between channels
%
%Usage:
% EPO= proc_coherence(EPO, FREQ, <opt>) 
%
%Description:
% This function calculates the event-related coherence between speficied
% channel pairings. The default output is the complex valued coherence,
% but different options for creating a real valued output exist (opt.output).
%
%Input:
% EPO   - data structure of epoched data
% FREQ  - frequency of interest [Hz]
% OPT - struct or property/value list of optional properties:
%  .win   - fourier window
%           if OPT.win is a scalar, a square (boxcar) window of that
%           length is used. Default value: EPO.fs (1s square window).
%  .step  - window step size, default: 1.
%  .normalize - 1 or 0: normalize coherence or not.
%  .output - may be one of {'complex', 'real', 'imag', 'abs', 'angle',
%           'cartesian', 'polar'}. 'magnitude' is the same as 'abs'.
%  .pairing - string: pair of channels, e.g. 'C3-C4', or 
%            cell array string: list of channel pairings, e.g.,
%            {'C3-C4', 'C3-Cz', 'C4-Cz', 'C3-P3', 'C4-P4', 'P3-P4'}.
%            'ALL' is a shorthand for all possible channel pairings.
%            When EPO contains only two channels, opt.pairing has not
%            to be set.
%
%Output:
% EPO   - updated data structure

% Author(s): Benjamin Blankertz, Apr 2005

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'win', dat.fs, ...
                  'step', 1, ...
                  'normalize', 1, ...
                  'output', 'complex', ...
                  'pairing', []);

[T, nChans, nEvents]= size(dat.x);
if ~iscell(opt.pairing),
  if strcmp(upper(opt.pairing), 'ALL'),
    kk= 0;
    opt.pairing= cell(1, nChans*(nChans-1)/2);
    for c1= 1:nChans,
      for c2= c1+1:nChans,
        kk= kk+1;
        opt.pairing{kk}= [dat.clab{c1} '-' dat.clab{c2}];
      end
    end
  elseif isempty(opt.pairing),
    if nChans~=2,
      error('more than 2 channels: specify a channel pairing in opt.pairing');
    end
    opt.pairing= {[dat.clab{1} '-' dat.clab{2}]};
  else
    opt.pairing= {opt.pairing};
  end
end

if length(opt.pairing)==1,
  joStr= opt.pairing{1};
  is= find(joStr=='+' | joStr=='-');
  if length(is)~=1,
    error(sprintf('unexpected string in opt.pairing: %s\n', joStr));
  end
  chan1= chanind(dat, joStr(1:is-1));
  chan2= chanind(dat, joStr(is+1:end));
else
  if strcmp(opt.output, 'complex'),
    out_x= complex(ones([1 length(opt.pairing) nEvents]));
  elseif ismember(opt.output, {'cartesian', 'polar'}),
    out_x= ones([2 length(opt.pairing) nEvents]);
  else
    out_x= ones([1 length(opt.pairing) nEvents]);
  end
  for cc= 1:length(opt.pairing),
    oo= proc_coherence(dat, freq, opt, 'pairing', opt.pairing{cc});
    out_x(:,cc,:)= oo.x;
    if cc==1,
      out= oo;
    else
      out.clab= cat(2, out.clab, oo.clab);
    end
  end
  out.x= out_x;
  return;
end


if length(opt.win)==1,
  opt.win= ones(opt.win,1);
end
opt.win= opt.win(:)';
N= length(opt.win);
if N>T, error('window longer than signals'); end

[mm,bInd]= min(abs(freq-[(0:N/2)*dat.fs/N]));

nWindows= 1 + max(0, floor((T-N)/opt.step));
X= zeros(nWindows, nEvents);
Y= zeros(nWindows, nEvents);
iv= 1:N;
ExpWin= opt.win .* exp(-2*pi*i*(bInd-1)*(0:N-1)/N);
for wi= 1:nWindows,
  X(wi,:)= ExpWin * squeeze( dat.x(iv, chan1, :) );
  Y(wi,:)= ExpWin * squeeze( dat.x(iv, chan2, :) );
  iv= iv + opt.step;
end
xo= mean(X.*conj(Y), 1);
if opt.normalize,
  xo= xo ./ sqrt(mean(X.*conj(X),1)) ./ sqrt(mean(Y.*conj(Y),1));
end
out= copy_struct(dat, 'not', 'x','t','yUnit','clab');

xo= reshape(xo, [1, 1, nEvents]);
switch(lower(opt.output)),
 case 'complex',
  out.x= xo;
 case 'real',
  out.x= real(xo);
 case {'imag', 'imaginary'},
  out.x= imag(xo);
 case 'angle',
  out.x= angle(xo);
 case {'abs', 'magnitude'},
  out.x= abs(xo);
 case 'cartesian',
  out.x= cat(1, real(xo), imag(xo));
 case 'polar',
  out.x= cat(1, abs(xo), angle(xo));
 otherwise,
  error(sprintf('%s is unknown as policy for opt.output', opt.output));
end

out.t= dat.t(end);
out.clab= {[dat.clab{chan1} '-' dat.clab{chan2}]};
