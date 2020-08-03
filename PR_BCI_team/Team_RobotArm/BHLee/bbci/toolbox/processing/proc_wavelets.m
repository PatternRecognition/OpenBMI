function [dat info] = proc_wavelets(dat,varargin)
% PROC_WAVELETS -  calculates the continuous wavelet transform for a
% specified range of scales. Coefficients are obtained by convolution of 
% the EEG data with a (possibly complex) wavelet. (this is realized in the Fourier domain)
%
%Usage:
% dat = proc_wavelets(dat,<OPT>)
% dat = proc_wavelets(dat,freq,<OPT>)
%
%Arguments:
% DAT      -  data structure of continuous or epoched data (2D or 3D)
% OPT - struct or property/value list of optional properties:
% 'mother' -  mother wavelet (default 'morlet'). Morlet is currently the
%             only implemented wavelet.
% 'freq'   -  frequencies to be considered (default: alpha band [8:12])
% 'res' - 
% 'support' - since wavelets such as Morlet do not have compact 
%             support (ie, they are not zero at some point), they have to be cut off
%             at some point for filtering purposes. Support is given as
%             times the e-folding time, that is the time after which the
%             wavelet power falls off by e^-2. (default 2)
% 'norm' -    if set to 'unit', all time-frequency bins are
%             scaled to unit power and directly comparable. If set to
%             'amplitude', the amplitude directly reflects the amplitude in
%             the signal. (default 'unit')
% 'w0'    -   unitless frequency constant defining the trade-off between
%             frequency resolution and time resolution. For Morlet
%             wavelets, it sets the width of the Gaussian window. (default 7)
% 'vectorize' - vectorization is used for faster processing at the cost of
%             higher memory consumption (default 1)
%
%Returns:
% DAT    -    updated data structure with a higher dimension (3D or 4D).
%             For continuous data, the dimensions correspond to 
%             time x frequency x channels. For epoched data, time x
%             frequency x channels x epochs.
% INFO   -    struct with the following fields:
% .fun       - wavelet functions (in Fourier domain)
% .length    - length of each wavelet in time samples
% .freq      - wavelet center frequencies
%
% Interpretation of the data: Wavelet coefficients are complex, that is,
% consisting of a real part real(dat.x) and an imaginary part imag(dat.x). 
% Use abs(dat.x) and angle(dat.x) to get amplitude and phase spectra.
%
%Memory consumption: The dimensionality of the data is increased by one dimension,
% leading to a substantial increase in memory consumption. Selection
% of a small subset of electrodes and trials is strongly recommended.
%
% See also PROC_SPECTROGRAM.

% Author: Matthias Treder (2010,2012)

clear i norm psi

if numel(varargin)>0 && ~ischar(varargin{1})
  freq = varargin{1};
  varargin(1) = [];
else
  freq = 8:12;
end

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'mother','morlet', ...
                 'w0',7,...
                 'freq', freq, ...
                 'vectorize', 1, ...
                 'support', 2);


dt = 1/dat.fs;  % time resolution (determined by sampling frequency)
e = exp(1);  % Euler's number

% Prepare output stuff
info = struct();
info.fun = cell(1,numel(freq));
info.frequency = opt.freq;


% Needed for Fourier version
N = size(dat.x,1);      % Length of signal = length of FFT
siz = size(dat.x);
siz(1)=[];
if siz(end)==1; siz(end)=[]; end  % cope with vectors with singleton dimensions
w = [(2*pi*[1:floor(N/2)])/(N*dt)  -(2*pi*[floor(N/2)+1:N])/(N*dt)]; % Angular frequency for k<= N/2

F = fft(dat.x);
% dat.x = zeros([length(opt.freq) size(dat.x)]);
dat.x = zeros([N length(opt.freq) siz]);

%% Define mother wavelet, normalization factor, and fourier period
% [ and normalization ..]
switch(opt.mother)
  case 'morlet'
    psi0 = 'pi^(-1/4) * (w>0) .* e.^(-((s*w-opt.w0).^2)/2)'; % Mother wavelet
    scf = '(opt.w0+sqrt(2+opt.w0^2))/(4*pi*f)'; % Get scale as a function of Fourier frequency  
    efold = 'sqrt(2)*s'; % E-folding time
  otherwise
    error(['Wavelet ' opt.mother ' not implemented.'])
end

%% Define scales corresponding to the desired frequencies
scales = zeros(1,numel(opt.freq));    % Wavelet scales
for ii=1:numel(opt.freq)
  f=opt.freq(ii);
  scales(ii) = eval(scf);
end

%% Wavelet transform
for ii=1:numel(scales)
    % Traverse scales and create wavelet functions
    s = scales(ii);
    norm = sqrt( (2*pi*s)/dt );  % Normalization of wavelet in Fourier space
    fun = norm * eval(psi0)';
%     info.fun{ii} = fun;
    
    dat.x(:,ii,:,:) = ifft(F .* repmat(fun,[1 siz]));
    % Wavelet transform
%     if size(dat.x,3)==3
%       dat.x(ii,:,:) = ifft(F .* repmat(fun,[1 nchan]));
%     else % epoched data
%       dat.x(ii,:,:,:) = ifft(F .* repmat(fun,[1 nchan size(dat.x,4)]));
%     end
end
