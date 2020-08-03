function dat_fft = plot_spectrum(dat, band, varargin)
% Computes the spectrum within a given frequency range (band) and plots it.
%
% In:
%   dat     - can be cnt or epo
%   band    - the frequency range within the fft is computed
%
% Out:
%   dat_fft - struct that contains the fft data
%
% 
% Sven Daehne, 01.2011, sven.daehne@tu-berlin.de

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt ...
    ,'log_scale', 0 ...
    ,'fft_window', kaiser(dat.fs) ...
    ,'show_channel_labels', 1 ...
);



if nargin < 2
    band = [0, 50];
end

if isfield(dat, 'xUnit') && strcmp(dat.xUnit, 'Hz')
    dat_fft = dat;
    idx = (dat.t >= band(1)) & (dat.t <= band(2));
    dat_fft.x = dat_fft.x(idx,:,:);
    dat_fft.t = dat_fft.t(idx);
else
    dat_fft = proc_spectrum(dat, band, opt.fft_window);
end

if size(dat.x) > 2
    X = mean(dat_fft.x, 3);
%     std_spectrum = std(dat_fft.x, [], 3);
else
    X = dat_fft.x;
end

if opt.log_scale
    X = log10(X);
end

imagesc(dat_fft.t, 1:size(X,2), X');
colorbar
xlabel('frequency [Hz]')
ylabel('channels')
if opt.show_channel_labels && isfield(dat_fft, 'clab')
    set(gca, 'yTick', 1:size(X,2));
    set(gca, 'ytickLabel', dat_fft.clab);
end
