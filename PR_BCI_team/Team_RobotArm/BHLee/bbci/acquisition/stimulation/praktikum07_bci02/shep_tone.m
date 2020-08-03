function [y] = shep_tone (base_freq,halftone,varargin)
% generates shepard-tone or chord of shepards tones
% in:
% 
%          base_freq    frequency of tone around amplitude envelope is
%                       centered
%          
%          halftone     index of generated tone in halftone steps from center
%                       tone( as specified by base_freq)
%[y] = shep_tone ('PARAM1',val1, 'PARAM2',val2,...)
%   specifies one or more of the following name/value pairs:
%          tones:     number of partial tones (octaves)
%          duration:  duration of tone
%          fs:        sampling rate
%          max_phi:   max. value for phase shift of partial tones against
%                     each other. phase shift is equally distributed.
%                     gleichverteilt
%          sigma:     wideness of gaussian envelope, that specifies the
%                     amplitude of the partial tones
%          fade:      duration of fade in/fade out time relative to total
%                     duration 
%                     
%          norm:      normalization, default:0 (no normalization)
%
% bsp: shep_tone (440, 1,'tones', 7,'duration',.4, 'fs', 22100, 'max_phi', 0,'sigma', 4, 'fade', 0.1);
% bsp: shep_tone ('base_freq', 400, 'halftone', 3);
% bsp: shep_tone ('base_freq', 400, 'halftone', [0 4 7]);

%I. Sturm 1.11.07

opt = propertylist2struct (varargin{:});
opt = set_defaults (opt, 'tones', 7, 'duration', 1, 'fs', 22100, 'max_phi', 0.0, 'sigma', 2, 'fade', 0.08, 'norm', 0);

%number of partialtones must be odd 
if (mod (opt.tones,2) == 0)
    opt.tones = opt.tones - 1;
    disp (['Note! Number of tones is changed to ' num2str (opt.tones) '!']);
end
 

T = 1/opt.fs;
t = 0:T:opt.duration-T;

y = zeros (size (t));

for n = 1 : size (halftone, 2)
	octaves = (-(opt.tones-1)/2 : 1 : (opt.tones-1)/2)'; %base_freq is placed in the center of the partialtones
	% gaussian amplitude envelope
	A = exp((-1/(2*(opt.sigma^2)))*(log2(((base_freq*2^(halftone(n)/12)*2.^octaves)/(base_freq))).^2));

	octaves = 2 .^ octaves;
	octaves = repmat (octaves, 1, size (t, 2));

	x = repmat (t, opt.tones, 1) .* 2 .* pi .* base_freq .* 2^(halftone(n)/12) .* octaves;

	phase = unifrnd (0, opt.max_phi, opt.tones, 1);
	phases = repmat (phase, 1, size(t, 2));
	x = x + phases;

	y_ton = sin (x);

	% envelope in the frequency domain
	A = repmat (A, 1, size (t, 2));
	y_ton = y_ton .* A;

	% sum up partial tones
	y_ton = sum (y_ton);

	% envelope in the time domain
	len = size(y_ton, 2);
	uebergang = round (len * opt.fade);
	z = 1:uebergang;
	z = z / uebergang;
	y_ton(1:uebergang) = y_ton(1:uebergang) .* z;
	y_ton(end-uebergang+1:end) = y_ton(end-uebergang+1:end) .* fliplr (z);
	
	y = y + y_ton;
   
end

if (opt.norm)
	y = y / max (y);
end
 y=y';