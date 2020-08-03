function [b, a]= getButterFilter(band, fs, transition, Rp, Rs)
%[b, a]= getFIRfilter(band, fs, <transition_width, Rp, Rs>)
%[b, a]= getFIRfilter(band, dat, <transition_width, Rp, Rs>)

if ~exist('transition','var'), transition= [2 3]; end
if ~exist('Rp','var'), Rp= 3; end
if ~exist('Rs','var'), Rs= 24; end
if isstruct(fs), fs= fs.fs; end
if length(transition)==1, transition= transition*[1 1]; end

wp= band/fs*2;
ws= (band + ([-1 1].*transition))/fs*2;
[n, wn]= buttord(wp, ws, Rp, Rs);
[b,a]= butter(n, wn);
