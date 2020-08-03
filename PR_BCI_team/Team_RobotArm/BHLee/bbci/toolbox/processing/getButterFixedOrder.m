function [b, a]= getButterFixedOrder(band, fs, order);
%[b, a]= getButterFixedOrder(band, fs, <order=8>);
%[b, a]= getButterFixedOrder(band, dat, <order=8>);

if ~exist('order', 'var'), order=8; end

if isstruct(fs), fs= fs.fs; end

[b,a]= butter(order, band/fs*2);
