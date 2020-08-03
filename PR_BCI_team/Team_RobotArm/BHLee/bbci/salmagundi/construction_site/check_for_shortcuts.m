function check_for_shortcuts(cnt, varargin)
%CHECK_FOR_SHORTCUTS - Generate some plots to check for electrode shortcuts
%
%Synopsis:
% check_for_shortcuts(FILENAME, <OPT>)
% check_for_shortcuts(CNT, <OPT>)
%
%Arguements:
% FILENAME: Name of a raw EEG file
% CNT: Struct of continuous EEG signals
% OPT: struct or property/value list of optional properties
%   .thresh: Correlation coefficients above this threshold are considered
%       to be shortcuts. Default ?
%   .ival: Time interval (msec) to be read from file (if FILENAME given)
%   .band: If non-empty, a butterworth band-pass filter is applied to the data
%       with the specified band, default [5 40]
%   .reject_channels: reject bad channels, default 1.
%   .lar: If true, local average reference is applied, default 1.

% Author: Benjamin Blankertz


opt= propertylist2struct(varargin{:});

%cnt= 'temp/bridge_test'; opt= [];

opt= set_defaults(opt, ...
                  'band', [5 40], ...
                  'thresh', 0.98, ...
                  'lar', 1, ...
                  'reject_channels', 1, ...
                  'ival', [0 inf]);

if ischar(cnt),
  file= cnt;
  hdr= eegfile_readBVheader(file);
  Wps= [40 49]/hdr.fs*2;
  [n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
%  W= [1 2 40 49]/hdr.fs*2
%  [n, Ws]= cheb2ord(W([2 3]), W([1 4]), 3, 24);
  [filt.b, filt.a]= cheby2(n, 50, Ws);
  [cnt,mrk]= eegfile_readBV(file, 'fs',100, 'filt',filt, ...
                            'clab',{'not','E*'}, ...
                            'ival', opt.ival);
end
if ~isempty(opt.band),
  [b,a]= butter(5, opt.band/cnt.fs*2);
  cnt= proc_filt(cnt, b, a);
end
clab= cnt.clab;
if opt.reject_channels,
  winlen= 2000;
  winlen_sa= winlen/1000*cnt.fs;
  mk.fs= cnt.fs;
  mk.pos= 1:winlen_sa:size(cnt.x,1)-winlen_sa;
  opt_rej= strukt('do_bandpass', 0, ...
                  'whiskerlength', 1.5);
  [dmy, rClab]= reject_varEventsAndChannels(cnt, mk, [0 winlen_sa], opt_rej);
  cnt= proc_selectChannels(cnt, 'not',rClab);
end
if opt.lar,
  mnt= getElectrodePositions(cnt.clab);
  cnt= proc_localAverageReference(cnt, mnt);
end

[C,p]= corrcoef(cnt.x); 
figure(1);
subplot(2,2,1);
imagesc(C); colorbar;
set(gca, 'CLim', [-1 1]);
subplot(2,2,2);
D= abs(C-eye(size(C)));
%[nn,x]= hist(D(:), 50);
[nn,x]= hist(D(:), [0:0.01:0.99]+0.005);
hb= bar(x, nn, 1, 'FaceColor','y');
hold on;
nn2= nn;
nn2(find(x<opt.thresh))= 0;
hb= bar(x, nn2, 1, 'FaceColor','r');
hl= line(opt.thresh*[1 1], ylim, 'Color','k');
moveObjectBack(hl);
hold off;
subplot(2,2,4);
plot(sort(D(:)), 'LineWidth',1.25);
hl= line(xlim, opt.thresh*[1 1], 'Color',0.5*[1 1 1]);
moveObjectBack(hl);
subplot(2,2,3);
E= C;
E(find(C<opt.thresh))= NaN;
imagesc(E); colorbar
%set(gca, 'CLim',[opt.thresh-0.5*(1-opt.thresh) 1]);
set(gca, 'CLim',[opt.thresh-0.001 1]);

D= abs(C-eye(size(C)));
Dl= abs(tril(C, -1));
[so,si]= sort(Dl(:), 1, 'descend');
idx= si(find(so>opt.thresh));
ii= 0;
cls= {};
while length(idx)>0,
  ii= ii+1;
  jj= idx(1);
  [c1,c2]= ind2sub(size(D), jj);
  cls{ii}= sort([c1 c2]);
  cc= 0;
  while cc<length(cls{ii}),
    cc= cc+1;
    n= find(D(cls{ii}(cc),:)>opt.thresh);
    cls{ii}= unique([cls{ii} n]);
  end
  fprintf('cluster %d: %s\n', ii, vec2str(clab(cls{ii})));
  szd= size(D,1);
  for cc= cls{ii},
    idx(find(ismember(idx, cc:szd:szd^2)))= [];
    idx(find(ismember(idx, 1+(cc-1)*szd:cc*szd)))= [];
  end
  D(cls{ii},:)= 0;
  D(:,cls{ii})= 0;
end

nCluster= length(cls);
clusterno= NaN*zeros(length(clab), 1);
for ci= 1:nCluster,
  clusterno(cls{ci})= ci;
end
mnt= getElectrodePositions(clab);
figure(2);
clf;
colormap(jet(nCluster));
H= showScalpLoading(mnt, clusterno, 1, 'none', [0.5 nCluster+0.501]);
set(H.hcb, 'YTick',1:clusterno);

if exist('file', 'var'),
  axis_title(untex(file));
end
