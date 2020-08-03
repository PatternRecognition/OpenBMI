function plot_classifierweights(fv,w, color, varargin)
%PLOT_CLASSIFIERWEIGHTS plots classification outputs for selfpaced-like
%features
%
% usage:
%      plot_classifierweights(fv,w,color,chan);
%
% input:
%      fv     the non-concatenated feature vectors
%      w      the weighting as vector on the concatened features
%      color  1 = colored, = gray
%      chan   channels given similar as for proc_selectChannels
%             the follwing arguments are also possible:
%             'channelimportance' or 'ci': means over time are given
%             'timeimportance' or 'ti': means over channels are given
%
% Guido DOrnhege, 31/07/03

if ~exist('color','var') | isempty(color)
    color = 1;
end

w = reshape(w,[size(fv.x,1),size(fv.x,2)]);

ci = 0;ti = 0;
if sum(strcmp('channelimportance',varargin))+sum(strcmp('ci',varargin)) > 0
  ci = 1;
end

if sum(strcmp('timeimportance',varargin))+sum(strcmp('ti',varargin)) > 0
  ti = 1;
end

if length(varargin)-ci-ti==0
    chan = 1:length(fv.clab);
else
    chan = chanind(fv.clab,varargin{:});
end

w = w(:,chan);

cl = {fv.clab{chan}};

cw = 1.8;
wd = 1.2;
  
nT = size(w,1);
nC = size(w,2);

h= axes('position',[0 0.04+0.96*wd*ci/(nT+wd*ci),  0.96/(nC+cw+wd*ti) 0.96-0.96*wd*ci/(nT+wd*ci)]);


if ci
  hi = axes('position',[cw*0.96/(nC+cw+wd*ti),0.04,0.96*nC/(nC+cw+wd*ti),0.96/(nT+wd*ci)]);
  imagesc(sum(w,1));
  set(hi,'YTick',[]);
  set(hi,'XTick',1:size(w,2));
  set(hi,'XTickLabel',cl);
  if ~color
    colormap(1-gray);
  end
end

if ti
  hi = axes('position',[0.96*(nC+cw+wd-1)/(nC+cw+wd*ti),0.04+0.96*wd*ci/(nT+wd*ci),0.96-0.96*(nC+cw+wd-1)/(nC+cw+wd*ti),0.96-0.96*wd*ci/(nT+wd*ci)]);
  imagesc(sum(w,2));
  set(hi,'XTick',[]);
  set(hi,'YTick',1:size(w,1));
  set(hi,'YTickLabel',fv.t);
  set(hi,'YAxisLocation','right');
  if ~color
    colormap(1-gray);
  end
end

hh = axes('position',[cw*0.96/(nC+cw+wd*ti),0.04+wd*ci*0.96/(nT+wd*ci),0.96*nC/(nC+cw+wd*ti),0.96-wd*0.96*ci/(nT+wd*ci)]);


imagesc(w);

if ~color
    colormap(1-gray);
end

colorbar(h)

if ci
  set(hh,'XTick',[]);
else
  set(hh,'XTick',1:size(w,2));
  set(hh,'XTickLabel',cl);
end

if ti 
  set(hh,'YTick',[]);
else
  set(hh,'YTick',1:size(w,1));
  set(hh,'YTickLabel',fv.t);
  set(hh,'YAxisLocation','right');
end



