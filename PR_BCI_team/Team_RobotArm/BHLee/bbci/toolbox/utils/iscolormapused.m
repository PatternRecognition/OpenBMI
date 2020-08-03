function yesorno= iscolormapused(hf)
%ISCOLORMAPUSED - Determines whether the figure's colormap is in use
%
%Description:
%  This function determines whether there is some child of the specified
%  figure that uses the figure's colormap. Since all subplots of one
%  figure all share a common colormap (figure property), this information
%  is important when one wants to add a new image with a local colormap
%  without destroying other subplots on the figure. In this case the
%  function image_local_cmap can be used.
%
%Usage:
%YESORNO= iscolormapused(<HF>);
%
%Input:
%  HF: Handle of the figure. If none is specified the current figure
%      is inspected, i.e., default gcf.
%
%Example:
%  clf; subplot(1,3,1); plot(randn(100,2));
%  iscolormapused
%  subplot(1,3,2);
%  patch([0 1 1 0],[0 0 1 1], [1 0.1 0.7]);  %% specify color directly
%  iscolormapused
%  subplot(1,3,3); 
%  patch([0 1 1 0],[0 0 1 1], 2);  %% specify color by indexing into colormap
%  iscolormapused
%
%See also image_local_cmap.

if nargin==0,
  hf= gcf;
end

if isfield(get(hf),'CData') & ~isempty(get(hf,'CData')),
  yesorno= 1;
  return;
end

hc= get(hf, 'children');
yesorno= 0;

ii= 0;
while ~yesorno & ii<length(hc),
  ii= ii+1;
  yesorno= iscolormapused(hc(ii));
end
