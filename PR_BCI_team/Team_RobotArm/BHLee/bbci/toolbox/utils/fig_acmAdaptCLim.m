function fig_acmAdaptCLim(acm, h_ax, varargin)
%FIG_ACMADAPTCLIM - Helper function to be used with fig_adaptColormap
%
%See help of fig_adaptColormap

if nargin<2 | isempty(h_ax),
  h_ax= gca;
elseif length(h_ax)>1,
  fig_acmAdaptCLim(acm, h_ax(1), varargin{:});
  for ii= 2:length(h_ax),
    fig_acmAdaptCLim(acm, h_ax(ii));
  end
  return;
end

cLim= get(h_ax, 'CLim');
ud= get(h_ax, 'UserData');
if isempty(ud) | ~isfield(ud, 'origCLim'),
  ud.origCLim= cLim;
  ud.cmap_ival= size(colormap,1) + [-acm.nColors+1 0];
  set(h_ax, 'UserData',ud);
end

newCLim= cLim(2) - [diff(cLim)*acm.cLimFactor 0];
set(h_ax, 'CLim', newCLim + [0 -0.0001]*diff(newCLim));

if nargin<3,
  h_cb= axis_getColorbarHandle(h_ax);
else
  h_cb= varargin{1};
end
if ~isnan(h_cb) & ~isempty(h_cb) & h_cb~=0,
  cLimAdapt= cLim+[0.00001*diff(cLim) 0];
  h_im= get(h_cb, 'Children');
  if length(h_im)>1,
    h_im= h_im(end);
    warning('hack');
  end
  sz= size(get(h_im, 'CData'));
  vv= linspace(cLimAdapt(1), cLimAdapt(2), acm.nColors+1);
  dd= [vv(1) vv(end-1)] + diff(cLimAdapt)/acm.nColors/2;
  cml= size(get(get(h_ax, 'Parent'), 'Colormap'), 1);
  cd= [cml-acm.nColors+1:cml];
  if sz(1)==1,
    %% orientation= 'horiz';
    set(h_cb, 'XLim',cLimAdapt);
    set(h_im, 'CData',cd, 'XData', dd);
  else
    %% orientation= 'vert';
    set(h_cb, 'YLim',cLimAdapt);
    set(h_im, 'CData',cd', 'YData', dd);
  end
end
