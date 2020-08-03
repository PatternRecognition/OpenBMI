function acm= fig_addColormap(cmap, varargin)
%FIG_ADDCOLORMAP - Add a colormap to the current figure
%
% !!! TESTING !!!
%
%Description:
% This function adds a colormap to the current figure. If applied
% correctly, the function allows to used multiple colormaps within
% a single figure.
%
%Synopsis:
% CLIMFACTOR= fig_addColormap(CMAP)
%
%Arguments:
% CMAP - [N 3] double array defining the colormap.
%
%Returns:
% ACM - Struct to be used for adapting the CLim and colorbar of new axes:
%    .cLimFactor - Factor for rescaling the CLim of a new axis that should
%       use the new colormap.
%    .nColors - Number of colors in the new colormap.
%
%Example:
% % make a colorful plot using colormap 'hot'
% clf; subplot(1, 2, 1);
% colormap(hot(11));
% imagesc([1:100]); colorbar;
%
% % add a new colormap 'copper' to the figure
% acm= fig_addColormap(copper(21));
% % do your rendering as usual
% subplot(1, 2, 2);
% imagesc(100+[1:100]'); colorbar;
% % and correct the CLim and colorbar
% fig_acmAdaptCLim(acm);
% 
%See: cmap_* for some nice colormaps.

%Known Bug(s):
%In some special cases, fig_addColormap changes the mapping a little bit
%clf;
%colormap default;
%ax1= subplot(1,2,1);
%colormap(cmap_rainbow(5));
%imagesc(1:6); colorbar
%
%%Look before the next command
%acm= fig_addColormap(cool(7));
%%and compare with this.

% Author(s): Benjamin Blankertz, Oct 2006


hf= gcf;
hc= findobj(hf, 'Type','axes', 'Tag','');
cm= get(hf, 'Colormap');
cml1= size(cm, 1);
cml2= size(cmap, 1);

set(hf, 'Colormap',[cm; cmap]);
for ii= 1:length(hc),
  hax= hc(ii);
  cLim= get(hax, 'CLim');
  ud= get(hax, 'UserData');
  if isempty(ud) | ~isfield(ud, 'origCLim'),
    ud.origCLim= cLim;
    ud.cmap_ival= [1 cml1];
%    ud.cmap_ival= [cml1+1 cml1+cml2];
    set(hax, 'UserData',ud);
  end
  epsilon= 0.000001;
  cf= diff(ud.origCLim)/(diff(ud.cmap_ival)+1-epsilon);
  newCLim= [ud.origCLim(1) - cf*(ud.cmap_ival(1)-1) ...
            ud.origCLim(2) + cf*(cml1+cml2-ud.cmap_ival(2)+epsilon)];
  hcb= axis_getColorbarHandle(hax);
  if ~isempty(hcb),
    cbXLim= get(hcb, 'XLim');
    cbYLim= get(hcb, 'YLim');
%    cbIm= get(hcb, 'Children');
%    cbData= get(cbIm, 'CData');
  end
  set(hax, 'CLim',newCLim);
  if ~isempty(hcb),
    if numel(cbXLim)==1
        set(hcb, 'XLim', cbXLim);
        set(hcb, 'YLim', cbYLim);
    else
        for kk=1:numel(cbXLim)
            set(hcb(kk), 'XLim', cbXLim{kk});
            set(hcb(kk), 'YLim', cbYLim{kk});        
        end
    end
  end
end

acm.cLimFactor= (cml1+cml2)/cml2;
acm.nColors= size(cmap, 1);
