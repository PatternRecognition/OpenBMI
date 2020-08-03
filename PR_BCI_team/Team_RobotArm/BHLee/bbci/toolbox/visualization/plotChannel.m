function H= plotChannel(epo, clab, varargin)
%PLOTCHANNEL - wrapper function calling specialized functions to plot the 
%classwise averages of one channel. Function plotChannel2D Plots 
% 2D data (time x amplitude OR frequency x amplitude); 
% plotChannel3D plots 3D data (frequency x time x amplitude). 
%
%Usage:
% H= plotChannel(EPO, CLAB, <OPT>)
%
%Input:
% EPO  - Struct of epoched signals, see makeEpochs
% CLAB - Name (or index) of the channel to be plotted.
%
% OPT  is a struct or property/value list of optional properties, see
% plotChannel_2D for 2D plots and plotChannel_3D for 3D plots.
%
%Output:
% H - Handle to several graphical objects.
%
% See plotChannel2D and plotChannel3D for more infos on plotting.

if getDataDimension(epo)==2
  H = plotChannel2D(epo,clab,varargin{:});
else
  H = plotChannel3D(epo,clab,varargin{:});
end
