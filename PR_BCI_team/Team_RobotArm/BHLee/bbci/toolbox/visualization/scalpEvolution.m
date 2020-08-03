function h= scalpEvolution(erp, mnt, ival, varargin)
%h= scalpEvolution(erp, mnt, ival, <opts>)
%
% Draws scalp topographies for specified intervals,
% separately for each each class. For each classes topographies are
% plotted in one row and shared the same color map scaling. (In future
% versions there might be different options for color scaling.)
%
% IN: erp  - struct of epoched EEG data. For convenience used classwise
%            averaged data, e.g., the result of proc_average.
%     mnt  - struct defining an electrode montage
%     ival - [nIvals x 2]-sized array of interval, which are marked in the
%            ERP plot and for which scalp topographies are drawn.
%            When all interval are consequtive, ival can also be a
%            vector of interval borders.
%     opts - struct or property/value list of optional fields/properties:
%      .ival_color - [nColors x 3]-sized array of rgb-coded colors
%                    with are used to mark intervals and corresponding 
%                    scalps. Colors are cycled, i.e., there need not be
%                    as many colors as interval. Two are enough,
%                    default [0.6 1 1; 1 0.6 1].
%      .legend_pos - specifies the position of the legend in the ERP plot,
%                    default 0 (see help of legend for choices).
%      the opts struct is passed to scalpPattern
%
% OUT h - struct of handles to the created graphic objects.
%
%See also scalpEvolutionPlusChannel, scalpPatterns, scalpPlot.

%% blanker@first.fhg.de 01/2005

h= scalpEvolutionPlusChannel(erp, mnt, [], ival, varargin{:});

if nargout<1,
  clear h
end
