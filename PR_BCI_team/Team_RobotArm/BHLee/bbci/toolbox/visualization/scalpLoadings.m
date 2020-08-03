function h= scalpLoadings(erp, mnt, ival, varargin)
%SCALPLOADINGS - Display classwise topographies
%
%Usage:
% H= scalpLoadings(ERP, MNT, IVAL, <OPTS>)
%
%Input:
% ERP  - struct of epoched EEG data. For convenience used classwise
%        averaged data, e.g., the result of proc_average.
% MNT  - struct defining an electrode montage
% IVAL - The time interval for which scalp topographies are to be plotted.
%        May be either one interval for all classes, or specific
%        intervals for each class. In the latter case the k-th row of IVAL
%        defines the interval for the k-th class.
% OPTS - struct or property/value list of optional fields/properties:
%  The opts struct is passed to scalpPattern.
%
%Output:
% H: Handle to several graphical objects.
%
%See also scalpEvolution, scalpPatternsPlusChannel, scalpPlot.

% Author(s): Benjamin Blankertz, Jan 2005

h= scalpLoadingsPlusChannel(erp, mnt, [], ival, varargin{:});

if nargout<1,
  clear h
end
