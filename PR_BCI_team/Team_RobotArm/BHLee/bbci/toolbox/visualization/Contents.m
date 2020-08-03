% eegVisualize - methods for visualizing EEG data
%
% plot signals in time (or frequency) domain and utilities
%   showERP             - show class averages of epoched data for one channels
%   grid_plot           - do the same for multi channels in grid layout
%   grid_getSubplots    - get axes handles for channel subplots
%   grid_markTimePoint  - mark a specific time point in grid plot
%   grid_markIval       - mark a specific time interval in grid plot
%   grid_markRange      - mark a range on the y axis in grid plot
%   showEvent           - show one single epoch in continuous view
%
% generate or manipulate electrode montages
%   setElectrodeMontage - generate an electrode montage for given channels
%   setDisplayMontage   - configure the channel layout for grid_plot
%   restrictMontage     - restrict a montage to a subset of channels
%   excenterNonEEGchans - excenter some channels from the grid layout
%
% scalp topographies
%   plotScalpPattern    - display a vector as scalp map
%   showERPscalps       - display average temporal evolution of scalp 
%                         potentials
%   showLinClassyPatterns - works only for certain features and classifiers
%                         display trained classifier as series of scalp maps
%
% other utilities
%   plotOverlappingHist - plots joint histograms for two classes from
%                         one channel of epoched data
%   saveFigure          - saving figures in eps format
%   saveFigure_cbw      - saving figures in color and black'n'white versions
%   addTransferRates    - add an axis with information transfer rates
%                         on the right side of a classification error curve
%   moveObjectBack      - put a graphical object in the background
%
%
% the following global variables are used.
%   EEG_FIG_DIR - for saving figures
%   EEG_CFG_DIR - for loading grid layouts for display montages
