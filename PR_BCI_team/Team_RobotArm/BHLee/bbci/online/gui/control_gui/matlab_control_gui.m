function matlab_control_gui(setup_file, varargin)
% MATLAB_CONTROL_GUI starts a matlab control gui for bbci_bet
% 
% usage: 
%       matlab_control_gui(setup_file);
%
% input:
%    setup_file    a name of a setup_file for the gui, can be empty
%
% Uses many functions in the same directory
%
% Guido Dornhege, Matthias Krauledat
% $Id: matlab_control_gui.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

global nice_gui_font general_port_fields

if ~exist('setup_file','var'),
  setup_file= [];
end

opt = propertylist2struct(varargin{:});
player = length(general_port_fields);
               
if isempty(nice_gui_font)
  nice_gui_font = 'Helvetica';
end

%CONSTRUCT A FIGURE
fig = figure;
message_box(true);

%PLOT THE GENERAL FIGURE
plot_matlab_control_gui(fig);

%ACTIVATE THE GENERAL SETUP
activate_control_gui(fig,'general');

% LOAD THE SETUP FILE if given as argument
if ~isempty(setup_file),
  load_control_setup(fig,'default','all','setup_file',setup_file);
end

% LOAD THE classifier if given as argument
if isfield(opt,'classifier')
  if isequal(opt.classifier, 'auto') || isnumeric(opt.classifier)
    im_lucky(fig, 'classifier',opt.classifier);
  elseif ischar(opt.classifier)
    if ~isequal(opt.classifier(end-3:end),'.mat')
      disp('append ''.mat'' to classifier name');
      opt.classifier = [opt.classifier '.mat'];
    end
    add_setup(fig,player, 'classifier', opt.classifier);
  else
    error('The classifier option should be a string with the name of classifier, either with absoluth path or without. In alternative the option ''auto'' can be used or just the number of the classifier to select');
  end
else
  if ~isempty(setup_file),
    im_lucky(fig);
  end
end

 
