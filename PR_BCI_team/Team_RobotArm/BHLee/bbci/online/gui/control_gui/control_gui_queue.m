function out = control_gui_queue(fig,task,varargin);
% CONTROL_GUI_QUEUE ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% administrate all important variables of the gui
%
% input:
%    fig    the handle of the gui
%    task   set_setup:  set a setup by varargin{1}
%           get_setup:  get a setup into out
%           remove:     remove the setup varargin{1}
%           set_general: set general handle by varargin{1}
%           set_control_player1: set control_player1 handle by varargin{1}
%           set_control_player2: set control_player2 handle by varargin{1}
%           set_graphic_player1: set graphic_player1 handle by varargin{1}
%           set_graphic_player2: set graphic_player2 handle by varargin{1}
%           set_general_ax: set general_ax handle by varargin{1}
%           set_control_player1_ax: set control_player1_ax handle by varargin{1}
%           set_control_player2_ax: set control_player2_ax handle by varargin{1}
%           set_graphic_player1_ax: set graphic_player1_ax handle by varargin{1}
%           set_graphic_player2_ax: set graphic_player2_ax handle by varargin{1}
%           get_general: get general handle into out
%           get_control_player1: get control_player1 handle into out
%           get_control_player2: get control_player2 handle into out
%           get_graphic_player1: get graphic_player1 handle into out
%           get_graphic_player2: get graphic_player2 handle into out
%           get_general_ax: get general_ax handle into out
%           get_control_player1_ax: get control_player1_ax handle into out
%           get_control_player2_ax: get control_player2_ax handle into out
%           get_graphic_player1_ax: get graphic_player1_ax handle into out
%           get_graphic_player2_ax: get graphic_player2_ax handle into out
%     varargin further infos for all set routines and remove
%
% output:
%     out   results of the get_routines
% 
% if control_gui_queue is called without arguments everything is deleted
%
% Guido Dornhege
% $Id: control_gui_queue.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $


% some persistent variables
persistent figures setups general control_player1 control_player2 graphic_player1 graphic_player2 general_ax control_player1_ax control_player2_ax graphic_player1_ax graphic_player2_ax graphic_master graphic_master_ax

% reset
if ~exist('fig','var')
  figures = [];
  setups = [];
  general = [];
  control_player1 = [];
  control_player2 = [];
  graphic_master = [];
  graphic_player1 = [];
  graphic_player2 = [];
  general_ax = [];
  control_player1_ax = [];
  control_player2_ax= [];
  graphic_master_ax = [];
  graphic_player1_ax = [];
  graphic_player2_ax = [];
  return;
end


% find the figure and do the task
ind = find(figures==fig);

out = [];

switch task
 case 'set_setup'
  if isempty(ind)
    figures = [figures,fig];
    if isempty(setups)
      setups = {};
    end
    setups = {setups{:},varargin{1}};
    general = [general,0];
    control_player1 = [control_player1,0];
    control_player2 = [control_player2,0];
    graphic_player1 = [graphic_player1,0];
    graphic_master = [graphic_master,0];
    graphic_player2 = [graphic_player2,0];
    if isempty(general_ax), general_ax = {};end
    if isempty(control_player1_ax), control_player1_ax = {};end
    if isempty(control_player2_ax), control_player2_ax = {};end
    if isempty(graphic_master_ax), graphic_master_ax = {};end
    if isempty(graphic_player1_ax), graphic_player1_ax = {};end
    if isempty(graphic_player2_ax), graphic_player2_ax = {};end
    general_ax = {general_ax{:},struct};
    control_player1_ax = {control_player1_ax{:},struct};
    control_player2_ax = {control_player2_ax{:},struct};
    graphic_master_ax = {graphic_master_ax{:},struct};
    graphic_player1_ax = {graphic_player1_ax{:},struct};
    graphic_player2_ax = {graphic_player2_ax{:},struct};
  else        
    setups{ind} = varargin{1};
  end
  
 case 'get_setup'
  if ~isempty(ind)
    out = setups{ind};
  end
  
 case 'remove'
  figures(ind) = [];
  setups(ind) = [];
  general(ind) = [];
  control_player1(ind) = [];
  control_player2(ind) = [];
  graphic_master(ind) = [];
  graphic_player1(ind) = [];
  graphic_player2(ind) = [];
  general_ax(ind) = [];
  control_player1_ax(ind) = [];
  control_player2_ax(ind) = [];
  graphic_master_ax(ind) = [];
  graphic_player1_ax(ind) = [];
  graphic_player2_ax(ind) = [];
 
 case 'set_general'
  general(ind) = varargin{1};
  
 case 'set_control_player1'
  control_player1(ind) = varargin{1};

 case 'set_control_player2'
  control_player2(ind) = varargin{1};

 case 'set_graphic_player1'
  graphic_player1(ind) = varargin{1};
 
 case 'set_graphic_master'
  graphic_master(ind) = varargin{1};

 case 'set_graphic_player2'
  graphic_player2(ind) = varargin{1};

 case 'get_general'
  out = general(ind);

 case 'get_control_player1'
  out = control_player1(ind);
 
 case 'get_control_player2'
  out = control_player2(ind);

 case 'get_graphic_master'
  out = graphic_master(ind);
 
 case 'get_graphic_player1'
  out = graphic_player1(ind);
 
 case 'get_graphic_player2'
  out = graphic_player2(ind);

 case 'set_general_ax'
  remove_ax(general_ax{ind});
  general_ax{ind} = varargin{1};
  
 case 'set_control_player1_ax'
  remove_ax(control_player1_ax{ind});
  control_player1_ax{ind} = varargin{1};

 case 'set_control_player2_ax'
  remove_ax(control_player2_ax{ind});
  control_player2_ax{ind} = varargin{1};

 case 'set_graphic_player1_ax'
  remove_ax(graphic_player1_ax{ind});
  graphic_player1_ax{ind} = varargin{1};

 case 'set_graphic_master_ax'
  remove_ax(graphic_master_ax{ind});
  graphic_master_ax{ind} = varargin{1};

 case 'set_graphic_player2_ax'
  remove_ax(graphic_player2_ax{ind});
  graphic_player2_ax{ind} = varargin{1};

 case 'get_general_ax'
  out = general_ax{ind};

 case 'get_control_player1_ax'
  out = control_player1_ax{ind};
 
 case 'get_control_player2_ax'
  out = control_player2_ax{ind};

 case 'get_graphic_master_ax'
  out = graphic_master_ax{ind};
 
 case 'get_graphic_player1_ax'
  out = graphic_player1_ax{ind};
 
 case 'get_graphic_player2_ax'
  out = graphic_player2_ax{ind};
 otherwise
  error('');
  
  
end

  
  
  