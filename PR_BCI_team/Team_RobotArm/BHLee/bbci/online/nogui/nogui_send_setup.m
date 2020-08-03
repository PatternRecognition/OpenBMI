function nogui_send_setup(setup, typ, init)
% SEND_INFORMATIONS ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% provide informations to be send via udp
%
% usage:
%    send_informations(fig,typ,init);
% 
% input:
%    setup as obtained by nogui_load_setup
%    typ   control_player1, control_player2, graphic_player1, graphic_player2 
%          or all (default)
%          defines the port (or all ports) informations should be submitted
%    init  0  send all informations
%          1  send all informations + init signal (default)
%          char:  send only this variable
%
% Guido Dornhege
% $Id: send_informations.m,v 1.16 2007/05/22 12:59:48 neuro_cvs Exp $

global general_port_fields

if nargin<3,
  init= 1;
end
if nargin<2,
  typ= 'all';
end

str1 = '';str2 = '';
str3 = '';str4='';
str5 = '';

receive_list = [];
status = [];

if islogical(init) | isnumeric(init)
  switch typ
   case 'all'
     if setup.general.graphic_master
       str1 = ['bbci.fb_machine = ''' setup.graphic_master.machine '''; bbci.fb_port = ' num2str(setup.graphic_player1.fb_port) ';bbci.control_port = ' num2str(setup.control_player1.port) ';'];
     else
       str1 = ['bbci.fb_machine = ''' setup.graphic_player1.machine '''; bbci.fb_port = ' num2str(setup.graphic_player1.fb_port) ';bbci.control_port = ' num2str(setup.control_player1.port) ';'];
     end
    str1 = [str1,get_eval_string(rmfields(setup.control_player1,{'machine','port','fields','fields_help','update_port','gui_machine'}))];
    if setup.general.player>1
      if setup.general.graphic_master
        str2 = ['bbci.fb_machine = ''' setup.graphic_master.machine '''; bbci.fb_port = ' num2str(setup.graphic_player2.fb_port) ';bbci.control_port = ' num2str(setup.control_player2.port) ';'];
      else
        str2 = ['bbci.fb_machine = ''' setup.graphic_player2.machine '''; bbci.fb_port = ' num2str(setup.graphic_player2.fb_port) ';bbci.control_port = ' num2str(setup.control_player2.port) ';'];
      end
      str2= [str2,get_eval_string(rmfield(setup.control_player2,{'machine','port','fields','fields_help','update_port','gui_machine'}))];
    end
    if setup.general.graphic_master
      if init,receive_list = 255;end
      str5 = ['feedback_opt.graphic_master = true;',get_eval_string(rmfield(setup.graphic_master,{'machine','port','fields','fields_help'}))];
    end
    
    if setup.general.graphic
      if setup.general.graphic_master
        str3 = ['feedback_opt.player1.fb_port = ' num2str(setup.graphic_player1.fb_port)  ';'];
        player1= rmfield(setup.graphic_player1,{'machine','port','fb_port','fields','fields_help','update_port','gui_machine'});
        if isfield(player1,'feedback_opt')
          player1 = player1.feedback_opt;
          str3 = [str3,get_eval_string(player1,'feedback_opt.player1')];
        end
        if setup.general.player>1
          player2= rmfield(setup.graphic_player2,{'machine','port','fb_port','fields','fields_help','update_port','gui_machine'});
          str4 = ['feedback_opt.player2.fb_port = ' num2str(setup.graphic_player2.fb_port) ';'];
          if isfield(player2,'feedback_opt')
            player2 = player2.feedback_opt;
            str4 = [str4,get_eval_string(player2,'feedback_opt.player2')];
          end
          
        end
      else
        if init,receive_list = [255];end
        str3 = ['feedback_opt.graphic_master = false;','feedback_opt.fb_port = ' num2str(setup.graphic_player1.fb_port)  ';'];
        str3 = [str3,get_eval_string(rmfields(setup.graphic_player1,{'machine','port','fb_port','fields','fields_help','update_port','gui_machine'}))];
        if setup.general.player>1
          if init,receive_list = [255,-255];end
          str4 = ['feedback_opt.graphic_master = false;','feedback_opt.fb_port = ' num2str(setup.graphic_player2.fb_port) ';'];
          str4= [str4,get_eval_string(rmfield(setup.graphic_player2,{'machine','port','fb_port','fields','fields_help','update_port','gui_machine'}))];
        end
      end
    end
    if init
%    if length(setup.general.setup_list1)>0
%      str1 = [str1,'setup_list = {'];
%      for i = 1:length(setup.general.setup_list1)
%        str1 = [str1, '''', setup.general.setup_list1{i} ''','];
%      end
%      str1 = [str1(1:end-1),'};'];
%      setup.general.setup_list_default1 = setup.general.setup_list1;
%      control_gui_queue(fig,'set_setup',setup);
%    end
    
%    if setup.general.player>1 & length(setup.general.setup_list2)>0
%      str2 = [str2,'setup_list = {'];
%      for i = 1:length(setup.general.setup_list2)
%        str2 = [str2, '''', setup.general.setup_list2{i} ''','];
%      end
%      str2 = [str2(1:end-1),'};'];
%      setup.general.setup_list_default2 = setup.general.setup_list2;
%      control_gui_queue(fig,'set_setup',setup);
%    end
    str1 = ['bbci.player=1;',str1,'loop=false;'];
    str2 = ['bbci.player=2;',str2,'loop=false;'];
    str3 = [str3,'loop=false;'];
    str4 = [str4,'loop=false;'];
    str5 = [str5,'loop=false;'];
    % also send the gui machine name
    str1 = ['bbci.gui_machine=''',setup.gui_machine,''';',str1];
    str2 = ['bbci.gui_machine=''',setup.gui_machine,''';',str2];
    str3 = ['feedback_opt.gui_machine=''',setup.gui_machine,''';',str3];
    str4 = ['feedback_opt.gui_machine=''',setup.gui_machine,''';',str4];
    if ~isempty(setup.control_player1.update_port)
      str1 = [str1,'bbci.update_port = ' num2str(setup.control_player1.update_port) ';'];
    end
    if ~isempty(setup.control_player2.update_port)
      str2 = [str2,'bbci.update_port = ' num2str(setup.control_player2.update_port) ';'];
    end
    if ~isempty(setup.graphic_player1.update_port)
      str3 = [str3,'feedback_opt.update_port = ' num2str(setup.graphic_player1.update_port) ';'];
    end
    if ~isempty(setup.graphic_player2.update_port)
      str4 = [str4,'feedback_opt.update_port = ' num2str(setup.graphic_player2.update_port) ';'];
    end
  end

   case 'control_player1'
     if setup.general.graphic_master
       str1 = ['bbci.fb_machine = ''' setup.graphic_master.machine '''; bbci.fb_port = ' num2str(setup.graphic_player1.fb_port) ';bbci.control_port = ' num2str(setup.control_player1.port) ';'];
     else
       str1 = ['bbci.fb_machine = ''' setup.graphic_player1.machine '''; bbci.fb_port = ' num2str(setup.graphic_player1.fb_port) ';bbci.control_port = ' num2str(setup.control_player1.port) ';'];
     end
     str1 = [str1,get_eval_string(rmfield(setup.control_player1,{'machine','port','fields','fields_help','update_port','gui_machine'}))];
    if init
        str1 =  ['bbci.player=1;',str1,'loop=false;'];
    end 
    if ~isempty(setup.control_player1.update_port)
      str1 = [str1,'bbci.update_port = ' num2str(setup.control_player1.update_port) ';'];
    end
   case 'control_player2'
     if setup.general.graphic_master
       str2 = ['bbci.fb_machine = ''' setup.graphic_master.machine '''; bbci.fb_port = ' num2str(setup.graphic_player2.fb_port) ';bbci.control_port = ' num2str(setup.control_player2.port) ';'];
     else
       str2 = ['bbci.fb_machine = ''' setup.graphic_player2.machine '''; bbci.fb_port = ' num2str(setup.graphic_player2.fb_port) ';bbci.control_port = ' num2str(setup.control_player2.port) ';'];
     end
     str2 = [str2,get_eval_string(rmfield(setup.control_player2,{'machine','port','fields','fields_help','update_port','gui_machine'}))];
    if init
        str2 =  ['bbci.player=1;',str2,'loop=false;'];
    end 
    if ~isempty(setup.control_player2.update_port)
      str2 = [str2,'bbci.update_port = ' num2str(setup.control_player2.update_port) ';'];
    end
   case 'graphic_master'
          str5 = get_eval_string(rmfield(setup.graphic_master,{'machine','port','fields','fields_help','update_port'}));
          if init
            str5 = [str5,'feedback_opt.graphic_master = true;loop=false;'];
            receive_list = [255];
          end

   case 'graphic_player1'
    str1 = [str1,'bbci.fb_port = ' num2str(setup.graphic_player1.fb_port) ';'];
    if setup.general.graphic_master
      str3 = ['feedback_opt.player1.fb_port = ' num2str(setup.graphic_player1.fb_port)  ';'];
      player1= rmfield(setup.graphic_player1,{'machine','port','fb_port','fields','fields_help','update_port','gui_machine'});
      if isfield(player1,'feedback_opt')
        player1 = player1.feedback_opt;
        str3 = [str3,get_eval_string(player1,'feedback_opt.player1')];
      end
    else
     if init,receive_list = [255];end
     str3 = ['feedback_opt.fb_port = ' num2str(setup.graphic_player1.fb_port) ';'];
      str3 = [str3,get_eval_string(rmfield(setup.graphic_player1,{'machine','port','fb_port','fields','fields_help'}))];
    end
    if init
        str3 =  [str3,'loop=false;'];
    end 
    if ~isempty(setup.graphic_player1.update_port)
      str3 = [str3,'feedback_opt.update_port = ' num2str(setup.graphic_player1.update_port) ';'];
    end
   case 'graphic_player2'
    str2 = [str2,'bbci.fb_port = ' num2str(setup.graphic_player2.fb_port) ';'];
    if setup.general.graphic_master
      str4 = ['feedback_opt.player2.fb_port = ' num2str(setup.graphic_player2.fb_port)  ';'];
      player2= rmfield(setup.graphic_player2,{'machine','port','fb_port','fields','fields_help','update_port','gui_machine'});
      if isfield(player2,'feedback_opt')
        player2 = player2.feedback_opt;
        str4 = [str4,get_eval_string(player2,'feedback_opt.player2')];
      end
    else
        if init,receive_list = [-255];end
      str4 = ['feedback_opt.fb_port = ' num2str(setup.graphic_player2.fb_port) ';'];
      str4 = [str4,get_eval_string(rmfield(setup.graphic_player2,{'machine','port','fb_port','fields','fields_help','update_port','gui_machine'}))];
    end
    if init
        str4 =  [str4,'loop=false;'];
    end 
    if ~isempty(setup.graphic_player2.update_port)
      str4 = [str4,'feedback_opt.update_port = ' num2str(setup.graphic_player2.update_port) ';'];
    end
  end
else
  % only use this string
  str1 = '';str2 = '';str3 = ''; str4 = ''; str5 = '';
  switch typ
   case 'control_player1'
    if any(init=='=')
      str1 = init;
    else
      str1 = sprintf('%s = %s;',init,get_text_string(eval(sprintf('setup.%s.%s',typ,init))));
    end
   case 'control_player2'
    if any(init=='=')
      str2 = init;
    else
      str2 = sprintf('%s = %s;',init,get_text_string(eval(sprintf('setup.%s.%s',typ,init))));
    end
   case 'graphic_player1'
    if any(init=='=')
      str3 = init;
    else
      if setup.general.graphic_master
        initt = init;
        if length(initt)>=length('feedback_opt.') & strcmp(init(1:length('feedback_opt.')),'feedback_opt.')
          initt = [initt(1:length('feedback_opt.')), 'player1',initt(length('feedback_opt.'):end)];
        end
        str3 = sprintf('%s = %s;',initt,get_text_string(eval(sprintf('setup.%s.%s',typ,init))));
      else 
        str3 = sprintf('%s = %s;',init,get_text_string(eval(sprintf('setup.%s.%s',typ,init))));
      end
    end
   case 'graphic_player2'
    if any(init=='=')
      str4 = init;
    else
      if setup.general.graphic_master
        initt = init;
        initt(1:length('feedback_opt.'))
        if length(initt)>=length('feedback_opt.') & strcmp(initt(1:length('feedback_opt.')),'feedback_opt.')
          initt = [initt(1:length('feedback_opt.')), 'player2',initt(length('feedback_opt.'):end)];
        end
        str3 = sprintf('%s = %s;',initt,get_text_string(eval(sprintf('setup.%s.%s',typ,init))));
      else 
        str4 = sprintf('%s = %s;',init,get_text_string(eval(sprintf('setup.%s.%s',typ,init))));
      end
    end
   case 'graphic_master'
    if any(init=='=')
      str5 = init;
    else
      str5 = sprintf('%s = %s;',init,get_text_string(eval(sprintf('setup.%s.%s',typ,init))));
    end
  end
  init = false;
end

if init
  nowi = now;
  str1 = sprintf('%stimecheck=%10.10f;\n',str1,nowi);
  str2 = sprintf('%stimecheck=%10.10f;\n',str2,nowi);
  str3 = sprintf('%stimecheck=%10.10f;\n',str3,nowi);
  str4 = sprintf('%stimecheck=%10.10f;\n',str4,nowi);
  str5 = sprintf('%stimecheck=%10.10f;\n',str5,nowi);
end
  
% test if string makes sense
flag = test_eval_string(str1);
flag = flag&test_eval_string(str2);
flag = flag&test_eval_string(str3);
flag = flag&test_eval_string(str4);
flag = flag&test_eval_string(str5);

% error handling if it does not make sense
if ~flag
  error('some part of the setup not valid')
end

 
% send the string by send_evalstring
switch typ
 case 'all'
  if setup.general.active1
    send_evalstring(setup.control_player1.machine,setup.control_player1.port,str1);
  end
  if setup.general.player>1 & setup.general.active2
    send_evalstring(setup.control_player2.machine,setup.control_player2.port,str2);
  end
  
  if setup.general.graphic_master
    if setup.general.graphic
      if setup.general.player>1
        send_evalstring(setup.graphic_master.machine,setup.graphic_master.port,[str3,str4,str5]);
      else
        send_evalstring(setup.graphic_master.machine,setup.graphic_master.port,[str3,str5]);
      end
    else
      send_evalstring(setup.graphic_master.machine,setup.graphic_master.port,[str5]);
    end
  else
      
    if setup.general.graphic
      if setup.general.player>1 
        if strcmp(setup.graphic_player1.machine,setup.graphic_player2.machine) & setup.graphic_player1.port == setup.graphic_player2.port & setup.general.active1 & setup.general.active2
          str3 = [str3,str4];
        else
          if setup.general.active2
            if isempty(setup.graphic_player2.machine)
              send_evalstring(setup.control_player2.machine,setup.control_player2.port,str4);
            else
              send_evalstring(setup.graphic_player2.machine,setup.graphic_player2.port,str4);
            end
          end
        end
      end
      if setup.general.active1
        if isempty(setup.graphic_player1.machine)
          send_evalstring(setup.control_player1.machine,setup.control_player1.port,str3);
        else
          send_evalstring(setup.graphic_player1.machine,setup.graphic_player1.port,str3);
        end
      end
    end
  end
 case 'control_player1'
  if setup.general.active1
    send_evalstring(setup.control_player1.machine,setup.control_player1.port,str1);
  end
 case 'control_player2'
    if setup.general.active2
      send_evalstring(setup.control_player2.machine,setup.control_player2.port,str2);
    end
 case 'graphic_master'
  if setup.general.graphic_master,
    send_evalstring(setup.graphic_master.machine, ...
                    setup.graphic_master.port, str5);
  end
    
 case 'graphic_player1'
  if ~isempty(str1) & setup.general.active1
    send_evalstring(setup.control_player1.machine,setup.control_player1.port,str1);
  end       
  if ~setup.general.graphic_master 
    if setup.general.active1
      if isempty(setup.graphic_player1.machine)
        send_evalstring(setup.control_player1.machine,setup.control_player1.port,str3);
      else
        send_evalstring(setup.graphic_player1.machine,setup.graphic_player1.port,str3);
      end
    end
  else
    send_evalstring(setup.graphic_master.machine,setup.graphic_master.port,str3);
  end 
 case 'graphic_player2'
  if ~isempty(str2) & setup.general.active2
    send_evalstring(setup.control_player2.machine,setup.control_player2.port,str2);
  end       
  if ~setup.general.graphic_master
    if setup.general.active2
      if isempty(setup.graphic_player2.machine)
        send_evalstring(setup.control_player2.machine,setup.control_player2.port,str4);
      else
        send_evalstring(setup.graphic_player2.machine,setup.graphic_player2.port,str4);
      end
    end
  else
    send_evalstring(setup.graphic_master.machine,setup.graphic_master.port,str4);
  end  
end

if setup.general.player>1 & ~setup.general.active2 
  receive_list = setdiff(receive_list,-255);
end

if ~setup.general.active1
  receive_list = setdiff(receive_list,255);
end

return;


function does_it_eval = test_eval_string(string);
% THIS FUNCTION EVALUATES WHETHER A STRING MAKES SENSE
does_it_eval = false;

try
  eval([string ';']);
  does_it_eval = true;
end
