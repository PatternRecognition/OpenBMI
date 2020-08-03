function bbci_bet_apply_offline(cnt, mrk, varargin);

global general_port_fields parport_on BBCI_DIR

if isempty(parport_on), parport_on = 1;end

path(path, [BBCI_DIR 'simulation/feedback_listener']);

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'cycle', 0, ...
                  'realtime', 0, ...
                  'setup_list', 1, ...
                  'cmb_setup', 'concat', ...
                  'modifications', {});
  
setup_list= opt.setup_list;
cmb_setup= opt.cmb_setup;
modifications= opt.modifications;

new_setup = '';

for i = 1:2:length(modifications)
  new_setup = sprintf('%s;%s = modifications{%d};',new_setup,modifications{i},i+1);
end

modifications = {};

parPort_list = [];

% we are optimistic.  
run = 1; % the status of the loop (run=1 -> the loop runs, run = 0->
           % the loop should stop)
errorMessage = [];


if isempty(setup_list)
  if ~isempty(general_port_fields) & isfield(general_port_fields(1),'control')
    setup_list = general_port_fields(1).control{2};
  else
    setup_list = 12470;
  end
end

BBCI_BET_APPLY = true;

if isnumeric(setup_list)
  
  if setup_list<=2
    if isnumeric(setup_list)
      parPort_list = 229+setup_list;
    end
    if ~isempty(general_port_fields) & isfield(general_port_fields(1),'control')
      setup_list = general_port_fields(setup_list).control{2};
    else
      setup_list = 12471-setup_list;
    end
  end
  
  %open the listening port for interaction
  control_communication = get_data_udp(setup_list);
  
  % wait for a setup if not given
  while run & isnumeric(setup_list)
    pause(0.1);
    get_setup;
  end
  get_data_udp(control_communication);
end


% set some defaults

%%%%% DO IT IN A LOOP UNTIL EXIT
while run
  
  %%% GET THE SETUPS
  if ~iscell(setup_list)
    setup_list = {setup_list};
  end
  load(setup_list{1},'bbci')
  
%  bbci.control_port = 12471-bbci.player;
  bbci = set_defaults(bbci,'player',1);
  
  if ~isfield(bbci,'control_port')
      if ~isempty(general_port_fields) & isfield(general_port_fields(1),'control')
          bbci.control_port = general_port_fields(min(bbci.player,length(general_port_fields))).control{2};
      else
          bbci.control_port = 12471-bbci.player;
      end
  end
  [cont_proc,feature,cls,post_proc,marker_output] = feval(['cmb_setup_' cmb_setup],bbci,setup_list{:});
  bbci.setup_list = setup_list;
  fprintf(escape_printf(new_setup));
  eval(new_setup);
    
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% set all defaults and check the input%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % All defaults were already set by bbci_bet_prepare
  
  % Possible to specify a single function instead of cell array. Can't check
  % this directly in the params, convert to cell if function is not a cell.
  for pr = 1:length(cont_proc),
    %sometimes processing is not needed
    if ~isfield(cont_proc(pr),'procFunc')
      cont_proc(pr).procFunc = {};
      cont_proc(pr).procParam = {{}};    
    end
    %it is possible to specify a single string as processing function 
    %without parameters
    if ~iscell(cont_proc(pr).procFunc),
      cont_proc(pr).procFunc = {cont_proc(pr).procFunc};
      cont_proc(pr).procParam = {cont_proc(pr).procParam};
    end
  end
  
  % make sure that ringbuffer size is not too small: Extract the largest
  % relevant window & add one second buffer
  maxIlenApply = max([feature.ilen_apply])+1000;
  if bbci.ringBufferSize<maxIlenApply,
    error('Ring buffer too small');
  end
  
  % Set defaults for post_proc. Avoid strcmp in apply loop, make sure the
  % .proc field exists:
  if ~exist('post_proc','var')
    post_proc = [];
  end
  if isempty(post_proc) | ~isfield(post_proc, 'proc'),
    post_proc.proc = [];
  end
  if ~isfield(post_proc, 'proc_param'),
    post_proc.proc_param = {};
  end
  
  % Check clab for wildcards
  flag = false;
  for pr = 1:length(cont_proc)
    flag = flag | checkWildcards(cont_proc(pr).clab);
  end
  if flag
    run = 0;
    errorMessage = 'No wildcards are allowed in clabs';
  end
  
  if ~exist('marker_output','var')
    marker_output = [];
  end
  
  bbci = set_defaults(bbci,'fb_machine',general_port_fields(1).bvmachine,'fb_port',general_port_fields(min(bbci.player,length(general_port_fields))).control{3},'other_clients',{},'other_ports',[]);
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% initialize all important communication streams %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  %%%% map for player to the cont\_proc.clab
  for i = 1:length(cont_proc)
    for j = 1:length(cont_proc(i).clab)
      if bbci.player==2 & cont_proc(i).clab{j}(1)~='x'
        cont_proc(i).clab{j} = ['x' cont_proc(i).clab{j}];
      end
      if bbci.player==1 & cont_proc(i).clab{j}(1)=='x'
        cont_proc(i).clab{j} = cont_proc(i).clab{j}(2:end);
      end
    end
  end
  
  if isempty(bbci.fb_machine)
    if ~isempty(bbci.other_clients)
      warning('other clients not supported in this mode');
    end
    if ~exist('feedback_opt','var')
      feedback_opt = {};
    end
    feedback_opt.reset = 1;
    feedback_opt= set_defaults(feedback_opt, ...
                               'parPort', 1, ...
                               'type', '', ...
                               'log', 1);
    if feedback_opt.parPort==1
      do_set(255);
    end
    %% unique for offline application
    if ~isempty(feedback_opt.type),
      feedback_opt.type= strrep(feedback_opt.type, 'feedback_', 'fbl_');
    end
   
    if isfield(feedback_opt,'graphic_master') & feedback_opt.graphic_master
      error('bbci_bet_apply is not allowed to be graphic_master');
    end
    loop_fb = true;
    run_fb = true;
    fig = figure;
  else
    
    
    % UDP for sending data
    if isempty(bbci.other_clients)
      send_data_udp(bbci.fb_machine,bbci.fb_port);
    else
      send_data_udp(cat(2,{bbci.fb_machine},bbci.other_clients),[bbci.fb_port,bbci.other_ports]);
    end
  end
  
  old_port = bbci.fb_port;
  old_machine = bbci.fb_machine;
  
   
  % UDP FOR REQUIRING INTERACTION
  control_communication = get_data_udp(bbci.control_port);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% initialize important run time variables %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  %form: name = value; % documentation
  
  % Each processing has its own state variable(s). Initially, these are empty
  procState = cell([1 length(cont_proc)]);
  for i = 1:length(cont_proc)
    procState{i} = cell([1, length(cont_proc(i).procFunc)]);
  end
  % check the number of outputs for each function. If it is greater than 1,
  % it is assumed that a state variable is used. If you don't like this
  % behavior, you have to supply a field 'use_state_var'.
  for ii = 1:length(cont_proc)
    if ~isfield(cont_proc(ii),'use_state_var')
      use_state_var = [];
      for jj = 1:length(cont_proc(ii).procFunc)
        use_state_var(jj) = (nargout(cont_proc(ii).procFunc{jj})>1);
      end
      cont_proc(ii).use_state_var = use_state_var;
    end
  end
  
  
  % Init the storage module for the processed data
  storeContData('init', length(cont_proc), length(cnt.clab), ...
                bbci, 'sloppyfirstappend',1);
  
  
  % For each processing, compute the actual channel indices from the string
  % labels
  for pr = 1:length(cont_proc)
    cont_proc(pr).chans = chanind(cnt.clab, cont_proc(pr).clab);
    if length(cont_proc(pr).clab) ~= length(cont_proc(pr).chans)
      run = 0;
      errorMessage = 'clabs mismatch';
    end
  end
  
  % Initialize the classifier!!!
  cls = getClassifier('init',cls,bbci);
  
  getFeature('init',feature,bbci,cont_proc);
  
  % Initialize the marker queue
  adminMarker('init',bbci);
  
  % Initialize the standardPostProc
  standardPostProc('init',cls,bbci);
  
  
  % Open the log file.
  important_log_file_vars = struct('cont_proc',{cont_proc},'cls',{cls},'feature',{feature},'post_proc',{post_proc});
  logFileNumber = writeClassifierLog('init',bbci,important_log_file_vars);
  
  % init marker Output
  marker_output = performMarkerOutput('init',bbci,marker_output);
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
  %%%%% other important initializations %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  [cls,bbci]=adaptation(cls,bbci,'init');
  
  if isfield(bbci,'initialize_functions')
    for bb = 1:length(bbci.initialize_functions)
      feval(bbci.initialize_functions{bb},bbci.initialize_params{bb}{:});
    end
  end					
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% THE MAIN WORKING LOOP %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  loop = true;
  controlnumber = 0;

  clabs = cnt.clab;
  for i = 1:length(clabs)
    if clabs{i}(1)=='x'
      clabs{i} = clabs{i}(2:end);
    end
  end

  fprintf('\nUsing the following classifier:\n%s\n\n', toString(cls));
  blocksize= 40;  %% or bbci.minDataLength
  cnt_step= blocksize/1000*cnt.fs;
  cnt_idx= [1 cnt_step];
  block= 0;
  waitForSync;
  while loop
    %%% get the EEG %%%
    block= block + cnt_step;
    timestamp = block*1000/bbci.fs;
    data= cnt.x([cnt_idx(1):cnt_idx(2)], :);
    
    %%% save the marker in the marker queue %%%
    mrk_idx= find(mrk.pos>=cnt_idx(1) & mrk.pos<=cnt_idx(end));
    markerPos= ( mrk.pos(mrk_idx)-cnt_idx(end)+1 ) * 1000/cnt.fs;
    markerToken= mrk.desc(mrk_idx);
    markerDescr= mrk.type(mrk_idx);
    adminMarker('add',timestamp,markerPos,markerToken,markerDescr);

    cnt_idx= cnt_idx + cnt_step;
    if cnt_idx(end) > size(cnt.x,1),
      if opt.cycle,
        cnt_idx= [1 cnt_step];
      else
        loop= false;
        run= false;
      end
    end

    % Make struct from data because most processings require structs.
    % for player 2 kill initial x in channel names
    
    data = struct('fs',bbci.fs,'x',data,'clab',{clabs});
    
    %%% apply the cnt processing %%%
    for pr = 1:length(cont_proc)
      theProc = cont_proc(pr);

      % Do this only on the selected subset of channels
      procData = proc_selectChannels(data, theProc.chans);
      
      % Each processing step can have a list of functions to call
      % sequentially. 
      for i = 1:length(theProc.procFunc),
        if ~theProc.use_state_var(i)
          procData = feval(theProc.procFunc{i}, procData, theProc.procParam{i}{:});
        else
          % Each processing step has its own set of state variables
          [procData,procState{pr}{i}] = ...
              feval(theProc.procFunc{i}, procData, procState{pr}{i}, theProc.procParam{i}{:});
        end
      end
      % Store via container function. This also handles the case that a huge block
      % is written (larger than ringBufferSize)
      storeContData('append', pr, procData.x);
    end % end cnt processing

    %%% calculate the classifiers %%%
    [out,out_aa] = getClassifier('apply',size(data.x,1)*1000/bbci.fs,cls);
    
    %%% if necessary perform mrk_out %%%
    mrkOut = performMarkerOutput('apply',marker_output,size(data.x,1)*1000/bbci.fs,timestamp);
    
%%% log out %%%
    writeClassifierLog('cls',timestamp,cat(2,out_aa,mrkOut));

    %%% Adapt the classifier, if necessary %%%
    [cls,bbci] = adaptation(cls,bbci,timestamp); %commented from claudia
    
    %%% individual post_process the data. If no function is given as
    % post_proc.proc, then do what we always do. Otherwise, call a
    % feedback-dependent postprocessing
    if isempty(post_proc.proc),
      % Concatenate all classifier outputs into a huge vector. Each cell
      % entry is a column vector
      udp = vertcat(out{:}, mrkOut{:});
    else
      % Feedback-dependent postprocessing with parameters
      udp = feval(post_proc.proc, out, mrkOut, post_proc.proc_param{:});
    end
    
    % This routine needs to convert the postprocessed classifier output
    % to something that can be sent over upd. This routine might need
    % things like player number, so give it all the setup 
    if isfield(bbci,'feedback') & ~isempty(bbci.feedback)
      udp = feval(['bbci_bet_feedbacks_' bbci.feedback], udp, bbci);
    end
    % The feedback routine needs to return a double array

    %%% send udp%%%
    controlnumber = controlnumber+1;
    
    if isempty(bbci.fb_machine)
      if run_fb
        if ~loop_fb; feedback_opt.reset = 1; loop_fb = true; end
        dat = num2cell(udp);
        for bla = 1:length(parPort_list)
          do_set(parPort_list(bla));
          do_set('counter',controlnumber,timestamp,logFileNumber(1));
        end
        if ~isempty(feedback_opt.type)
          feedback_opt = feval(feedback_opt.type,fig,feedback_opt,dat{:});
        end
      else
        set(fig,'Visible','off');
      end
      drawnow;
    else
      if parport_on
        send_data_udp([logFileNumber(1); controlnumber;timestamp;length(parPort_list); parPort_list';udp]);
      else
        send_data_udp([logFileNumber(1); controlnumber;timestamp;0;udp]);
      end
    end
    
    parPort_list = [];
    
    %%% log udp %%%
    writeClassifierLog('udp',timestamp,udp);
    
    setupchanged = 0;
    get_setup;
    %%% GIVES VARIABLE setupchanged BACK IF SOMETHING CHANGES %%%
    if setupchanged
      %%%% SEND MARKER TO PARALLEL PORT IF FEEDBACK IS CHANGED %%%%
      parPort_list = [parPort_list,229+bbci.player];
      if old_port~=bbci.fb_port | ~strcmp(old_machine,bbci.fb_machine)
        send_data_udp;
        
        if isempty(bbci.fb_machine)
          if ~isempty(bbci.other_clients)
            warning('other clients not supported in this mode');
          end
          feedback_opt.reset = 1;
          if isfield(feedback_opt,'parPort') & feedback_opt.parPort==1
            do_set(255);
          end
   
          if isfield(feedback_opt,'graphic_master') & feedback_opt.graphic_master
            error('bbci_bet_apply is not allowed to be graphic_master');
          end
          loop_fb = true;
          run_fb = true;
        else
          if isempty(bbci.other_clients)
            send_data_udp(bbci.fb_machine,bbci.fb_port);
          else
            send_data_udp(cat(2,{bbci.fb_machine},bbci.other_clients),[bbci.fb_port,bbci.other_ports]);
          end
          old_port = bbci.fb_port;
          old_machine = bbci.fb_machine;
        end
      end
      
      writeClassifierLog('change', timestamp, new_setup);
    end

    if opt.realtime,
      waitForSync(blocksize);
    end
  end  % end while loop schleife
%catch
%  % inform about the error
%  errorMessage = lasterr;
%end


  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% Close all communications %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  send_data_udp;if exist('fig','var'), try,close(fig), end; end
  get_data_udp(control_communication);
  writeClassifierLog('exit',timestamp);
  
  
  %%%%%%%%%%%%%%%%%%%%%%%
  %%%%% Say goodbye %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%
  
end

if ~isempty(errorMessage)
  writeClassifierLog('message',timestamp, ['Error: ' errorMessage]);
  error(errorMessage);
end


fprintf('bbci_bet finishes\n');
