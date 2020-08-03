function res = pyff(command, varargin)

% PYFF -  wrapper function used to initialize and communicate with Pyff
%         (http://bbci.de/pyff) from Matlab
%
%Usage:
% res = pyff(command, <OPT>)
%
%IN:
% 'command' -  command to be issued: 'startup', 'init', 
%              'setdir','set', 'get', 'refresh','play', 'stop','quit',
%              'save_settings', 'load_settings'
%
% res = pyff('startup',<OPT>): startup pyff in a separate console window.
% res contains the command string that is executed. For
% this command, OPT is a property/value list or struct of options with 
% the following (optional) fields/properties:
%  .dir      - pyff source directory (default for win: D:\svn\pyff\src;
%              default for unix: ~/svn/pyff/src)
%  .parport     - port number for parallel port. Default is dec2hex(IO_ADDR),
%              if IO_ADDR exists, otherwise [].
%  .a        - additional feedback directory (default [])
%  .gui      - if 1, pyff gui is started along with the feedback controller (default 0)
%  .l        - loglevel (default 'debug')
%  .bvplugin - if 1, the BrainVision Recorder plugin is started (win: default 1;
%              unix: default 0)
%
% pyff('init',feedback): load and initialize pyff feedback 
%  feedback  - strint containing the name of the feedback
%
% pyff('setdir',<OPT>): set directory and filename for EEG recording. 
%  .today_dir - directory for saving EEG data (default TODAY_DIR)
%  .vp_code   - the code of the VP (Versuchsperson) (default VP_CODE)
%  .basename  - the basename of the EEG file (default '')
% Special cases: If OPT=='' all entries are set to '', ensuring that no EEG is 
% recorded. If OPT is not given at all, all entries are set to default.
%
% pyff('set',OPT): set feedback variables after the feedback has been
% initialized. OPT is a property/value list or struct of with
% fields/properties referring to names and values of variables. The type of
% the variable (string, integer, or float) is automatically checked. 
% NOTE: Unlike Python, Matlab automatically turns integers into floats! 
% For example, var1 = 1.0 and var1 = 1 both yield a float. This should not be a problem 
% in most cases, but if you definitely need an integer, you should provide the 
% type along with the value, e.g. pyff('set','var1',int16(1)).
% Alternatively, you can use the 'setint' command (below).
%
% pyff('setint',OPT): use this command to set feedback variables
% explicitly to integers. If you do this, each provided variable is
% cast to an integer.
%
% res = pyff('get',...): get the value of feedback variables.
% Give a list of variables to be inspected. res is a cell array containing
% the respective values of the variables. [wishful thinking, I think it's
% not possible]
%
% pyff('play'): start playing currently initialized feedback
% pyff('play', 'basename', BASENAME, <PARAM>): start acquisition in BV Recorder, then start playing
%    currently initialized feedback. PARAM can be 'impedances', 0 to avoid
%    impedance measurement.
% pyff('stop'):  stop current feedback
% pyff('quit'):  quit current feedback (and stop acquisition in BV
%    Recorder).
%
% pyff('load_settings', FILENAME),
% pyff('save_settings', FILENAME): load or save the parameters of the feedback
%   to FILENAME. The appendix '.json' is automatically appended.
%   If FILENAME does not contain file separators '\' or '/',
%   TODAY_DIR is prepended.
%
%
% General OPT:
% 'os'       - operating system 'win' or 'unix' (usually you do not need to
%              set this since the script figures out your OS)
% 'replace_html_entities' - replaces symbols like '<' in string to be set via
%                    XML by their according HTML (XML compatible) entities
%OUT:
% res        - final command as a string
%
% ISSUES: 
% *command 'getvariables' (aka 'refresh') does not make variables
% appear in the GUI

% Matthias Treder 2010

global IO_ADDR TODAY_DIR VP_CODE acquire_func general_port_fields
persistent ACQ_STARTED

if strcmp(command, 'init'),
  opt = [];
  ACQ_STARTED= 0;
elseif ismember(command, {'save_settings','load_settings'}),
  opt = [];
  if length(varargin)~=1,
    error('load/save_settings expetects exactly one argument (filename)');
  end
  settings_file= [varargin{1} '.json'];
  if ~any(ismember('/\', settings_file)),
    settings_file= [TODAY_DIR settings_file];
%   % Avoid overwriting? - Maybe it is intended, so we don't.
%    if strcmp(command,'save_settings') && exists(settings_file, 'file'),
%      new_str= datestr(now, 'yyyy-mm-dd_HH:MM:SS.FFF');
%      settings_file= [settings_file, '_', now_str];
%    end
  end
elseif ismember(command, {'set' 'setint'})
  vars = propertylist2struct(varargin{:});
  opt = [];
  if isfield(vars,'replace_html_entities')
    warning 'Found variable ''replace_html_entities'', assuming its a Pyff parameter, not a variable'
    opt.replace_html_entities = vars.replace_html_entities;
    vars = rmfield(vars,'replace_html_entities');
  end
else
  opt= propertylist2struct(varargin{:});
end


% Default settings, separately for different OS

% Figure out os (unless it was set manually)
if ~isfield(opt,'os')
  if isunix || ismac
    opt.os = 'unix';
  else
    opt.os = 'win';
  end
end
% os-specific default options
[opt, isdefault]= ...
        set_defaults(opt, ...
                     'parport',NaN, ...
                     'a',[], ...
                     'gui', 0,...
                     'l', 'debug',...
                     'bvplugin',0, ...
                     'os','win',...
                     'replace_html_entities',1,...
                     'basename','', ...
                     'host', 'localhost', ...
                     'port', 12345, ...
                     'output_protocol', []);
switch(opt.os)
  case 'win'
    opt= set_defaults(opt, ...
                      'dir','D:\svn\pyff\src');
  case 'unix'
    opt= set_defaults(opt, ...
                      'dir','~/svn/pyff/src');
  otherwise
    error('Unsupported os: %s.',opt.os);
end

% Set default port       
if isdefault.parport 
  if exist('IO_ADDR','var')
    opt.parport = dec2hex(IO_ADDR);
  else
    opt.parport = [];
  end
end


% Settings for 'setdir' command
if strcmp(command,'setdir')
  if nargin>1 && strcmp(varargin{1},'')
    opt.today_dir = '';
    opt.vp_code = '';
  else
    ff = isfield(opt,{'today_dir' 'vp_code'}); %% check if fields exist
    if ~ff(1) && exist('TODAY_DIR','var')
      opt.today_dir = TODAY_DIR;
    end
    if ~ff(2) && exist('VP_CODE','var')
      opt.vp_code = VP_CODE;
    end
  end
end
               
%% Execute command
res = [];
switch(command)

  case 'startup',
    
    if strcmp(opt.os,'win')
      % get system path
      curr_path = getenv('PATH');
      % also sets the path back to the normal system variable. Matlab adds a
      % reference to itself to the beginning of the system path which
      % breaks PyQT.QtCore (possibly also other imports that require dll)
      comstr = ['set PATH=' curr_path '& cmd /C "cd ' opt.dir ' & python FeedbackController.py'];
      opt.a= strrep(opt.a, '/', filesep);
    elseif strcmp(opt.os,'unix')
      comstr = ['xterm -e python ' opt.dir  '/FeedbackController.py'];
      opt.a= strrep(opt.a, '\', filesep);
    end
    
    if ~isempty(opt.parport)
      comstr = [comstr ' --port=0x' num2str(opt.parport)];
    end
    if ~isempty(opt.a)
      comstr = [comstr ' -a "' opt.a '"'];
    end
    if opt.gui==0
      comstr = [comstr ' --nogui'];
    end
    if ~isempty(opt.l)
      comstr = [comstr ' -l ' opt.l];
    end
    if opt.bvplugin
      comstr = [comstr ' -p brainvisionrecorderplugin'];
    end
    if ~isempty(opt.output_protocol)
      comstr = [comstr ' --protocol ' opt.output_protocol];
    end
    
    if strcmp(opt.os,'win')
      comstr = [comstr '" &'];
    elseif strcmp(opt.os,'unix')
      comstr = [comstr ' &'];
    end
    system(comstr);
    res = comstr;
    send_udp_xml('init', opt.host, opt.port);
    general_port_fields.feedback_receiver= 'pyff';
    
  case 'init'
    send_udp_xml('interaction-signal', 's:_feedback', varargin{1},'command','sendinit');
    
  case 'setdir'
    send_udp_xml('interaction-signal', 's:TODAY_DIR',opt.today_dir, ...
      's:VP_CODE',opt.vp_code, 's:BASENAME',opt.basename);
    
  case 'set'
    settings= {};
    fn= fieldnames(vars);
    for ii= 1:numel(fn)
      val = vars.(fn{ii});
      % Take value or (if it is a cell) the value of its first element
      if ischar(val) || (iscell(val) && ischar(val{1}))
        typ= 's:';
        if opt.replace_html_entities
          val = replace_html_entities(val);
        end
      elseif isinteger(val) || (iscell(val) && isinteger(val{1}))
        typ= 'i:';
      elseif islogical(val)
        typ= 'b:';
        if val
          val= 'True';
        else
          val= 'False';
        end
      else
        typ= '';
      end
      settings= cat(2, settings, {[typ fn{ii}], val});
    end
    send_udp_xml('interaction-signal', settings{:});
  
  case 'setint'
    settings= {};
    fn= fieldnames(vars);
    typ= 'i:';
    for ii= 1:numel(fn)
      settings= cat(2, settings, {[typ fn{ii}], vars.(fn{ii})});
    end
    send_udp_xml('interaction-signal', settings{:});

  case 'play'
    if isempty(varargin),
      ACQ_STARTED= 0;
    else
      ACQ_STARTED= 1;
      bvr_startrecording(varargin{2}, 'append_VP_CODE',1, varargin{3:end});
      pause(0.01);
    end
    send_udp_xml('interaction-signal', 'command', 'play'); 
    
  case 'stop'
    send_udp_xml('interaction-signal', 'command', 'stop'); 
    
  case 'quit'
    send_udp_xml('interaction-signal', 'command', 'quit'); 
    if strcmp(func2str(acquire_func), 'acquire_bv') && ACQ_STARTED,
        bvr_sendcommand('stoprecording');
        ACQ_STARTED= 0;
    end
     
  case 'save_settings'
    send_udp_xml('interaction-signal', 'savevariables', settings_file);
    
  case 'load_settings'
    send_udp_xml('interaction-signal', 'loadvariables', settings_file);
    
  otherwise 
    error('Unknown command "%s".',command)
end
