
function out = smi(cmd_str, varargin)
%
% USAGE:
%   function out = smi(cmd_str)
%
% IN:   cmd_str     -   command string
%
%                       (1) name of the library function
%
%                       OR:
%
%                       (2) 'init': loads iViewX API library into matlab and
%                                 initializes data structures
%
%                           'connect': connect to iViewX on the smi-laptop via
%                                      ethernet
%
%                           'test_connection': prints a message on the iViewX
%                                              command line if connection works
%
%                           'disconnect': close ethernet connection to iViewX
%                                         on the smi-laptop
%
%                           'outit': unload iViewX API library
%
%       varargin    -   (1) arguments for the library function
%
%                       (2) if cmd_str == 'init': eyetracker default data struct 
%                           if cmt_str == 'connect': connection options
%
%
%
% OUT:      out     -   flag received by the libary (cf. iView X SDK manual
%                       (acquisition\eyetracker\iViewXAPI\Docs\iView X SDK Manual.pdf)
%
% Simon Scholler, 2011
%

global BCI_DIR

v = version;
opt= propertylist2struct(varargin{:});
opt = set_defaults(opt, 'oldmatlab', v(1)<7 || (v(1)==7 && v(3)<5), ...
                        'ip_smi', '127.0.0.1', ...    % IP address of SMI laptop (possibly remote)
                        'ip_bbci', '127.0.0.1', ...   % IP address of (this) BBCI laptop                        
                        'port_smi', 4444,  ...
                        'port_bbci', 5555);           

                      
switch cmd_str
    
    case 'init' %loads iViewX API library into matlab and initializes data structures      
        if opt.oldmatlab
            header = 'iViewXAPI_oldmatlab.h';
            init_fcn = str2func('eye_initAPI_oldmatlab');            
        else
            header = 'iViewXAPI_newmatlab.h';
            init_fcn = str2func('eye_initAPI_newmatlab');                        
        end
      
        d = pwd;
        cd([BCI_DIR '\acquisition\eyetracker\iViewXAPI\Binaries\']);
        try
           if ~libisloaded('iViewXAPI')      
               loadlibrary('iViewXAPI.dll', header);
           end
        catch e
           error(e.identifier, [e.identifier ': ' e.message '\n\n iViewXAPI not installed or path not set.\n', ...
                'Note that for 64-bit systems, you need matlab version 8 or above, otherwise the c-compiler does not work.'])
        end
        cd(d)
        
        [out.pSystemInfoData, out.pSample32Data, out.pEvent32Data, out.pAccuracyData, out.Calibration] = init_fcn();
        
        out.Calibration.targetFilename = int8('');
        out.pCalibrationData = libpointer('CalibrationStruct', out.Calibration);        
        
    case 'connect' % connect to iViewX on the smi-laptop via ethernet
        if ~isnumeric(opt.port_smi) || ~isnumeric(opt.port_bbci)
           error('Port must be numeric.')
        end
        if ~ischar(opt.ip_smi) || ~ischar(opt.ip_bbci)
           error('IP adresses must be strings.')
        end
        if opt.oldmatlab
           out = calllib('iViewXAPI', 'iV_Connect', opt.ip_smi, int32(opt.port_smi), opt.ip_bbci, int32(opt.port_bbci));
        else
           out = calllib('iViewXAPI', 'iV_Connect', int8(opt.ip_smi), int32(opt.port_smi), int8(opt.ip_bbci), int32(opt.port_bbci));
        end
        
    case 'test_connection'  % prints out a message on the iViewX command line
        out = calllib('iViewXAPI', 'iV_SendCommand', int8('Connection working!'));
        
    case 'disconnect'  % close ethernet connection to iViewX on the smi-laptop      
        out = calllib('iViewXAPI', 'iV_Disconnect');
        
    case 'outit'    % unload iViewX API library
        d = pwd;
        cd([BCI_DIR '\acquisition\eyetracker\iViewXAPI\Binaries\']);
        unloadlibrary('iViewXAPI');
        cd(d)
        
    case 'sync_marker'
        out = calllib('iViewXAPI', 'iV_SendCommand', int8('ET_REM SyncMarker'));
        
    otherwise  % call library function
        calllib('iViewXAPI', cmd_str, varargin{:});
end


