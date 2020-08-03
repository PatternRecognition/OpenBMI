function storeData(fs, host, folder, filename, varargin)
% storeData(fs, host, folder, filename, <OPT>)
% storeData(fs, host, folder, filename, <SCALE>)
%  
%  storeData uses acquire_func to recevie data. Store data writes the data
%  to a file in the brainvision dataformat.
%
% INPUT:   fs : The sampling frequency for the acquire function
%        host : The host for the acquire function
%      folder : The folder for the data file. (We require here an absolute
%               path.)
%    filename : The filename for the data file.
%         OPT : struct or property/value list of optional properties:
%               'scale': factor applied to data before storing them
%                        as INT16.
%               'quit_marker': marker that cause the function to
%               quit storing the data, default 'S254'.
%               Integers are interpreted as stimulus markers.

% 2011/05/19 - Max Sagebaum
%              - change logic to imediatly append to files.
%              - added gracefull exit
% 2011/09/01 - Max Sagebaum
%              - added information about sample count and time to the
%                message box.
% 2011/09/15 - Max Sagebaum
%              - Update is now every 10 ms.
%              - Removed milliseconds and samples from the display
% 2011/10/19 - Martijn Schreuder
%              - Using default scale of 0.1, to reduce sampling loss
% 2011/11/17 - Benjamin Blankertz
%              - Allow selection of quit marker
% 2011/11/22 - Benjamin Blankertz
%              - Store time in 'New Segement' marker
 
  global acquire_func
  
  if length(varargin)==1 && isnumeric(varargin{1}),
    opt= struct('scale', varargin{1});
  else
    opt= propertylist2struct(varargin{:});
  end
  opt= set_defaults(opt, ...
                    'scale', 0.1, ...
                    'quit_marker', 'S254', ...
                    'position', [50 50 155 80]);
  if isnumeric(opt.quit_marker),
    opt.quit_marker= sprintf('S%3d', opt.quit_marker);
  end
  
  state = acquire_func(fs,host);
  
  if usejava('jvm')
  % create a msgbox displaying the string
      figureHandle = figure('Name', 'StoreData' ,'MenuBar','none', 'Position',opt.position);
      uicontrol(figureHandle,'Style','text', 'String','Time: ', 'Position',[5 50 50 20]);
      timeHandle = uicontrol(figureHandle,'Style','text', 'String','0', 'Position',[55 50 100 20]);
      buttonHandle = uicontrol(figureHandle,'Style','pushbutton','String','Stop','Position',[55 0 50 25]);
      set(buttonHandle,'Callback', @stopCallback); 

      % create the two anonymous functions
      funcHandles.Stop = @() stopfun(buttonHandle); % false if message box still exists
      funcHandles.Clear = @() clearfun(figureHandle); % delete message box
      fig_exist = 1;
  else
      funcHandles.Stop = 0;
      funcHandles.Clear = [];
      fig_exist = 0;
  end
  
  fullName = [folder '/' filename];
  
  % write the header
  opt_hdr = struct;
  
  % set scaling factor
  opt_hdr.scale = opt.scale;
 
  opt_hdr.fs = fs;
  opt_hdr.clab = state.clab;
  opt_hdr.precision = 'int16';
  opt_hdr.DataPoints= 0;
  eegfile_writeBVheader(fullName, opt_hdr);
  
  % write the empty marker file
  eegfile_writeBVmarkers(fullName);
  
  % open the marker file and the data file
  dataFile = fopen([fullName '.eeg'],'w');
  mrkFile = fopen([fullName '.vmrk'],'A');
  if(-1 == dataFile || -1 == mrkFile) 
    error('cannot write to %s.vhdr or %s.eeg', fullName, fullName);
  end
  
  % output the marker section start
  msg= sprintf('Mk1=New Segment,,1,1,0,%s000', ...
               datestr(now,'yyyymmddHHMMSSFFF'));
  fprintf(mrkFile, [msg 13 10]);
  
  mrkCount = 1;
  sampleCount = 0;
  run = 1;
  
  startTime = rem(now,1);
  lastDraw = clock;
  
  while(~funcHandles.Stop() && run)
    [x, blockNr, mrkPos, mrkType, mrkDesc] = acquire_func(state);    
    xSize = size(x,1); 
    % scale the data
    x = diag(1./opt_hdr.scale)*x';

    % write the data to the eeg file
    fwrite(dataFile, x, 'int16');

    for i = 1:length(mrkPos)
      mrkCount = mrkCount + 1;
      fprintf(mrkFile, ['Mk%d=%s,%s,%d,1,0' 13 10], mrkCount, mrkDesc{i}, mrkType{i}, sampleCount + mrkPos(i));
    end

    % add the samples to the sampleCount
    sampleCount = sampleCount + xSize;
    
    curDrawTime = clock;
    if(fig_exist && etime(curDrawTime, lastDraw) > 1/60)
      % update the data in the figure
      curTime = rem(now,1);
    
      set(timeHandle,'String', datestr(curTime - startTime, 'HH:MM:SS'));
      
      drawnow;
      lastDraw = curDrawTime;
    end

    % check if we received stop signal
    if any(strcmp(mrkType, opt.quit_marker))
      run = 0;
    end
  end
  
  funcHandles.Clear();
  
  fclose(dataFile);
  fclose(mrkFile);
  
  acquire_func('close');
end

function stopCallback(hObject,eventdata)
  if ishandle(hObject),
      delete(hObject) ;
  end
end

function r = stopfun(H)
 r = ~ishandle(H) ;
end

function clearfun(H)
  if ishandle(H),
      delete(H) ;
  end
end
