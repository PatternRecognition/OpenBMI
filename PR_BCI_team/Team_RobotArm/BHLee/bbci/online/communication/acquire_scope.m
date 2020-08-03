function acquire_scope(fs, host, samples)
% acquire_scope uses acquire_func to get data from a server and display
% it in a figure.
%
% !!!This is currently only a quick imlementation for debugging
% purposes.!!!
%  SYNPOSIS
%    acquire_sigserv(fs, host, samples)                          [start]
%
% ARGUMENTS
%                fs: sampling frequency for the acquire_func
%              host: hostname for the acquire_func
%           samples: The number of samples we display
%       - and many more (in the future)
%
% RETURNS
%     - nothing
% DESCRIPTION
%    acquire_scope will connect to a data source through the acquire_func.
%    The data from the acquire_func will be displayed in a figure.
%
% AUTHOR
%    Max Sagebaum
%   
%    2011/09/15 - Max Sagebaum
%                    - file created
  
  global acquire_func
  
  state = acquire_func(fs,host);
  
  % create a msgbox displaying the string
  figureHandle = figure('Name', 'Acquire Scope' ,'MenuBar','none');
  
  % create the two anonymous functions
  funcHandles.Stop = @() stopfun(figureHandle); % false if message box still exists
  funcHandles.Clear = @() clearfun(figureHandle); % delete message box
  
  run = 1;
  
  channelNumber = length(state.chan_sel);
  dataLines = zeros(channelNumber, samples);  
  markerLine = zeros(1, samples);
  
  dataPos = 1;
  
  lastDraw = clock;
  while(~funcHandles.Stop() && run)
    [x, blockNr, mrkPos, mrkType, mrkDesc] = acquire_func(state);    
    x = x';
    xSize = size(x,2); 
    % scale the data
    x = diag(1./state.scale)*x;
    
    if( dataPos + xSize > samples)
      % we have an overflow handle it
      samplesLeft = samples - dataPos + 1;      
      dataLines(:, dataPos:samples) = x(:,1:samplesLeft);
      
      % update the markers
      pos = find(mrkPos <= samplesLeft);
      markerLine(:, mrkPos(pos) + dataPos - 1) = 1;
      
      %reset the position and update the x data
      dataPos = 1;
      x = x(:, (samplesLeft+1):xSize);
      xSize = size(x,2);
      keepPos = find(mrkPos > samplesLeft);
      mrkPos = mrkPos(keepPos) - samplesLeft; % adjust also for the position change in the xValues
      mrkType = mrkType(keepPos);
      mrkDesc = mrkDesc(keepPos);
    end
    
    dataLines(:, dataPos:(dataPos + xSize -1)) = x(:, 1:xSize);
    markerLine(:, mrkPos + dataPos - 1) = 1;
    
    dataPos = dataPos + xSize;
    
    curTime = clock;
    if(etime(curTime, lastDraw) > 1/60)
      plot([dataLines; markerLine]');
      drawnow;
      lastDraw = curTime;
    end
    
  end
  
  funcHandles.Clear();
  
  acquire_func('close');
end

function r = stopfun(H)
 r = ~ishandle(H);
end

function clearfun(H)
  if ishandle(H),
      delete(H) ;
  end
end