function [MLI] = proc_detectMarkerfulIntervals(mrkPos, xLength, SollDist, tailWidth)
% proc_detect_MarkerfulIntervals - calculate time intervals that are densely
% populated with markers and return interval boundaries. Useful to reduce
% large cnt files before further processing.
%
% Synopsis: 
%  [ MLI ] = proc_detectMarkerfulIntervals(mrkPos, xLength, SollDist, tailWidth)
%
%  INPUT ARGUMENTS
%    'mrkpos'      marker positions of cnt file, e.g. in mrk.pos
%    'xLength'     number of samples in cnt file, e.g. in size(cnt.x,1)
%    'SollDist'    minimal distance (in samples) between markerful intervals
%                  needed to define two different intervals
%    'tailWidth'   markerful interval boundaries are returned with an extra number of
%                  tailWidth sample points at both ends, e.g. 
%  
%  OUTPUT
%    MLI      - size [2 nIntervals] array of interval boundaries (in sample points)
%     	        Every column of MLI defines a time interval
%	        
%
% See also:  proc_concatBlocks
%  
% Author(s): Michael Tangermann, TU Berlin
%

  
  if (2*tailWidth) >= SollDist
    fprintf(' SollDist should be at least twice as long as tailWidth \n. ABORT.\n');
    return;
  end
  
  markerPos = (sort(mrkPos));
  i_goodIntervals = find (diff([-SollDist markerPos xLength+SollDist])>SollDist);
  MLI = [markerPos(i_goodIntervals(1:end-1))-tailWidth ; ...
	 markerPos(i_goodIntervals(2:end)-1)+tailWidth];

  % Take care that intervals do not extend outside of CNT
  MLI(1,1)= max( MLI(1,1),1);
  MLI(end,end) = min(MLI(end,end),xLength);
  
  return
  
  