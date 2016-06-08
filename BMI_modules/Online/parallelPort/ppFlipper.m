function ppFlipper(data)
% ppFlipper(data) 
%
% write data on the parallelport; 
% It assumes the parallelport to be at IO_ADDR.
%

% kraulem 07/07

global IO_ADDR
ppWriteStay(IO_ADDR, data);
return;