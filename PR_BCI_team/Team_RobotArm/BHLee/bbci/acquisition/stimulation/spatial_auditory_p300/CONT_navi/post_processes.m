function [newState, selection, output] = post_processes(clOut, currentState, Lut, history, varargin);
%POST_PROCESSES Summary of this function goes here
%   Detailed explanation goes here

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
    'test', []);

if ~isfield(opt.procVar, 'WinApi'),
    import java.awt.Robot
    import java.awt.event.KeyEvent
    opt.procVar.WinApi = Robot();
    opt.procVar.Left = KeyEvent.VK_LEFT;
    opt.procVar.Right = KeyEvent.VK_RIGHT;
    opt.procVar.Up = KeyEvent.VK_UP;
    opt.procVar.Down = KeyEvent.VK_DOWN;    
end
  
% release all buttons
opt.procVar.WinApi.keyRelease(opt.procVar.Left);
opt.procVar.WinApi.keyRelease(opt.procVar.Right);
opt.procVar.WinApi.keyRelease(opt.procVar.Up);
opt.procVar.WinApi.keyRelease(opt.procVar.Down);
pause(0.01);

threshold = 0.3;
% calculate which keys to press
if clOut.vec(1) > threshold,
    opt.procVar.WinApi.keyPress(opt.procVar.Up);
end
if clOut.vec(1) < -threshold,
    opt.procVar.WinApi.keyPress(opt.procVar.Down);
end
if clOut.vec(2) > threshold*2,
    opt.procVar.WinApi.keyPress(opt.procVar.Right);
end
if clOut.vec(2) < -threshold*2,
    opt.procVar.WinApi.keyPress(opt.procVar.Left);
end
% press keys until next stimuli

newState = 1;
selection = [];
output = opt.procVar;
clOut.vec
end