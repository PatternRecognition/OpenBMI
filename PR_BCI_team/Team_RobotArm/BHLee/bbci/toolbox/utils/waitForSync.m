%rc= waitForSync(t)
%
% wait for timing synchronization purpose
%
% use waitForSync or waitForSync(0) for initialization. 
% subsequent calls waitForSync(t) wait until t msec have passed since 
% the last call to waitForSync.
% if waitForSync was called too late, the lateness [msec] is returned
% and the next time step will be measured from the actual calling time
%
% >> waitForSync; tic; pause(rand); waitForSync(1000); pause(0.9); waitForSync(1000); toc
% elapsed_time =
%     2.0000
%
% >> waitForSync; pause(1.1); waitForSync(1000)
% ans =
%        104.16
