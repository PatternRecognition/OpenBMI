% Make a buffer for two/four channels, length 3
storeContData('init', 2, [2 4], 'ringBufferSize', 3, 'fs', 1000);
% Accessing it now should raise an error
w = storeContData('window', 1, 1);
% Store a too long sequence
storeContData('append', 1, [1 1; 2 2; 3 3; 4 4]);
% Retrieving a window of length 1 should return the 4's
storeContData('window', 1, 1)
storeContData('window', 1, 2)
% should return 2, 3, 4
storeContData('window', 1, 3)
% This should give an error
storeContData('window', 1, 4)
% This as well (data too small)
storeContData('append', 1, 5);
% Test writing of data when there is overflow in the writing
storeContData('append', 1, [5 5]);
storeContData('window', 1, 2)
storeContData('append', 1, [6 6; 7 7]);
storeContData('window', 1, 2)
storeContData('append', 1, [8 8; 9 9]);
storeContData('window', 1, 2)
% This should return 7, 8
storeContData('window', 1, 2, -1)
% This is the tricky one: Interval with timeshift exceeds buffer length,
% need to pad with lots of NaNs
storeContData('window', 1, 3, -1)
storeContData('window', 1, 3, -2)
storeContData('window', 1, 3, -3)
% Also in the case where interval length exceeds the buffer size:
storeContData('window', 1, 4, -3)

storeContData('append', 2, [5 5 5 5]);
storeContData('window', 2, 1)

% Store a too long sequence
storeContData('append', 1, [1 1; 2 2; 3 3; 4 4]);
% Try with timeshift: This should return the 3
storeContData('window', 1, 1, -1)
storeContData('window', 1, 2, -1)
storeContData('window', 1, 3, -1)
storeContData('window', 1, 1, -2)

storeContData('cleanup')
