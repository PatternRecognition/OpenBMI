function soundGeneration(VOL)
% server eeg
warning('off', 'instrument:fread:unsuccessfulRead');
sock = tcpip('localhost', 30000, 'NetworkRole', 'Server');
set(sock, 'InputBufferSize', 1024);
% Open connection to the client
fprintf('%s \n','Client Connecting...');
fopen(sock)
fprintf('%s \n','Client Connected');
set(sock, 'ReadAsyncMode', 'continuous');
set(sock, 'TimeOut', 0);

beepLengthSecs=0.1;
rate=44100;
beepY = MakeBeep(8000,beepLengthSecs,rate) * VOL;
Snd('Open');
tic;
while true
    s = fread(sock, 1);
    if s == 1
        toc
        Snd('Play',beepY,rate);
        tic;
    elseif s == 2
        break;
    end
end
end