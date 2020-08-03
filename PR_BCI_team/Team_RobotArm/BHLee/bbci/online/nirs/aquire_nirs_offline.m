%NIRS offline tester:

%- Select one of the files (could use guiopen to select) to get the data from
%- Open the file and retreive some data.
%- Send the data back in an "online" fashion. For this the following could be done
% while i < length(data)
%    actualFrame = data(i,:)
%    wait(1/sf) % Where sf is the sampling frquency, and hence 1/sf the period
% end
%- Meanwhile, check for markers and other info that the real online function
% should provide.

uiload
if exist('uni')
    nirs = uni;
else
    fprintf('Select a valid (sorted) file');
end
data = nirs.dat;
sf = nirs.sf;
mrk = nirs.mrk;
while i < length(data)
    actualFrame = data(i,:);
    fprintf('.')
    pause(1/sf) % Where sf is the sampling freq, and hence 1/sf the period
    if any(mrk(:,1)==i)
        marker = mrk(mrk(:,1)==i,2);
    end   
end
