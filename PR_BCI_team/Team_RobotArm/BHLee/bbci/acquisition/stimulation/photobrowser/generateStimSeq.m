Sequences = [];
mSizeX = 8;
mSizeY = 6;
GroupSize = 6;
FramesToFind = 15;
DiagCondActive = 1;
TestMode = 1;
maxCandidateFrames = 25;

numSequences=1000;
SaveName=['/home/schroedm/svn/ida/public/bbci/acquisition/stimulation/photobrowser/Seq_Screensize_' int2str(mSizeX) 'x' int2str(mSizeY) '_GroupSize_' int2str(GroupSize) '_Frames_' int2str(FramesToFind) '.mat'];



for i=1:numSequences
    disp(['Generating sequence no. ' int2str(i) ' of ' int2str(numSequences)]);
    [seq, stat] = pseudoRandMatrix(mSizeX,mSizeY,GroupSize,FramesToFind,DiagCondActive,TestMode, maxCandidateFrames);
    Sequences{i}.seq = seq;
    Sequences{i}.stat = stat;
end

save([SaveName],'Sequences')