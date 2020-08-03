function [seq, stat] = pseudoRandMatrix(mSizeX,mSizeY,GroupSize,FramesToFind,DiagCondActive,TestMode, maxCandidateFrames)


% [seq] = pseudoRandMatrix(mSizeX,mSizeY,GroupSize,FramesToFind,DiagCondActive)
% generates a predefined number of pseudo-randomized stimulation sequences
% e.g. for the elements of the 2D matrix photo browser.
% Generated sequences obey a number of conditions, that avoids neighboring
% elements to be highlighed close in time.
%
% Input:
%   mSizeX: (1x1) integer - Size of the 2D matrix in X direction
%   mSizeY: (1x1) integer - Size of the 2D matrix in Y direction
%   GroupSize:  (1x1) integer - Group size of elements to be highlighted at the same time.
%                           The repetitive highlighting of several groups forms a sequence.
%                           At the end of a sequence, every matrix element has been highlighted once.
%   FramesToFind: (1x1) integer - Number of different sequences to be generated.
%   DiagCondActive: logic 1/0   - Turn on/off the check for diagonal neighboring elements (default: 1)
%   TestMode: logic 1/0   - Turn on/off small visualization of test
%                           procedure that analyzes the distribution of
%                           group members highlighted together.
%   maxCandidateFrames: (1x1 integer) - number of candidate frames that are
%                                       tested according to variance criterion in 
%                                       in order to append the best one
%                                       to the current sequence.
%
% Example:
%
% seq = pseudoRandMatrix(5,6,5,2,1,0)
% Loesung gefunden 1 Nachbarn zwischen den Zeilen: 26
% Loesung gefunden 2 Nachbarn zwischen den Zeilen: 25
% Beste gefundene Loesung (nach Nachbarn zwischen Zeilen): 2 mit einem Wert von 25
% seq(:,:,1) =
%     13    28    10    16     3
%     14    24     1    26    12
%     18     8    29    27    20
%     23     7     9    21    25
%     19    22     2    11     5
%     17     4     6    15    30
% seq(:,:,2) =
%     27    11    24     9     2
%     23    25    12    21    15
%      6    20    18     3     5
%     22    13    30    10     1
%      8    28    17    19    26
%      7    29     4    16    14
%
% David List, Michael Tangermann Januar 2010

if(nargin<6)
    TestMode = 0;
end
if(nargin<5)
    DiagCondActive = 1;
end
if (nargin==3)
    FramesToFind = 1;
end

% Sanity check of dimensionality
if rem(mSizeX*mSizeY,GroupSize)~=0
    warning('Group size (GroupSize) must be a integer factor of matrix size (mSizeX*mSizeY)');
    return;
end

currentGroup  = [];
seq = [];
sizeArr = mSizeX * mSizeY;
groupsPerFrame = ceil(sizeArr/GroupSize);
neighbours = [];

LUT_Neighborhood = zeros(sizeArr,8);
for i = 1:sizeArr
    LUT_Neighborhood(i,:) = getNeighbours(mSizeX, mSizeY, i,DiagCondActive);
end

for idxCurrentFrame = 1:FramesToFind  %  Zeitbegrenzung, falls es keine komplette Loesung gibt.
    FrameFound = 0;
    idxCurrentFrame;
    bestFrame=[];
    varBestFrame = [];
    for idxCandidateFrame=1:maxCandidateFrames %tryNumFrames


        FrameFound = 0;
        while(~FrameFound)
            randomVector = randperm(sizeArr);
            currentFrame = [];
            lenghtCurrentFrame = 0;
            for k = 1:groupsPerFrame  % number of groups needed
                currentGroup = [];
                neighbours = [];
                % Fill one line of the sequence (a group)
                for j = 1:GroupSize
                    PtrRandomVectorElement = length(randomVector);
                    candidate = randomVector(PtrRandomVectorElement); % chose last element of permutation vector
                    while((ismember(candidate, neighbours) == 1))
                        PtrRandomVectorElement = PtrRandomVectorElement - 1;
                        if(PtrRandomVectorElement == 0) break; end % no candidate left: frame finished, or no solution for remaining elements
                        candidate = randomVector(PtrRandomVectorElement);
                    end
                    if(PtrRandomVectorElement > 0)
                        neighbours = [neighbours,LUT_Neighborhood(candidate,:)];
                        randomVector(PtrRandomVectorElement) = [];
                        currentGroup = [currentGroup,candidate]; % candidate finally added to a line!
                    else  % no candidate left: frame finished, or no solution for remaining elements
                        break;
                    end
                end

                if (length(currentGroup) == GroupSize) % Group (line) is filled. Add it to the current frame
                    lenghtCurrentFrame = lenghtCurrentFrame + GroupSize;
                    currentFrame = [currentFrame;currentGroup];
                end

                if(PtrRandomVectorElement == 0)
                    break; % no candidate left: frame finished, or no solution for remaining elements
                end
            end % Frame is filled

            if(lenghtCurrentFrame == sizeArr)    % Is current frame completely filled (a valid frame)?
                FrameFound = 1;
            end
        end

        % ToDo: Prevent these tests in case of first frame
        if idxCandidateFrame==1
            seq(:,:,idxCurrentFrame) = currentFrame;
            stat = test_sequenceStatistics(seq, mSizeX, mSizeY, 0);
            bestFrame=currentFrame;
            %idxCandidateFrame
            %varBestFrame=stat.var_overallSeq
        else
            seq(:,:,idxCurrentFrame) = currentFrame;
            stat = test_sequenceStatistics(seq, mSizeX, mSizeY, 0);
            if stat.var_overallSeq < varBestFrame
                %idxCandidateFrame
                %varBestFrame=stat.var_overallSeq
                bestFrame=currentFrame;
            end
        end
    end    
    seq(:,:,idxCurrentFrame) = bestFrame;
end


if TestMode
    stat = test_sequenceStatistics(seq, mSizeX, mSizeY, 1);
end

end % function




%%
function [neighbours] = getNeighbours(mSizeX,mSizeY,element,DiagCondActive)
% Search for a single element of matrix (mSizeX,mSizeY) all neighbors
% within the matrix.
%
% Input:
%   mSizeX: (1x1) integer - Size of the 2D matrix in X direction
%   mSizeY: (1x1) integer - Size of the 2D matrix in Y direction
%   element: (1x1) integer - matrix position index to find neighbors for
%   DiagCondActive: logic 1/0   - Turn on/off the check for diagonal neighboring elements (default: 1)
%
% Output:
%
%   neighbours: (1x8) integer - list of matrix positions where neighbors
%                               are located
%
% David List, Jan 2010

size = mSizeX * mSizeY;
neighbours = [];

% Add neighbours in column
% only one if first or last column
if(mod(element-1,mSizeX) == 0)
    neighbours = [neighbours,element+1];
    col = 1;
elseif(mod(element,mSizeX) == 0)
    neighbours = [neighbours,element-1];
    col = -1;
else
    neighbours = [neighbours,element+1];
    neighbours = [neighbours,element-1];
    col = 0;
end

% Add neighbours in column
% only one if first or last line
if(element<=mSizeX)
    neighbours = [neighbours,element+mSizeX];
    if(col == 1 && DiagCondActive == 1)
        neighbours = [neighbours,element+mSizeX+1];
    elseif(col == -1 && DiagCondActive == 1)
        neighbours = [neighbours,element+mSizeX-1];
    elseif (DiagCondActive == 1)
        neighbours = [neighbours,element+mSizeX+1];
        neighbours = [neighbours,element+mSizeX-1];
    end
elseif(element>(size - mSizeX))
    neighbours = [neighbours,element-mSizeX];
    if(col == 1 && DiagCondActive == 1)
        neighbours = [neighbours,element-mSizeX+1];
    elseif(col == -1 && DiagCondActive == 1)
        neighbours = [neighbours,element-mSizeX-1];
    elseif(DiagCondActive == 1)
        neighbours = [neighbours,element-mSizeX+1];
        neighbours = [neighbours,element-mSizeX-1];
    end
else
    neighbours = [neighbours,element+mSizeX];
    neighbours = [neighbours,element-mSizeX];
    if(col == 1  && DiagCondActive == 1)
        neighbours = [neighbours,element+mSizeX+1];
        neighbours = [neighbours,element-mSizeX+1];
    elseif(col == -1  && DiagCondActive == 1)
        neighbours = [neighbours,element+mSizeX-1];
        neighbours = [neighbours,element-mSizeX-1];
    elseif(DiagCondActive == 1)
        neighbours = [neighbours,element+mSizeX+1];
        neighbours = [neighbours,element-mSizeX+1];
        neighbours = [neighbours,element+mSizeX-1];
        neighbours = [neighbours,element-mSizeX-1];
    end
end

while (length(neighbours) < 8)
    neighbours = [neighbours,NaN];
end

%matrix = zeros(mSizeY,mSizeX);
%for k = 1:mSizeY
%matrix(k,:) = ((k-1)*mSizeX+1):mSizeX*k;
%end
%X = rem(element,mSizeX);
%Y = floor(element/mSizeX)+ 1;
%neighbours = [matrix(X+1,Y),matrix(X-1,Y),matrix(X,Y-1),matrix(X,Y+1),matrix(X+1,Y+1),matrix(X-1,Y+1),matrix(X+1,Y-1),matrix(X-2,Y-1)]
end


%%
function stat = test_sequenceStatistics(seq, xDim, yDim, plotit)

% Test statistic of generated sequence. Plot image.
%   seq = pseudoRandMatrix(5,6,5,num_frames,1)
%   test_sequenceStatistics(seq, 5, 6)

numMatrixElements = xDim*yDim;
count=zeros(1,numMatrixElements);
num_frames = size(seq,3);

count=zeros(numMatrixElements,numMatrixElements);

for target=1:numMatrixElements % Teste alle Targets

    for i=1:num_frames % Gehe durch alle frames einer Sequenz
        seq_tmp=squeeze(seq(:,:,i)); % 3D nach 2D
        [row,col]=find(seq_tmp==target); % Finde Zeilen mit Target

        for num_row=1:size(row,1)  % Für jedes non-Target Element in dieser Zeile zaehle hoch
            count(target, seq_tmp(row , setdiff(1:size(seq_tmp,2),col(1)))) = count(target, seq_tmp(row , setdiff(1:size(seq_tmp,2),col(1))))+1;
        end
    end
end

if plotit
    imagesc(count) ; colorbar;
    title('Number of simultaneous highlights of neighboring elements')
    xlabel('Index of neigbor elements of same group/flash')
    ylabel('Target index')
end
stat.count = count;

% stat.loss_byElement = sum(((count.*(numMatrixElements/(num_frames*(size(seq,2)-1)))).^2)')./numMatrixElements;
% stat.loss_overallSeq = mean(stat.loss_byElement);

stat.var_byElement = var(count');
sorted = sort(stat.var_byElement);

% %if size(sorted,2)>3
%     % Loss based on top 3 and bottom 3 variances
%     stat.var_overallSeq = sum(sorted(end-2:end)) - sum(sorted(1:3));
% % else
% %     stat.var_overallSeq = max(stat.var_byElement);
% % end
    

% loss based on maximal variance
stat.var_overallSeq = max(stat.var_byElement);




end




