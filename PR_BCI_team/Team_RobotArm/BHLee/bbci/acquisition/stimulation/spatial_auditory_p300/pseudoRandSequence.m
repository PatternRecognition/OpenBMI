function [posSer, posConn] = pseudoRandSequence(varargin),

% Code to find the possible sequences (given the constraints) for the auditory P300 setup.
%
% Input: 
%  nrSpkr = varargin{1}		- nr of speakers that the sequence should be defined for
%  nrApart = varargin{2}	- distance (in nr of speakers) that should be between a stimulus and the next
%  nrBetweenRep = varargin{3}	- nr of stimuli that need to be between stimulation of the same direction
%  circular = varargin{4}	- do speaker 1 and N connect (are the neighbors?)
%
% Output:
%  posSer			- ALL possible combinations of stimuli in a single iteration
%  posConn			- ALL possible connections between ALL different rows of posSer
%
% Warning: because if its exhaustive nature, this function should not be called on the fly for a
% number of speakers larger than 6. Rather, the routine should be called and the result saved in 
% mat file. Loading this and then performing operations on it is the better way to go. 
%
% martijn@cs.tu-berlin.de

nrSpkr = varargin{1}; % nr of speakers that the sequence should be defined for
nrApart = varargin{2}; % distance (in nr of speakers) that should be between a stimulus and the next
nrBetweenRep = varargin{3}; % nr of stimuli that need to be between stimulation of the same direction
circular = varargin{4}; % do speaker 1 and N connect (are the neighbors?)
    
if length(varargin) == 4,
    %% init the recursive method
    fprintf('\nFinding all possible series.\n\n');
    initRun = 1;
    posSer = pseudoRandSequence(varargin{1}, varargin{2}, varargin{3}, circular, [1:nrSpkr]');
elseif length(varargin) == 5,
    posSer = varargin{5};
    if size(posSer, 2) == nrSpkr,
        % do nothing but return the posSer. The proper length has been
        % reached.
    else
        % add the new possibilities
        newSeq = NaN*ones([size(posSer,1)*(nrSpkr-size(posSer,2)), size(posSer,2)+1]);
        changeDone = 0;
        delRows = [];
        newSeqIdx = 1;
        for i = 1:size(posSer,1),
            excl = [posSer(i,end)-(nrApart-1):posSer(i,end)+(nrApart-1)];
            if circular,
                excl(find(excl > nrSpkr)) = excl(find(excl > nrSpkr))-nrSpkr;
            end
            addItems = setdiff([1:nrSpkr], [posSer(i,:) excl]);
            if ~isempty(addItems),
                changeDone = 1;
                for j = 1:length(addItems),
                    newSeq(newSeqIdx, :) = [posSer(i,:), addItems(j)];
                    newSeqIdx = newSeqIdx + 1;
                end
            end
        end
        if ~changeDone,
            error('No more additions possible.\n');
        else
            [delRow dmy] = find(isnan(newSeq));
            newSeq(delRow, :) = [];
            posSer = pseudoRandSequence(nrSpkr, nrApart, nrBetweenRep, circular, newSeq);
        end
    end
end


if exist('initRun', 'var') && initRun,
    posConn = [];
    fprintf('\nFinding all possible connections.\n\n');
    for i = 1:size(posSer,1),
        X = [1:size(posSer,1)];

        % ensure nrApart between last and first of both blocks
        excl = [posSer(i,end)-(nrApart-1):posSer(i,end)+(nrApart-1)];
        if circular,
            excl(find(excl > nrSpkr)) = excl(find(excl > nrSpkr))-nrSpkr;
            excl(find(excl < 0)) = excl(find(excl < 0))+nrSpkr;
        else
            excl(find(excl > nrSpkr)) = [];
            excl(find(excl < 0)) = [];
        end
        Xt = find(~ismember(posSer(:,1), excl));
        X = intersect(Xt, X);        
        
        % ensure nrBetweenRep for connections
        for j = 1:nrBetweenRep,
            Xt = find(~any(ismember(posSer(:,1:(nrBetweenRep-(j-1))), posSer(i,end-(j-1))),2));
            X = intersect(Xt, X);
        end

        if size(posConn,2) < size(X,2),
            posConn(i,size(X,2)) = 0;
        end
        posConn(i,1:size(X,2)) = X;
        if ~mod(i, 100),
            fprintf('Finished: %6.2f%s\n', (double(i)*100)/size(posSer,1), '%');
        end
    end
    posConn(find(posConn == 0)) = NaN;
end

end
