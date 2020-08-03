function cond = seq_generation()
% No Input
% OUTPUT :
% struct with fields
%       'trialcondition': the type of trial (for example 2A, 2B, C, etc..)
%       'sequence': the speakers sequence to run (already containing all iterations)
%       'tonesequence': corresponding tone sequence with mapping [4 7 1 3 6 2 8 5]
%       'target': the corresponding target tone from each sequence (speaker number)
%       'soa': SOA interval for each sequence (ms)
%       'seqsize': size of each raw sequence
%       'iterations': number of iterations of the sequence in each trial
% as we have 45 trials in a block (5*9 conditions), the size of each field is 45

cond = struct(  'trialcondition',{{}},...
                'sequence',{{}},...
                'tonesequence',{{}},...
                'target',[], ...
                'soa',[],...
                'seqsize',[],...
                'iterations',[], ...
                'preplaySeq', []);

% useful variables
conditions = {'2A','4A','6A','8A','2B','4B','6B','8B','C'};
% values corresponding to the above conditions
% help quantify the pseudo randomization: 
% avoid 2 consecutives sequences of the same length
val = [1 11 21 31 2 12 22 32 33]; 
sizes = [2 4 6 8 2 4 6 8 8];
soa = [225 225 225 225 900 450 300 225 225];
speak2tone = [4 7 1 3 6 2 8 5];


%% block organization: 45 trials in a random order (5 for each of the 9 conditions)
block = [];

subblock = val(randperm(9));

while(any(abs(diff(subblock))==1))
    subblock = val(randperm(9));
end

block = [block subblock];

for i = 2:5
    subblock = val(randperm(9));
    [v1 idx1] = find(block((i-2)*9+1:(i-1)*9)==33);
    [v2 idx2] = find(subblock==33);
    while(any(abs(diff(subblock))==1)||(abs(block((i-1)*9)-subblock(1))<=2)||(9-idx1+idx2<4))
        subblock = val(randperm(9));
        [v1 idx1] = find(block((i-2)*9+1:(i-1)*9)==33);
        [v2 idx2] = find(subblock==33);
    end
    block = [block subblock];
      
end

[tf, loc] = ismember(block, val);
cond.trialcondition = conditions(loc);
cond.soa = soa(loc);
cond.seqsize = sizes(loc);
% random target tone index for each sequence
cond.target = ceil(sizes(loc).*rand(1,45));

%% generate the 20 different sequences (for the whole block) to be iterated per trial
% we want the same sequence in conditions from A and B modalities
% C modality: take the existing function with 8 tones createSequence(random between 10 and 12,8)

allseq = cell(4,1); %where the 20 simple sequences are stored (to be used for modality A and B)
allseq2 = cell(4,1);
for i = 1:5
   allseq{1}{i}= repmat(balseq(2),1,10+ceil(3.*rand(1))-1); % random iteration between 10 and 12
   allseq{2}{i}= repmat(balseq(4),1,10+ceil(3.*rand(1))-1);
   allseq{3}{i}= repmat(balseq(6),1,10+ceil(3.*rand(1))-1);
   allseq{4}{i}= repmat(createSequence(1,8),1,10+ceil(3.*rand(1))-1);
   allseq2{1}{i}= mappingspeakers(allseq{1}{i}, speak2tone);
   allseq2{2}{i}= mappingspeakers(allseq{2}{i}, speak2tone);
   allseq2{3}{i}= mappingspeakers(allseq{3}{i}, speak2tone);
   allseq2{4}{i}= mappingspeakers(allseq{4}{i}, speak2tone);
end

% assigning given the previous organization of the block
% find the index
% modality A
idx1 = (block==1);
idx2 = (block==11);
idx3 = (block==21);
idx4 = (block==31);
% modality B
idx1b = find(block==2);
idx2b = find(block==12);
idx3b = find(block==22);
idx4b = find(block==32);
%shuffle inside these indexes for modality B to avoid repetition from A
idx1b = idx1b(randperm(5));
idx2b = idx2b(randperm(5));
idx3b = idx3b(randperm(5));
idx4b = idx4b(randperm(5));
% modality C
idx5 = find(block==33);

% assigning the sequence given the index
cond.sequence(idx1)=allseq{1};
cond.sequence(idx2)=allseq{2};
cond.sequence(idx3)=allseq{3};
cond.sequence(idx4)=allseq{4};
cond.sequence(idx1b)=allseq{1};
cond.sequence(idx2b)=allseq{2};
cond.sequence(idx3b)=allseq{3};
cond.sequence(idx4b)=allseq{4};
cond.tonesequence(idx1)=allseq2{1};
cond.tonesequence(idx2)=allseq2{2};
cond.tonesequence(idx3)=allseq2{3};
cond.tonesequence(idx4)=allseq2{4};
cond.tonesequence(idx1b)=allseq2{1};
cond.tonesequence(idx2b)=allseq2{2};
cond.tonesequence(idx3b)=allseq2{3};
cond.tonesequence(idx4b)=allseq2{4};

for j = idx5
    cond.sequence{j}=createSequence(10+ceil(3.*rand(1))-1,8);
    cond.tonesequence{j}=mappingspeakers(cond.sequence{j},speak2tone);
end

for i=1:45
cond.target(i) = cond.sequence{i}(cond.target(i)); %cond.target(i) = cond.tonesequence{i}(cond.target(i));
cond.iterations(i) = length(cond.sequence{i})/cond.seqsize(i);
end

cond.preplaySeq = ~strcmp('C', cond.trialcondition);

end

%% n has to be even, it's the length of the whole sequence
% balanced choice from tones equally reparted between left and right
% speakers
function seq = balseq(n)
seq = zeros(1,n);

%left = [1 2 3 4]; %left side of the subject
%right = [5 6 7 8]; %right side of the subject

idxl = randperm(4);
seq(1:n/2) = idxl(1:n/2);
idxr = randperm(4)+4;
seq(n/2+1:end) = idxr(1:n/2);

seq = seq(randperm(n));
end

function speakseq = mappingspeakers(seq, speak2tone)
speakseq = speak2tone(seq);
end
