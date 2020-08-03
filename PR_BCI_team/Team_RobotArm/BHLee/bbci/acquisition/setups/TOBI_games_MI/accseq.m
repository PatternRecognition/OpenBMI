function tnt = accseq(varargin)
% this function gives an ACCEPTABLE sequence of T/NT tones
% if there are less than 2 NT tones between 2 T tones, or if we have less
% than 12 T tones, we generate a new sequence
% varargin:
v = varargin;
tnt = genseq(v{:});

while (any(diff(find(tnt))<=2)||(sum(tnt)<12))
    tnt = genseq(v{:});
end

return
end

function tnt = genseq(varargin)
% this function gives a sequence of T/NT tones
ntarget = 16;
rapport = 6;
std = 2;
if (nargin~=0)
    switch nargin
        case 1
            ntarget = varargin{1};
        case 2
            ntarget = varargin{1};
            rapport = varargin{2};
        case 3
            ntarget = varargin{1};
            rapport = varargin{2};
            std = varargin{3};
    end
end
prentone = ceil(4.*rand()-1); % number of tones in the prelude
ntone = (ntarget + prentone)*rapport; % number of T and NT tones
tnt = zeros(ntone,1); % our T/NT sequence (ex 0 0 0 1 0 0 0 0 0 1 0 0...)

for i=1:(ntarget+prentone)

    mean = i * rapport;
    dev = round(mean + std*randn()); % repartition of the target tones around the 6th (and multiples) tone by a normal distribution

    while (dev > length(tnt))||(dev <= 0) % there might be problems at the beginning or at the end
        dev = round(mean + std*randn());
    end

    tnt( dev ) = 1;

end

return

end