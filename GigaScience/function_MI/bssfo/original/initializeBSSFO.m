function initBSSFO = initializeBSSFO( initSamplingMode, numBands, interval, bSampleFromMoG )

if bSampleFromMoG == 1
    initBSSFO.numBands = numBands;
    % Generate samples from a mixture of Gaussian
    N = numBands;
    stdv = 25;
    numMix = 2;
    mu = cat(2 , [8; 13] , [14; 30]);                %(d x 1 x M)
    mu = reshape( mu, [2, 1, numMix] );
    sigma = cat(2 , [stdv 0; 0 stdv] , [stdv 0; 0 stdv] );   %(d x d x M)
    sigma = reshape( sigma, [2 2 numMix] );
    p = cat(2 , [0.5] , [0.5]);                       %(1 x 1 x M)
    p = reshape( p, [1 1 numMix] );
    [initBSSFO.sample, index] = sample_mvgm(N , mu , sigma , p);
    for i=1:initBSSFO.numBands
        initBSSFO.sample(:, i) = checkValidity( initBSSFO.sample(:, i) );
    end
else
    % Generate samples from a uniform distribution
    for i=1:initBSSFO.numBands
        if initSamplingMode == 0
            initBSSFO.sample(1, i) = lowFreq + (i-1)*interval;
            initBSSFO.sample(2, i) = lowFreq + i*interval;
        elseif initSamplingMode == 1
            initBSSFO.sample(1, i) = lowFreq + (i-1)*interval;
            initBSSFO.sample(2, i) = initBSSFO.sample(1, i) + 4;
        else
            initBSSFO.sample(1, i) = rand*(highFreq-lowFreq) + lowFreq;
            initBSSFO.sample(2, i) = rand*(highFreq-lowFreq) + lowFreq;
            
            initBSSFO.sample(:, i) = checkValidity( initBSSFO.sample(:, i) );
        end
    end
end