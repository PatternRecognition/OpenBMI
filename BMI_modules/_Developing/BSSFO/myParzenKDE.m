function estimatedDensity = myParzenKDE( trainData, testData, windowWidth )
% estimatedDensity = myParzenKDE(trainingData,testingData,windowWidth)
numFeatures = size( trainData, 2 );
covariance = estimateCovariance( trainData );
% covariance = cov( trainData );

trainDataSize = size( trainData, 1 );
testDataSize = size( testData, 1 );
estimatedDensity = zeros( testDataSize, 1 );

for i=1:testDataSize
    x = testData(i, :);
    testSampleMatrix = ones(trainDataSize,1)*x;
    
    new_diff = testSampleMatrix - trainData;
    
    for k=1:numFeatures
        new_diff( abs(new_diff(:,k))>windowWidth, k ) = 10000000000; %big number;
    end
    
    estimatedDensity(i) = mean( (1/(windowWidth^numFeatures)) * ...
        mvnpdf((new_diff/windowWidth), zeros(1,numFeatures), covariance) );
end

%
% So, it is using            1/(windowwidth^d) * Normal(0, \sigma_j / window_width)
% as the window function for class j, instead of the spherical parzen window function
%
function covariance = estimateCovariance( samples )
numFeatures = size( samples, 2 );
sigma = zeros( numFeatures, numFeatures );
for i=1:numFeatures
    covariance(i,i) = var( samples(:,i) );
end
