function testinplace
% function testinplace()
% Install, tes, and benchmark inplace
% Correct bug: 29-Jun-2009

% Not yet installed
[p f ext] = fileparts(which('inplacearray'));
if strcmp(ext,'.m')
    % compile Inplace
    InplaceArray_install();
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Avoid data copy
% Could be faster, and memory saving is precious when manipulating
% large arrays

a=rand(1e5,100);
b = zeros(1,size(a,2));

fprintf('\nWithout inplace\n')
tic
for j=1:size(a,2)
    b(j) = mean(a(:,j));
end
toc % 0.014517 seconds.

fprintf('\nWith inplace\n')
tic
for j=1:size(a,2)
    aj = inplacecolumn(a,j);
    b(j) = mean(aj);
    releaseinplace(aj);
end
toc % 0.006115 seconds.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change an element of matrix without duplicating
a = zeros(2e3);

fprintf('\nWithout inplace\n')
tic
b = a;
b(11:20) = pi;
% do something with b
toc % 0.038104 seconds.

fprintf('\nWith inplace\n')
tic
a10 = inplacearray(a,10,10,1);
backupa10 = a10(1:end); % copy data
a10(1:10) = pi;
% do something with a
a10(1:end) = backupa10(:); % restore
releaseinplace(a10);
toc % 0.000106 seconds.

% 