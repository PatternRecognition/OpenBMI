function res = maketodouble(varargin);

n = length(varargin);

res = n;

for i = 1:n
  A = varargin{i};
  res = [res,make2double(A)];
end






function res = make2double(A);

if islogical(A)
  res = log2double(A);
elseif isnumeric(A)
  res = num2double(A);
elseif ischar(A)
  res = char2double(A);
elseif iscell(A)
  res = cell2double(A);
elseif isstruct(A)
  res = struct2double(A);
else 
  error('format of varargin not assisted\n');
end



function res = num2double(A);


res = [1,length(size(A)),size(A),transpose(A(:))];



function res = log2double(A);


res = [0,length(size(A)),size(A),transpose(double(A(:)))];


function res = char2double(A);

res = [2,length(size(A)),size(A),transpose(double(A(:)))];


function res = cell2double(A);

res = [3,length(size(A)),size(A)];

for i = 1:prod(size(A))
  res = [res,make2double(A{i})];
end


function res = struct2double(A);

field = fieldnames(A);
res = [4,length(size(A)),size(A),length(field)];

for i = 1:length(field)
  res = [res,char2double(field{i})];
  for j = 1:prod(size(A))
    res = [res,make2double(getfield(A(j),field{i}))];
  end
end



  
