function output = ssvep_cca_analysis(SMT, varargin)
in = varargin{:};
in = opt_cellToStruct(in);
dat = SMT;

nClasses = size(in.marker,1);
freq = in.freq;

t= 0:1/in.fs:in.time;
Y = cell(1,5);
r = cell(1,5);

for i = 1:nClasses
    ref = 2*pi*freq(i)*t;
    Y{i} = [sin(ref);cos(ref);sin(ref*2);cos(ref*2)];
end

for j = 1:nClasses
    [~,~,r{j}] = canoncorr(dat,Y{j}');
    r{j} = max(r{j});
end
output = cell2mat(r);
