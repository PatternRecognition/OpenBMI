function out = grandAverage_prototye(cell_struct, bool_arith)
%% Description
% cell of averaged data
numData = length(cell_struct);

if nargin == 1
    bool_arith = false;
end

for i = 1:numData
    dat = cell_struct{i};

    if isequal(dat.class{1,2}, 'sgnr^2')
        dat.x = atanh(sqrt(abs(dat.x)).*sign(dat.x));
    end
    cell_struct{i} = dat;
end
[time, trials, channels] = size(cell_struct{1}.x);
dat_av = zeros(time, trials, channels);

ncls = size(cell_struct{1}.class, 1);
for cls = 1:ncls
    sW = 0;
    swV = 0;
    for v = 1:numData
        if bool_arith %% options-> Arithmetic or WeightedMean
            W = 1./cell_struct{v}.se(:, cls,:).^2;
        else 
            % Arithmetic
            W = 1;
        end
        sW = sW + W;
        swV = swV + W.^2.*cell_struct{v}.se(:, cls, :).^2;
        dat_av(:, cls, :) = dat_av(:, cls,:) + W.*cell_struct{v}.x(:,cls,:);
    end
    dat_av(:, cls,:) = dat_av(:, cls,:)./sW;
    se(:, cls,:) = sqrt(swV)./sW;
end

if isequal(dat.class{1,2}, 'sgnr^2')
    dat_av = tanh(dat_av).*abs(tanh(dat_av));
end

out = cell_struct{1};
out.x = dat_av;
out.se = se;
end
