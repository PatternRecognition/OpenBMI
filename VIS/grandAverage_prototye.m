function out = grandAverage_prototye(cell_struct, bool_arith)
%% Description
% cell of averaged data
% 단순히 subject 별로 평균내고 평균낸걸로 다시 평균만 낼꺼야
% trial이 다른건 중요하지 않아

numData = length(cell_struct);

%%
% intersecting the channels

%% 
% for i = 1:numData
%     dat = cell_struct{i};
%     dat.x = permute(dat.x, [1 3 2]);
%     dat.se = permute(dat.se, [1 3 2]);
%     [time, channels, trials] = size(dat.x);
%     dat.x = reshape(dat.x, time*channels, trials);
%     dat.se = reshape(dat.se, time*channels, trials);
%     if isequal(dat.class{1,2}, 'sgnr^2')
%         dat.x = atanh(sqrt(abs(dat.x)).*sign(dat.x));
%     end
%     cell_struct{i} = dat;
% end
% 
% dat_av = zeros(time*channels, trials);
% 
% ncls = size(cell_struct{1}.class, 1);
% for cls = 1:ncls
%     sW = 0;
%     swV = 0;
%     for v = 1:numData
%         if true
%             W = 1./cell_struct{v}.se(:, cls).^2;
%             %W = 1./cell_struct{v}.se(:, cls,:).^2;
%         else 
%             W = 1;
%         end
%         sW = sW + W;
%         swV = swV + W.^2.*cell_struct{v}.se(:, cls).^2;
% %         swV = swV + W.^2.*cell_struct{v}.se(:, cls, :).^2;
%         dat_av(:, cls) = dat_av(:, cls) + W.*cell_struct{v}.x(:,cls);
%     end
%     dat_av(:, cls) = dat_av(:, cls)./sW;
%     se(:, cls) = sqrt(swV)./sW;
% end

if nargin == 1
    bool_arith = false;
end

for i = 1:numData
    dat = cell_struct{i};
%     dat.x = permute(dat.x, [1 3 2]);
%     dat.se = permute(dat.se, [1 3 2]);
%     [time, channels, trials] = size(dat.x);
%     dat.x = reshape(dat.x, time*channels, trials);
%     dat.se = reshape(dat.se, time*channels, trials);
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

% dat_av = reshape(dat_av, time, channels, trials);
% se = reshape(se, time, channels, trials);
if isequal(dat.class{1,2}, 'sgnr^2')
    dat_av = tanh(dat_av).*abs(tanh(dat_av));
end
% dat_av = permute(dat_av, [1 3 2]);
% se = permute(se, [1 3 2]);
out = cell_struct{1};
out.x = dat_av;
out.se = se;
end
