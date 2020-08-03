function dat= online_variance(dat, buffer_length, var_length, varargin)
%[cnt,state]= online_variance(cnt, state, buffer_length, var_length)
%
% calculate the variance of cnt online.
%
% IN   cnt           - data structure of continuous data
%      buffer_length - length of the internal buffer (in ms).
%      var_length    - max. length of the variance to calculate (in ms).
%
% OUT  cnt    - updated data structure
%

% kraulem 03/05

persistent state

if ~isstruct(dat);
     % buffer needs space for all new packages.

    state.buffer = zeros(buffer_length,dat);
    state.h_pointer = 1;
    state.l_pointer = ones(1,length(var_length));
    state.mu=zeros(length(var_length),dat);
    state.sigma=zeros(length(var_length),dat);
    return;
end 

var_length= ceil(var_length/1000*dat.fs);
buffer_length=max(buffer_length,2*max(var_length));


% store the new package
state.buffer(mod((state.h_pointer-1):(state.h_pointer+size(dat.x,1)-2),buffer_length)+1,:) = dat.x;
for i = 1:length(var_length)
    % set the pointers to new positions, depending on the filling status of the
    % buffer.
    if state.l_pointer(i)>state.h_pointer
     state.h_pointer=state.h_pointer+buffer_length;
    end 
    h_n = state.h_pointer+size(dat.x,1);
    l_n = max(h_n-var_length(i),1);
    o_len = state.h_pointer-state.l_pointer(i);
    n_len = h_n-l_n;

    % update the mean mu:
    mu_old = state.mu(i,:);
    mu_1 = sum(state.buffer(mod((state.l_pointer(i)-1):(l_n-2),buffer_length)+1,:),1);
    mu_2 = sum(state.buffer(mod((state.h_pointer-1):(h_n-2),buffer_length)+1,:),1);
    state.mu(i,:) = (o_len/n_len)*state.mu(i,:) - (1/n_len)*mu_1 + (1/n_len)*mu_2;

    % update the variance sigma:
    sigma_1 = state.buffer(mod((state.l_pointer(i)-1):(l_n-2),buffer_length)+1,:)-repmat(mu_old,l_n-state.l_pointer(i), 1);
    sigma_1 = sum(sigma_1.^2,1);
    sigma_2 = state.buffer(mod((state.h_pointer-1):(h_n-2),buffer_length)+1,:)-repmat(mu_old,h_n-state.h_pointer, 1);
    sigma_2 = sum(sigma_2.^2,1);
    state.sigma(i,:) = (o_len/n_len)*state.sigma(i,:) + (1/n_len)*(-sigma_1+sigma_2)-(state.mu(i,:)-mu_old).^2;

   % update the pointer
    state.l_pointer(i) = mod(l_n-1,buffer_length)+1;
    state.h_pointer = mod(state.h_pointer-1,buffer_length)+1;
end
% update the pointers
state.h_pointer = mod(h_n-1,buffer_length)+1;
 
% replace the content of dat:
dat.x = state.sigma;
