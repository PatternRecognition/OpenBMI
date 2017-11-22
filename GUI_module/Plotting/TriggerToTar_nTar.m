function cnt_tar = TriggerToTar_nTar(cnt,spellerText_on)

%% tranfer triger to target/non-target
cnt_tar=cnt;

spell_char = {'A', 'B', 'C', 'D', 'E', 'F', ...
    'G', 'H', 'I', 'J', 'K', 'L', ...
    'M', 'N', 'O', 'P', 'Q', 'R', ...
    'S', 'T', 'U', 'V', 'W', 'X', ...
    'Y', 'Z', '1', '2', '3', '4', ...
    '5', '6', '7', '8', '9', '_'};

load 'random_cell_order.mat';       % sequence of flickling

% indexes
seq=5;
n_text=length(spellerText_on);
n_trig=12;
index=1;
y_class=cell(1,n_text*seq*n_trig);
for i=1:n_text  % 36
for j=1:seq     % 5 (36*5=180)
for k=1:n_trig  % 12 (36*5*12=2160)
    if sum(ismember([spell_char{cell_order{1,j}(k,:)}],spellerText_on(i)))
        y_class{index}=1;
    else
        y_class{index}=2;
    end
    index=index+1;
end
end
end
y_logic(1,:)=([y_class{1,:}]==1);       % logic
y_logic(2,:)=([y_class{1,:}]==2);
y_dec=cell2mat(y_class);
cnt_tar.y_dec=y_dec;
cnt_tar.y_logic=y_logic;
cnt_tar.y_class=y_class;
cnt_tar.class={1,'target',; 2,'non-target'};

end

