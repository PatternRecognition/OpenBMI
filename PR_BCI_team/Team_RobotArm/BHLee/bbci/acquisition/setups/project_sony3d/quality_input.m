function q = quality_input()

while true
    user_entry = input('Please enter a quality score between 1 (best) and 3 (worst): ','s');
    if isempty(user_entry)
        user_entry = 'x';
    end
    user_entry = user_entry(end);
    if any(user_entry == '123')
        break
    end
end
    
q = str2double(user_entry);

