function[best_attribute,best_threshold,max_gain]=chooseAttribute(examples, attributes,option)
max_gain=-1;
best_attribute = -1;
best_threshold = -1;
if isequal(option,'optimized')
    for j=1:size(attributes,1)
        data=double(examples(:,j));
    %disp(data);
        l=min(data);
        m=max(data);
        for k=1:50
            threshold=l + k*(m-l)/51;        
            gain=information_gain(examples, j, threshold);        
            %disp(gain)
            %disp(threshold)
            %disp(k)        
            if gain>max_gain           
                max_gain=gain;
                best_attribute=j;           
                best_threshold=threshold;           
            end
        end
    end
elseif isequal(option,'randomized')
    j = datasample(attributes,1,'Replace',true);
    data=double(examples(:,j));
    %disp(data);
    l=min(data);
    m=max(data);
        for k=1:50
            threshold=l + k*(m-l)/51;        
            gain=information_gain(examples, j, threshold);        
            %disp(gain)
            %disp(threshold)
            %disp(k)        
            if gain>max_gain           
                max_gain=gain;
                best_attribute=j;           
                best_threshold=threshold;           
            end
        end
    
    %disp('randomized');
else
    %disp('forest3');
%disp(best_attribute);
%disp(best_threshold);
end
end