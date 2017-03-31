function [gain] = information_gain(examples, A, threshold)

    T = examples(1:end,end);
    
    dist = histc(T(:),unique(T));
    dist = dist / size(T,1);
    eval = 0;
    for i = 1:size(dist)
        if dist(i) ~= 0
            eval = eval + (-dist(i) * (log(dist(i)) / log(2)));
        end
    end
    %disp(eval);
    leftdataCount = 0;
    rightdataCount = 0;
    cr= 1;
    cl= 1;
    leftdata = [];
    rightdata = [];
    for i = 1:size(examples,1)
        if examples(i,A) >= threshold
            rightdataCount = rightdataCount + 1;
            rightdata(cr) = examples(i,end);
            cr = cr + 1;
        else
            leftdataCount = leftdataCount + 1;
            leftdata(cl) = examples(i,end);
            cl = cl + 1;
        end
    end    
    if isempty(leftdata)
        leftc=0;
        leftdata(1) = 0;
    else
        leftc = size(leftdata,2);
        leftdata = histc(leftdata(:),unique(leftdata));
        leftdata = leftdata / leftc;
    end
    if isempty(rightdata)
        rightc=0;
        rightdata(1) = 0;
    else
        rightc = size(rightdata,2);
        rightdata = histc(rightdata(:),unique(rightdata));    
    rightdata = rightdata / rightc;
    end
    
    hlval = 0;
    hrval = 0;
    %for left subtree
    for i = 1:size(leftdata,1)
        if leftdata(i) ~= 0
            hlval = hlval + (-leftdata(i) * (log2(leftdata(i))));            
        end
    end
    
    % for right subtree
    for i = 1:size(rightdata,1)
        if rightdata(i) ~= 0
            hrval = hrval + (-rightdata(i) * (log(rightdata(i)) / log(2)));
        end
    end
    
    hlval = (leftc / size(examples,1)) * hlval;
    hrval = (rightc / size(examples,1)) * hrval;
    gain = eval - hlval - hrval;    
end