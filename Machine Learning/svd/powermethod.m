function[ud]=powermethod(A,iterations)
dimensions=size(A,1);
%b = randi([0 1], 1,dimension);
b=zeros(1,dimensions);
b=b+1;
%b = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0 0.1 0.2 0.3 0.5 0.6];
%b=[1 1 1 1 1 1 1 1];
b=transpose(b);
%disp(size(b));
for i=1:iterations
    b=(A*b)/norm(A*b);
    %disp(norm(A*b));
    %disp(b);
end

ud=b;
%disp(ud);
end