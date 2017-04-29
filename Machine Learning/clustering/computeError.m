function [e]= computeError(data,mean)
%Computing Error
    %disp(mean);
    s=0;
    X=data(:,1:end-1);
    for f=1:size(X,1)
        j=data(f,end);
        ed=X(f,:)-mean(j,:);
        %disp(mean(j,:));
        ed=ed.^2;
        ed=sum(ed,2);
        ed=sqrt(ed);
        s=s+ed;
        %disp(s);
    end
    e=s;

end